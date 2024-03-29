import torch.nn.functional as F
import torch

class DecDecoder(object):
    def __init__(self, K, conf_thresh, num_classes, use_uncertainty= False, use_per_class_uncertainty= False):
        self.K = K
        self.conf_thresh = conf_thresh
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty
        self.use_per_class_uncertainty = use_per_class_uncertainty

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_clses = (topk_ind // self.K).int()
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def ctdet_decode(self, pr_decs):
        heat = pr_decs['hm']
        wh = pr_decs['wh']
        reg = pr_decs['reg']
        cls_theta = pr_decs['cls_theta']
        h3d = pr_decs['h3d']
        y3d = pr_decs['y3d']
        yaw = pr_decs['yaw']

        batch, c, height, width = heat.size()
        heat = self._nms(heat)

        scores, inds, clses, ys, xs = self._topk(heat)
        reg = self._tranpose_and_gather_feat(reg, inds)
        if self.use_uncertainty:
            reg = reg.view(batch, self.K, -1)
            if self.use_per_class_uncertainty:
                xz_unc_all_cls = reg[:, :, 2:].reshape(batch*self.K, -1)
                # Gather [0, clses[0]], [1, clses[1]], ... entries
                xz_unc         = xz_unc_all_cls.gather(1, clses.reshape(batch*self.K, 1).type(torch.int64))      # boxes
                xz_unc         = xz_unc.reshape(batch, self.K, 1)
            else:
                xz_unc = reg[:, :, 2].unsqueeze(2)
            # See https://github.com/abhi1kumar/DEVIANT/blob/main/code/lib/helpers/decode_helper.py#L81
            xz_score = (-(0.5*xz_unc).exp()).exp()
        else:
            reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        clses = clses.view(batch, self.K, 1).float()
        scores = scores.view(batch, self.K, 1)
        if self.use_uncertainty:
            scores = scores * xz_score
        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, self.K, 10)
        # add
        cls_theta = self._tranpose_and_gather_feat(cls_theta, inds)
        cls_theta = cls_theta.view(batch, self.K, 1)
        # h3d, y3d, yaw
        h3d = self._tranpose_and_gather_feat(h3d, inds)
        h3d = h3d.view(batch, self.K, 1)
        y3d = self._tranpose_and_gather_feat(y3d, inds)
        y3d = y3d.view(batch, self.K, 1)
        yaw = self._tranpose_and_gather_feat(yaw, inds)
        yaw = yaw.view(batch, self.K, 1)
        # Mask
        mask = (cls_theta>0.8).float().view(batch, self.K, 1)
        #
        tt_x = (xs+wh[..., 0:1])*mask + (xs)*(1.-mask)
        tt_y = (ys+wh[..., 1:2])*mask + (ys-wh[..., 9:10]/2)*(1.-mask)
        rr_x = (xs+wh[..., 2:3])*mask + (xs+wh[..., 8:9]/2)*(1.-mask)
        rr_y = (ys+wh[..., 3:4])*mask + (ys)*(1.-mask)
        bb_x = (xs+wh[..., 4:5])*mask + (xs)*(1.-mask)
        bb_y = (ys+wh[..., 5:6])*mask + (ys+wh[..., 9:10]/2)*(1.-mask)
        ll_x = (xs+wh[..., 6:7])*mask + (xs-wh[..., 8:9]/2)*(1.-mask)
        ll_y = (ys+wh[..., 7:8])*mask + (ys)*(1.-mask)
        #
        detections = torch.cat([xs,                      # cen_x
                                ys,                      # cen_y
                                tt_x,
                                tt_y,
                                rr_x,
                                rr_y,
                                bb_x,
                                bb_y,
                                ll_x,
                                ll_y,
                                scores,
                                clses,
                                h3d,
                                y3d,
                                yaw
                                ],
                               dim=2)

        # index = (scores>self.conf_thresh).squeeze(0).squeeze(1)
        # detections = detections[:,index,:]
        final_detections = []
        for j in range(batch):
            index = scores[j].flatten() > self.conf_thresh
            final_detections.append(detections[j][index].cpu().float().numpy())
        return final_detections