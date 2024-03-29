from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from panoptic_bev.utils.sequence import pad_packed_images
from panoptic_bev.helpers.more_util import packedSeq_to_tensor, clip_sigmoid, per_class_to_single_occupancy
from panoptic_bev.modules.heads.detection import loss as det_loss
from panoptic_bev.modules.heads.detection import func_utils

class PanopticBevNet(nn.Module):
    def __init__(self,
                 body,
                 transformer,
                 rpn_head,
                 roi_head,
                 sem_head,
                 transformer_algo,
                 rpn_algo,
                 inst_algo,
                 sem_algo,
                 po_fusion_algo,
                 dataset,
                 denoise_module=None,
                 det_head=None,
                 det_decoder=None,
                 app_feat=None,
                 classes=None,
                 front_vertical_classes=None,  # In the frontal view
                 front_flat_classes=None,  # In the frontal view
                 bev_vertical_classes=None,  # In the BEV
                 bev_flat_classes=None):  # In the BEV
        super(PanopticBevNet, self).__init__()

        # Backbone
        self.body = body

        # Transformer
        self.transformer = transformer

        # Modules
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_head = sem_head
        self.det_head = det_head
        if self.det_head is not None:
            self.det_decoder = det_decoder
            self.num_det_classes = self.det_head.det_params['hm']
            self.det_classes     = self.det_head.det_classes
            self.app_feat        = app_feat
            self.use_uncertainty = False
            self.use_per_class_uncertainty = False
        self.denoise_module = denoise_module

        # Algorithms
        self.transformer_algo = transformer_algo
        self.rpn_algo = rpn_algo
        self.inst_algo = inst_algo
        self.sem_algo = sem_algo
        self.po_fusion_algo = po_fusion_algo

        # Params
        self.dataset = dataset
        self.num_stuff = classes["stuff"]
        self.front_vertical_classes = front_vertical_classes
        self.front_flat_classes = front_flat_classes
        self.bev_vertical_classes = bev_vertical_classes
        self.bev_flat_classes = bev_flat_classes

        self.debug = False

    def make_region_mask(self, msk):
        if (self.bev_vertical_classes is None) or (self.bev_flat_classes is None):
            return

        B = len(msk)
        W, Z = msk[0].shape[0], msk[0].shape[1]
        v_region_msk = torch.zeros((B, 1, W, Z), dtype=torch.long).to(msk[0].device)
        f_region_msk = torch.zeros((B, 1, W, Z), dtype=torch.long).to(msk[0].device)

        for b in range(B):
            for c in self.bev_vertical_classes:
                v_region_msk[b, 0, msk[b] == int(c)] = 1
            for c in self.bev_flat_classes:
                f_region_msk[b, 0, msk[b] == int(c)] = 1

        return v_region_msk, f_region_msk

    def make_vf_mask(self, msk):
        if (self.front_vertical_classes is None) or (self.front_flat_classes is None):
            return

        if msk is None:
            return None

        B = msk.shape[0]
        H, W = msk.shape[2], msk.shape[3]
        vf_msk = torch.ones((B, 1, H, W), dtype=torch.long).to(msk.device) * 2  # Everything is initially unknown

        sem_msk = msk.detach().clone()
        sem_msk[sem_msk >= 1000] = sem_msk[sem_msk >= 1000] // 1000

        for c in self.front_vertical_classes:
            vf_msk[sem_msk == int(c)] = 0
        for c in self.front_flat_classes:
            vf_msk[sem_msk == int(c)] = 1

        return vf_msk

    def prepare_inputs(self, msk, cat, iscrowd, bbx):
        cat_out, iscrowd_out, bbx_out, ids_out, sem_out, po_out, po_vis_out = [], [], [], [], [], [], []
        for msk_i, cat_i, iscrowd_i, bbx_i in zip(msk, cat, iscrowd, bbx):
            msk_i = msk_i.squeeze(0)
            thing = (cat_i >= self.num_stuff) & (cat_i != 255)
            valid = thing & ~(iscrowd_i > 0)

            if valid.any().item():
                cat_out.append(cat_i[valid])
                bbx_out.append(bbx_i[valid])
                ids_out.append(torch.nonzero(valid))
            else:
                cat_out.append(None)
                bbx_out.append(None)
                ids_out.append(None)

            if iscrowd_i.any().item():
                iscrowd_i = (iscrowd_i > 0) & thing
                iscrowd_out.append(iscrowd_i[msk_i].type(torch.uint8))
            else:
                iscrowd_out.append(None)

            sem_out.append(cat_i[msk_i])

            # Panoptic GT
            po_msk = torch.ones_like(msk_i) * 255
            po_msk_vis = torch.ones_like(msk_i) * 255
            inst_id = 0
            for lbl_idx in range(cat_i.shape[0]):
                if cat_i[lbl_idx] == 255:
                    continue
                if cat_i[lbl_idx] < self.num_stuff:
                    po_msk[msk_i == lbl_idx] = cat_i[lbl_idx]
                    po_msk_vis[msk_i == lbl_idx] = cat_i[lbl_idx]
                else:
                    po_msk[msk_i == lbl_idx] = self.num_stuff + inst_id
                    po_msk_vis[msk_i == lbl_idx] = (cat_i[lbl_idx] * 1000) + inst_id
                    inst_id += 1
            po_out.append(po_msk)
            po_vis_out.append(po_msk_vis)

        return cat_out, iscrowd_out, bbx_out, ids_out, sem_out, po_out, po_vis_out

    def forward(self, img, bev_msk=None, front_msk=None, weights_msk=None, cat=None, iscrowd=None, bbx=None, calib=None, sem_msk=None,
                hm= None, reg_mask=None, ind=None, cat_ind=None, wh=None, reg=None, cls_theta=None, h3d=None, y3d=None, yaw=None,
                do_loss=False, do_prediction=False):
        result = OrderedDict()
        loss = OrderedDict()
        stats = OrderedDict()

        # Get some parameters
        img, _ = pad_packed_images(img)

        if bev_msk is not None:
            bev_msk, valid_size = pad_packed_images(bev_msk)
            img_size = bev_msk.shape[-2:]
        else:
            valid_size = [torch.Size([896, 768])] * img.shape[0]
            img_size = torch.Size([896, 768])

        if front_msk is not None:
            front_msk, _ = pad_packed_images(front_msk)

        if sem_msk is not None:
            sem_msk, _ = pad_packed_images(sem_msk)

        calib, _ = pad_packed_images(calib)
        if do_loss:
            # Prepare the input data and the ground truth labels
            cat, iscrowd, bbx, ids, sem_gt, po_gt, po_gt_vis = self.prepare_inputs(bev_msk, cat, iscrowd, bbx)
            if self.dataset == "Kitti360":
                vf_mask_gt = [self.make_vf_mask(front_msk)]
            elif self.dataset == "nuScenes":
                vf_mask_gt = [front_msk]  # List to take care of the "rgb_cameras"
            v_region_mask_gt, f_region_mask_gt = self.make_region_mask(sem_gt)
            v_region_mask_gt = [v_region_mask_gt]
            f_region_mask_gt = [f_region_mask_gt]
            if sem_msk is not None:
                sem_msk_dice = torch.zeros((len(self.bev_vertical_classes), bev_msk.shape[0], bev_msk.shape[2], bev_msk.shape[3])).cuda()
                for j, cls_index in enumerate(self.bev_vertical_classes):
                    valid_index = sem_msk[:, 0] == cls_index
                    sem_msk_dice[j][valid_index] = 1.0
                sem_msk_dice = sem_msk_dice.permute(1, 0, 2, 3)
            else:
                sem_msk_dice = None
        else:
            po_gt, po_gt_vis = None, None

        # Get the image features
        ms_feat = self.body(img)

        # Transform from the front view to the BEV and upsample the height dimension
        ms_bev, vf_logits_list, v_region_logits_list, f_region_logits_list = self.transformer(ms_feat, calib)
        if do_loss:
            vf_loss, v_region_loss, f_region_loss = self.transformer_algo.training(vf_logits_list, v_region_logits_list,
                                                                                   f_region_logits_list, vf_mask_gt,
                                                                                   v_region_mask_gt, f_region_mask_gt)
        elif do_prediction:
            vf_loss, v_region_loss, f_region_loss = None, None, None
        else:
            vf_logits_list, ms_bev, vf_loss, v_region_loss, f_region_loss = None, None, None, None, None

        # # RPN Part
        # if do_loss:
        #     obj_loss, bbx_loss, proposals = self.rpn_algo.training(self.rpn_head, ms_bev, bbx, iscrowd, valid_size,
        #                                                            training=self.training, do_inference=True)
        # elif do_prediction:
        #     proposals = self.rpn_algo.inference(self.rpn_head, ms_bev, valid_size, self.training)
        #     obj_loss, bbx_loss = None, None
        # else:
        #     obj_loss, bbx_loss, proposals = None, None, None
        #
        # # ROI Part
        # if do_loss:
        #     roi_cls_loss, roi_bbx_loss, roi_msk_loss, roi_cls_logits, roi_bbx_logits, roi_msk_logits = \
        #         self.inst_algo.training(self.roi_head, ms_bev, proposals, bbx, cat, iscrowd, ids, bev_msk, img_size)
        # else:
        #     roi_cls_loss, roi_bbx_loss, roi_msk_loss = None, None, None
        #     roi_cls_logits, roi_bbx_logits, roi_msk_logits = None, None, None
        # if do_prediction:
        #     bbx_pred, cls_pred, obj_pred, msk_pred, roi_msk_logits = self.inst_algo.inference(self.roi_head, ms_bev,
        #                                                                                       proposals, valid_size,
        #                                                                                       img_size)
        # else:
        #     bbx_pred, cls_pred, obj_pred, msk_pred = None, None, None, None

        # Segmentation Part
        if do_loss:
            sem_loss, sem_conf_mat, sem_pred, sem_logits, sem_feat = self.sem_algo.training(self.sem_head, ms_bev,
                                                                                            sem_gt, bbx, valid_size,
                                                                                            img_size, weights_msk,
                                                                                            calib, sem_msk_dice)
        elif do_prediction:
            sem_pred, sem_logits, sem_feat = self.sem_algo.inference(self.sem_head, ms_bev, valid_size, img_size)
            sem_loss, sem_reg_loss, sem_conf_mat = None, None, None
        else:
            sem_loss, sem_reg_loss, sem_conf_mat, sem_pred, sem_logits, sem_feat = None, None, None, None, None, None

        # Detection Part
        if self.det_head is not None:
            logits      = packedSeq_to_tensor(sem_logits)
            bs, _, h, w = logits.shape

            per_class_occupancy     = clip_sigmoid(logits)
            if self.denoise_module:
                per_class_occupancy = self.denoise_module(per_class_occupancy)
            occupancy   = per_class_to_single_occupancy(per_class_occupancy, self.num_det_classes)

            if self.app_feat == "front":
                # Append occupancy and frontal features
                occupancy  = occupancy.repeat(1, 2, 1, 1)                                                # bs x 2 x 104 x 104
                front_feat = torch.max(ms_feat[0], dim= 1)[0].unsqueeze(1)                               # bs x 1 x 96 x 360
                front_feat = F.interpolate(front_feat, (h, w), mode='bilinear')                          # bs x 1 x 104 x 104
                occupancy  = torch.cat((occupancy, front_feat), dim= 1)                                  # bs x 3 x 104 x 104
            elif self.app_feat == "bev":
                # Append occupancy and bev features
                occupancy  = occupancy.repeat(1, 2, 1, 1)                                                # bs x 2 x 104 x 104
                bev_feat   = torch.max(ms_bev[0], dim= 1)[0].unsqueeze(1)                                # bs x 1 x 104 x 104
                bev_feat   = F.interpolate(bev_feat, (h, w), mode='bilinear')                            # bs x 1 x 104 x 104
                occupancy  = torch.cat((occupancy, bev_feat), dim= 1)                                    # bs x 3 x 104 x 104
            else:
                occupancy  = occupancy.repeat(1, 3, 1, 1)                                                # bs x 3 x 104 x 104

            # Rotate sem maps by 90 degrees since sem maps are sideways
            occupancy   = torch.rot90(occupancy, k=1, dims=[2,3])
            occupancy   = F.interpolate(occupancy, hm[0][0].cpu().clone().numpy().shape, mode='bilinear')
            if self.debug:
                from matplotlib import pyplot as plt
                plt.subplot(221)
                plt.imshow(occupancy[0][1].cpu().float().detach().numpy(), vmin= 0, vmax= 2)
                plt.subplot(222)
                plt.imshow(torch.max(hm[0], dim= 0)[0].cpu().float(), vmin= 0, vmax= 1)
                plt.subplot(223)
                plt.imshow(occupancy[1][1].cpu().float().detach().numpy(), vmin= 0, vmax= 2)
                plt.subplot(224)
                plt.imshow(torch.max(hm[1], dim= 0)[0].cpu().float(), vmin= 0, vmax= 1)
                plt.show()
                plt.close()
            pr_decs     = self.det_head(occupancy)

            L_hm        = det_loss.FocalLoss()
            L_off       = det_loss.OffSmoothL1Loss(use_uncertainty= self.use_uncertainty,
                                               use_per_class_uncertainty= self.use_per_class_uncertainty,
                                               num_detector_classes= self.num_det_classes)
            L_wh        = det_loss.OffSmoothL1Loss()
            L_h3d       = det_loss.OffSmoothL1Loss()
            L_y3d       = det_loss.OffSmoothL1Loss()
            L_yaw       = det_loss.OffSmoothL1Loss()
            L_cls_theta = det_loss.BCELoss()

            hm       = packedSeq_to_tensor(hm)
            reg_mask = packedSeq_to_tensor(reg_mask)
            ind      = packedSeq_to_tensor(ind)
            reg      = packedSeq_to_tensor(reg)
            cat_ind  = packedSeq_to_tensor(cat_ind)
            wh       = packedSeq_to_tensor(wh)
            h3d      = packedSeq_to_tensor(h3d)
            y3d      = packedSeq_to_tensor(y3d)
            cls_theta= packedSeq_to_tensor(cls_theta)
            det_hm_loss  = L_hm (pr_decs['hm'] , hm)
            det_off_loss = L_off(pr_decs['reg'], reg_mask, ind, reg, cat_ind)
            det_wh_loss  = L_wh (pr_decs['wh'] , reg_mask, ind, wh)
            det_h3d_loss = L_h3d(pr_decs['h3d'], reg_mask, ind, h3d)
            det_y3d_loss = L_y3d(pr_decs['y3d'], reg_mask, ind, y3d)
            det_cls_theta_loss = L_cls_theta(pr_decs['cls_theta'], reg_mask, ind, cls_theta)
            #det_yaw_loss = L_yaw(pr_decs['yaw'], data_dict['reg_mask'], data_dict['ind'], data_dict['yaw'])

            loss['det_hm_loss']        = det_hm_loss
            loss['det_off_loss']       = det_off_loss
            loss['det_wh_loss']        = det_wh_loss
            loss['det_h3d_loss']       = det_h3d_loss
            loss['det_y3d_loss']       = det_y3d_loss
            loss['det_cls_theta_loss'] = det_cls_theta_loss
            #loss['yaw_loss']      = yaw_loss

            if do_prediction:
                detections   = self.det_decoder.ctdet_decode(pr_decs)
                detect_lines_batch = []
                for j, pred in enumerate(detections):
                    detect_lines_image = []
                    if len(pred) > 0:
                        decoded_pts    = []
                        decoded_scores = []
                        decoded_boxes  = []
                        p2 = np.eye(4)
                        p2[:3, :3] = calib[j].cpu().float().numpy()
                        pts0, scores0, boxes0 = func_utils.decode_prediction(pred, self.det_classes, calib= p2)
                        decoded_pts   .append(pts0)
                        decoded_scores.append(scores0)
                        decoded_boxes .append(boxes0)
                        #nms
                        results = {cat:[] for cat in self.det_classes}
                        for cat in self.det_classes:
                            if cat == 'background':
                                continue
                            pts_cat    = []
                            scores_cat = []
                            boxes_cat  = []
                            for pts0, scores0, boxes0 in zip(decoded_pts, decoded_scores, decoded_boxes):
                                pts_cat   .extend(pts0   [cat])
                                scores_cat.extend(scores0[cat])
                                boxes_cat .extend(boxes0 [cat])
                            pts_cat    = np.asarray(pts_cat   , np.float32)
                            scores_cat = np.asarray(scores_cat, np.float32)
                            boxes_cat  = np.asarray(boxes_cat , np.float32)
                            if pts_cat.shape[0]:
                                _, keep_index = func_utils.non_maximum_suppression(pts_cat, scores_cat)
                                boxes_with_scores = np.hstack((boxes_cat, scores_cat[:, np.newaxis]))
                                nms_results = boxes_with_scores[keep_index]
                                results[cat].extend(nms_results)

                                for l in range(pts_cat.shape[0]):
                                    output_str = "{} {:.2f} {:1d} ".format(cat, -1, -1) +  " ".join(["{:.2f}".format(x) for x in boxes_with_scores[l].tolist()]) + "\n"
                                    detect_lines_image.append(output_str)
                    detect_lines_batch.append(detect_lines_image)

                result['detections'] = detect_lines_batch

        # Panoptic Fusion. Fuse the semantic and instance predictions to generate a coherent output
        # if do_prediction:
        #     # The first channel of po_pred contains the semantic labels
        #     # The second channel contains the instance masks with the instance label being the corresponding semantic label
        #     po_pred, po_loss, po_logits = self.po_fusion_algo.inference(sem_logits, roi_msk_logits, bbx_pred, cls_pred,
        #                                                                 img_size)
        # elif do_loss:
        #     po_loss = self.po_fusion_algo.training(sem_logits, roi_msk_logits, bbx, cat, po_gt, img_size)
        #     po_pred, po_logits = None, None
        # else:
        #     po_pred, po_loss, po_logits = None, None, None

        # Prepare outputs
        # LOSSES
        # loss['obj_loss'] = obj_loss
        # loss['bbx_loss'] = bbx_loss
        # loss['roi_cls_loss'] = roi_cls_loss
        # loss['roi_bbx_loss'] = roi_bbx_loss
        # loss['roi_msk_loss'] = roi_msk_loss
        loss['sem_loss'] = sem_loss
        loss['vf_loss'] = vf_loss
        loss['v_region_loss'] = v_region_loss
        loss['f_region_loss'] = f_region_loss
        # loss['po_loss'] = po_loss

        # PREDICTIONS
        # result['bbx_pred'] = bbx_pred
        # result['cls_pred'] = cls_pred
        # result['obj_pred'] = obj_pred
        # result['msk_pred'] = msk_pred
        result['sem_pred'] = sem_pred
        result['sem_logits'] = sem_logits
        result['vf_logits'] = vf_logits_list
        result['v_region_logits'] = v_region_logits_list
        result['f_region_logits'] = f_region_logits_list
        # if po_pred is not None:
        #     result['po_pred'] = po_pred[0]
        #     result['po_class'] = po_pred[1]
        #     result['po_iscrowd'] = po_pred[2]

        # STATS
        stats['sem_conf'] = sem_conf_mat

        return loss, result, stats
