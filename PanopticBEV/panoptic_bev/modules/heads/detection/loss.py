import torch
import torch.nn as nn
import torch.nn.functional as F
from panoptic_bev.helpers.more_util import zero_tensor_like


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

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

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 1, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 1])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 1])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            loss = F.binary_cross_entropy(pred.masked_select(mask),
                                          target.masked_select(mask),
                                          reduction='mean')
            return loss
        else:
            return zero_tensor_like(output)

def laplacian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    From https://github.com/abhi1kumar/DEVIANT/blob/main/code/lib/losses/uncertainty_loss.py#L4-L12
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + 0.5*log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()

class OffSmoothL1Loss(nn.Module):
    def __init__(self, use_uncertainty= False, use_per_class_uncertainty= False, num_detector_classes= 2):
        super(OffSmoothL1Loss, self).__init__()
        self.use_uncertainty = use_uncertainty
        self.use_per_class_uncertainty = use_per_class_uncertainty
        self.num_detector_classes = num_detector_classes

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

    def forward(self, output, mask, ind, target, cls_id= None):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            if self.use_uncertainty:
                if self.use_per_class_uncertainty:
                    # Half are predictions and h
                    unc_start_index = pred.shape[2] - self.num_detector_classes
                    mask_for_cls_id = mask[:, :,  0].unsqueeze(2)
                    mask_for_pred   = mask[:, :, :unc_start_index]
                    pred_masked     = pred[:, :, :unc_start_index].masked_select(mask_for_pred)   # d*boxes
                    target_masked   = target                      .masked_select(mask_for_pred)   # d*boxes
                    cls_id_masked   = cls_id.unsqueeze(2)         .masked_select(mask_for_cls_id) # boxes
                    unc_all_cls     = pred[:, :, unc_start_index:].masked_select(mask_for_pred)
                    # unc_all_cls is row-major format
                    # Sanity Check:
                    # test = torch.zeros((3, 50, 4)).float()
                    # test[:, :, 0] = 0
                    # test[:, :, 1] = 1
                    # test[:, :, 2] = 2
                    # test[:, :, 3] = 3
                    # test.cuda().masked_select(mask_for_pred.repeat(1, 1, 2))
                    #tensor([0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1.,
                    #        2., 3.], device='cuda:0')
                    unc_all_cls = unc_all_cls.reshape(-1, self.num_detector_classes)     # boxes x num_classes
                    # Gather [0, cls_id_masked[0]], [1, cls_id_masked[1]], ... entries
                    unc_masked  = unc_all_cls.gather(1, cls_id_masked.unsqueeze(1))      # boxes
                    unc_index   = unc_start_index
                else:
                    unc_index      = pred.shape[2]-1
                    mask_for_pred   = mask[:, :, :unc_index]
                    mask_for_unc    = mask[:, :,  unc_index].unsqueeze(2)
                    pred_masked     = pred[:, :, :unc_index].masked_select(mask_for_pred) # 12
                    target_masked   = target                .masked_select(mask_for_pred) # 12
                    unc_masked      = pred[:, :,  unc_index].unsqueeze(2).masked_select(mask_for_unc)  # 6
                unc_masked      = unc_masked.reshape(-1, 1).repeat(1, unc_index).reshape(-1) # 12
                loss = laplacian_aleatoric_uncertainty_loss(input= pred_masked, target= target_masked, log_variance= unc_masked)
            else:
                loss = F.smooth_l1_loss(pred.masked_select(mask),
                                        target.masked_select(mask),
                                        reduction='mean')
            return loss
        else:
            return zero_tensor_like(output)

class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()

      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss

def isnan(x):
    return x != x

  
class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        self.L_hm = FocalLoss()
        self.L_wh =  OffSmoothL1Loss()
        self.L_off = OffSmoothL1Loss()
        self.L_cls_theta = BCELoss()

    def forward(self, pr_decs, gt_batch):
        hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        ## add
        cls_theta_loss = self.L_cls_theta(pr_decs['cls_theta'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['cls_theta'])

        if isnan(hm_loss) or isnan(wh_loss) or isnan(off_loss):
            print('hm loss is {}'.format(hm_loss))
            print('wh loss is {}'.format(wh_loss))
            print('off loss is {}'.format(off_loss))

        # print(hm_loss)
        # print(wh_loss)
        # print(off_loss)
        # print(cls_theta_loss)
        # print('-----------------')

        loss_all = {}
        loss_all['hm_loss'] = hm_loss
        loss_all['off_loss']= off_loss
        loss_all['wh_loss'] = wh_loss
        loss_all['cls_theta_loss'] = cls_theta_loss
        loss_all['total'] =  hm_loss + wh_loss + off_loss + cls_theta_loss
        return loss_all
