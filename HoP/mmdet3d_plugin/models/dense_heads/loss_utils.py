from matplotlib.pyplot import autoscale
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import build_loss

import pdb
from mmcv.runner import auto_fp16, force_fp32


class BinarySegmentationLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(BinarySegmentationLoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)

        return loss


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, label):
        # Implementation from
        # Translating Images to Maps, Saha et al., ICRA
        # https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/loss.py#L261-L272
        label = label.float()
        intersection = 2 * pred * label
        union = pred + label
        iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1)) / (
            union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
        )
        loss_mean = 1 - iou.mean()

        return loss_mean

class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False,
                 top_k_ratio=1.0, future_discount=1.0):

        super().__init__()

        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

        # self.ce_criterion = nn.CrossEntropyLoss(
        #     weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

        # self.nll_criterion = nn.NLLLoss(
        #     weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape
        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)

        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=self.class_weights.to(target.device).float(),
        )

        # ce_loss = self.ce_criterion(prediction, target)
        # pred_logsoftmax = F.log_softmax(prediction)
        # loss = self.nll_criterion(pred_logsoftmax, target)

        loss = loss.view(b, s, h, w)
        future_discounts = self.future_discount ** torch.arange(
            s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s, 1, 1)
        loss = loss * future_discounts.float()

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)

    def __repr__(self):
        return "Cross Entropy(Top_k= {}, Top_k_ratio= {:.2f})".format(self.use_top_k, self.top_k_ratio)

class MotionSegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False,
                 top_k_ratio=1.0, future_discount=1.0):

        super().__init__()

        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

    def forward(self, prediction, target, frame_mask=None):
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)

        future_discounts = self.future_discount ** torch.arange(
            s).type_as(prediction)
        future_discounts = future_discounts.view(1, s).repeat(b, 1)
        future_discounts = future_discounts.view(-1, 1)

        frame_mask = frame_mask.contiguous().view(-1)
        valid_prediction = prediction[frame_mask]
        valid_target = target[frame_mask]
        future_discounts = future_discounts[frame_mask]

        if frame_mask.sum().item() == 0:
            return prediction.abs().sum().float() * 0.0

        loss = F.cross_entropy(
            valid_prediction,
            valid_target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )
        loss = loss.flatten(start_dim=1)
        loss *= future_discounts

        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[1])
            loss, _ = torch.sort(loss, dim=1, descending=True)
            loss = loss[:, :k]

        return torch.mean(loss)
