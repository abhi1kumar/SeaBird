from math import ceil
import torch
import torch.nn.functional as functional

from panoptic_bev.utils.parallel import PackedSequence
from panoptic_bev.utils.sequence import pack_padded_images

class SemanticSegLoss:
    """Semantic segmentation loss

    Parameters
    ----------
    ohem : float or None
        Online hard example mining fraction, or `None` to disable OHEM
    ignore_index : int
        Index of the void class
    """

    def __init__(self, ohem=None, out_shape=(768, 704), ignore_index=255, ignore_labels=None,
                 bev_params=None, extrinsics=None):
        if ohem is not None and (ohem <= 0 or ohem > 1):
            raise ValueError("ohem should be in (0, 1]")
        self.ohem = ohem
        self.ignore_index = ignore_index
        self.ignore_labels = ignore_labels

        resolution = float(bev_params['cam_z']) / float(bev_params['f'])

        rows = torch.arange(0, out_shape[0])
        cols = torch.arange(0, out_shape[1])
        rr, cc = torch.meshgrid(rows, cols)
        idx_mesh = torch.cat([rr.unsqueeze(0), cc.unsqueeze(0)], dim=0)
        ego_position = torch.tensor([out_shape[0] // 2, 0]).view(-1, 1, 1)
        pos_mesh = idx_mesh - ego_position
        self.X, self.Z = pos_mesh[0] * resolution, pos_mesh[1] * resolution
        self.Y = abs(float(extrinsics['translation'][2]))

    def __call__(self, sem_logits, sem, weights_msk, intrinsics):
        """Compute the semantic segmentation loss

        Parameters
        ----------
        sem_logits : sequence of torch.Tensor
            A sequence of N tensors of segmentation logits with shapes C x H_i x W_i
        sem : sequence of torch.Tensor
            A sequence of N tensors of ground truth semantic segmentations with shapes H_i x W_i

        Returns
        -------
        sem_loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        sem_loss = []
        if weights_msk is None:
            for i, (sem_logits_i, sem_i) in enumerate(zip(sem_logits, sem)):
                if self.ignore_labels is not None:
                    sem_i[(sem_i == self.ignore_labels).any(-1)] = self.ignore_index  # Remap the ignore_labels to ignore_index
                sem_loss_i = functional.cross_entropy(sem_logits_i.unsqueeze(0), sem_i.unsqueeze(0),
                                                      ignore_index=self.ignore_index, reduction="none")

                f_x = intrinsics[i][0][0]
                f_y = intrinsics[i][1][1]
                self.X = self.X.to(sem_i.device)
                self.Z = self.Z.to(sem_i.device)

                S = torch.sqrt((f_x**2 * self.Z**2) + (f_x*self.X + f_y*self.Y)**2) / (self.Z**2)
                sensitivity_map = 1 / torch.log(1 + S)
                sensitivity_map[torch.isnan(sensitivity_map)] = 0.
                # sensitivity_wt = sensitivity_map * 10
                sensitivity_wt = sensitivity_map * 10

                # Distance-based weighting
                sem_loss_i *= (1 + sensitivity_wt.to(sem_i.device).squeeze(0))

                sem_loss_i = sem_loss_i.view(-1)

                if self.ohem is not None and self.ohem != 1:
                    top_k = int(ceil(sem_loss_i.numel() * self.ohem))
                    if top_k != sem_loss_i.numel():
                        sem_loss_i, _ = sem_loss_i.topk(top_k)

                sem_loss.append(sem_loss_i.mean())

            return sum(sem_loss) / len(sem_logits)

        for i, (sem_logits_i, sem_i, wt_msk_i) in enumerate(zip(sem_logits, sem, weights_msk)):
            if self.ignore_labels is not None:
                sem_i[(sem_i == self.ignore_labels).any(-1)] = self.ignore_index  # Remap the ignore_labels to ignore_index
            sem_loss_i = functional.cross_entropy(sem_logits_i.unsqueeze(0), sem_i.unsqueeze(0),
                                                  ignore_index=self.ignore_index, reduction="none")

            f_x = intrinsics[i][0][0]
            f_y = intrinsics[i][1][1]
            self.X = self.X.to(sem_i.device)
            self.Z = self.Z.to(sem_i.device)

            S = torch.sqrt((f_x**2 * self.Z**2) + (f_x*self.X + f_y*self.Y)**2) / (self.Z**2)
            sensitivity_map = 1 / torch.log(1 + S)
            sensitivity_map[torch.isnan(sensitivity_map)] = 0.
            # sensitivity_wt = sensitivity_map * 10
            sensitivity_wt = sensitivity_map * 10

            # Distance-based weighting
            sem_loss_i *= (1 + sensitivity_wt.to(sem_i.device).squeeze(0))

            # Multiply the instace-based weights mask
            sem_loss_i *= (wt_msk_i / 10000)

            sem_loss_i = sem_loss_i.view(-1)

            if self.ohem is not None and self.ohem != 1:
                top_k = int(ceil(sem_loss_i.numel() * self.ohem))
                if top_k != sem_loss_i.numel():
                    sem_loss_i, _ = sem_loss_i.topk(top_k)

            sem_loss.append(sem_loss_i.mean())

        return sum(sem_loss) / len(sem_logits)


class SemanticSegAlgo:
    """Semantic segmentation algorithm

    Parameters
    ----------
    loss : SemanticSegLoss
    num_classes : int
        Number of classes
    """

    def __init__(self, loss, num_classes, ignore_index=255, use_dice_loss= 0, seman_classes= (2,9)):
        self.loss = loss
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.use_dice_loss = True if use_dice_loss == 1 else False
        if self.use_dice_loss:
            self.seman_classes = seman_classes
            print("Using dice loss on", self.seman_classes)

    @staticmethod
    def _pack_logits(sem_logits, valid_size, img_size):
        sem_logits = functional.interpolate(sem_logits, size=img_size, mode="bilinear", align_corners=False)
        return pack_padded_images(sem_logits, valid_size)

    def _confusion_matrix(self, sem_pred, sem):
        confmat = sem[0].new_zeros(self.num_classes * self.num_classes, dtype=torch.float)

        for sem_pred_i, sem_i in zip(sem_pred, sem):
            valid = sem_i != self.ignore_index
            if valid.any():
                sem_pred_i = sem_pred_i[valid]
                sem_i = sem_i[valid]

                confmat.index_add_(0,
                                   sem_i.view(-1) * self.num_classes + sem_pred_i.view(-1),
                                   confmat.new_ones(sem_i.numel()))

        return confmat.view(self.num_classes, self.num_classes)

    @staticmethod
    def _logits(head, x, bbx, img_size, roi):
        sem_logits, sem_feat, roi_logits = head(x, bbx, img_size, roi)
        return sem_logits, sem_feat, roi_logits

    @staticmethod
    def dice_loss(pred, label):
        label = label.float()
        intersection = 2 * pred * label
        union = pred + label
        iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1.0) / (
            union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1.0
        )

        loss_mean = 1 - iou.mean()

        return loss_mean

    def _map_indices(self, input_indices):
        mapped_indices = input_indices.clone()
        for j, cls_index in enumerate(self.seman_classes):
            flag = (input_indices == j+1)
            mapped_indices[flag] = cls_index

        return mapped_indices.clone()

    def training(self, head, x, sem, bbx, valid_size, img_size, weights_msk, intrinsics, sem_msk_dice= None):
        """Given input features and ground truth compute semantic segmentation loss, confusion matrix and prediction

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute semantic segmentation logits given an input feature map. Must be callable as `head(x)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        sem : sequence of torch.Tensor
            A sequence of N tensors of ground truth semantic segmentations with shapes H_i x W_i
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        sem_loss : torch.Tensor
            A scalar tensor with the computed loss
        conf_mat : torch.Tensor
            A confusion matrix tensor with shape M x M, where M is the number of classes
        sem_pred : PackedSequence
            A sequence of N tensors of semantic segmentations with shapes H_i x W_i
        """
        # Compute logits and prediction
        sem_logits_, sem_feat, roi_logits = self._logits(head, x, bbx, img_size, False)
        sem_logits = self._pack_logits(sem_logits_, valid_size, img_size)
        if self.use_dice_loss:
            temp  = []
            for t in range(len(sem_logits._tensors)): #torch.stack(sem_logits, dim= 0))
                temp.append(sem_logits._tensors[t])
            pred_semantic_scores_2 = torch.stack(temp, dim= 0).float() # bs x 2 x 200 x 200
            pred_semantic_scores   = torch.clamp(torch.sigmoid(pred_semantic_scores_2), min= 1e-3, max=1-1e-3).clone()

            label    = sem_msk_dice
            sem_loss = self.dice_loss(pred_semantic_scores, label.float())

            background_mask = pred_semantic_scores <  0.5
            foreground_mask = pred_semantic_scores >= 0.5
            pred_semantic_scores[background_mask] = 0.
            pred_semantic_scores[foreground_mask] = 1.
            bs,cat, h, w          = pred_semantic_scores.shape
            weights               = 1.0 + torch.arange(cat, dtype= pred_semantic_scores.dtype, device= pred_semantic_scores.device)
            pred_semantic_logits  = weights.unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat(bs, 1, h, w) * pred_semantic_scores # bs x 2 x h x w
            gt_semantic_logits    = weights.unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat(bs, 1, h, w) * sem_msk_dice  # bs x 2 x h x w
            pred_semantic_indices = torch.max(pred_semantic_logits, dim=1)[0].long()
            gt_semantic_indices   = torch.max(gt_semantic_logits, dim=1)[0].long()
            sem_pred              = PackedSequence(list(self._map_indices(pred_semantic_indices))) # [h x w] x bs
            sem                   = list(self._map_indices(gt_semantic_indices))   # [h x w] x bs

            # Upsample to 200x200 resolution.
            pred_semantic_logits_200  = functional.interpolate(pred_semantic_logits.clone().float(), size= (200, 200), mode='bilinear')
            gt_semantic_logits_200    = functional.interpolate(gt_semantic_logits  .clone().float(), size= (200, 200), mode='bilinear')
            pred_semantic_indices_200 = torch.max(pred_semantic_logits_200, dim=1)[0].long()
            gt_semantic_indices_200   = torch.max(gt_semantic_logits_200, dim=1)[0].long()
            sem_pred_200              = PackedSequence(list(self._map_indices(pred_semantic_indices_200))) # [200 x 200] x bs
            sem_200                   = list(self._map_indices(gt_semantic_indices_200))   # [200 x 200] x bs

        else:
            # Compute loss
            sem_loss = self.loss(sem_logits, sem, weights_msk, intrinsics=intrinsics)
            sem_pred = PackedSequence([sem_logits_i.max(dim=0)[1] for sem_logits_i in sem_logits])
            sem_pred_200              = sem_pred
            sem_200                   = sem

        # Compute confusion matrix
        conf_mat = self._confusion_matrix(sem_pred_200, sem_200)

        return sem_loss, conf_mat, sem_pred, sem_logits, sem_feat

    def inference(self, head, x, valid_size, img_size):
        """Given input features compute semantic segmentation prediction

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute semantic segmentation logits given an input feature map. Must be callable as `head(x)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        sem_pred : PackedSequence
            A sequence of N tensors of semantic segmentations with shapes H_i x W_i
        """
        sem_logits_, sem_feat, _ = self._logits(head, x, None, img_size, False)
        sem_logits = self._pack_logits(sem_logits_, valid_size, img_size)
        if self.use_dice_loss:
            temp  = []
            for t in range(len(sem_logits._tensors)): #torch.stack(sem_logits, dim= 0))
                temp.append(sem_logits._tensors[t])
            pred_semantic_scores_2 = torch.stack(temp, dim= 0).float() # bs x 2 x 200 x 200
            pred_semantic_scores   = torch.clamp(torch.sigmoid(pred_semantic_scores_2), min= 1e-3, max=1-1e-3).clone()
            background_mask = pred_semantic_scores <  0.5
            foreground_mask = pred_semantic_scores >= 0.5
            pred_semantic_scores[background_mask] = 0.
            pred_semantic_scores[foreground_mask] = 1.
            bs,cat, h, w          = pred_semantic_scores.shape
            weights               = 1.0 + torch.arange(cat, dtype= pred_semantic_scores.dtype, device= pred_semantic_scores.device)
            pred_semantic_logits  = weights.unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat(bs, 1, h, w) * pred_semantic_scores # bs x 2 x 200 x 200
            pred_semantic_indices = torch.max(pred_semantic_logits, dim=1)[0].long()
            sem_pred              = PackedSequence(list(pred_semantic_indices)) # [200 x 200] x bs
        else:
            sem_pred = PackedSequence([sem_logits_i.max(dim=0)[1] for sem_logits_i in sem_logits])
        return sem_pred, sem_logits, sem_feat
