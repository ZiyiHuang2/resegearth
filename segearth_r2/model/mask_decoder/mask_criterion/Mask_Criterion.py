import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import sys
import os
from torch.cuda.amp import autocast

from fvcore.nn import giou_loss

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from scipy.optimize import linear_sum_assignment
from segearth_r2.model.mask_decoder.Mask2Former_Simplify.utils.misc import is_dist_avail_and_initialized, \
    nested_tensor_from_tensor_list
from segearth_r2.model.mask_decoder.Mask2Former_Simplify.utils.point_features import point_sample, \
    get_uncertain_point_coords_with_randomness
from segearth_r2.model.mask_decoder.Mask2Former_Simplify.utils.matcher import HungarianMatcher, batch_dice_loss_jit, \
    batch_sigmoid_ce_loss_jit, batch_sigmoid_focal_loss
from segearth_r2.model.mask_decoder.Mask2Former_Simplify.utils.criterion import SetCriterion


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    # inputs = inputs.flatten(1)
    # numerator = 2 * (inputs * targets).sum(-1)
    # denominator = inputs.sum(-1) + targets.sum(-1)
    # loss = 1 - (numerator + 1) / (denominator + 1)
    # return loss.sum() / num_masks
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))



class Criterion(nn.Module):

    def __init__(self, matcher, losses, num_points, oversample_ratio, importance_sample_ratio, device,
                 union_weight=0.1, union_warmup_steps=2000,
                 setpp_closed_loop=True,
                 lambda_union_single=0.01,
                 lambda_union_multi=0.05,
                 lambda_coverage_single=0.005,
                 lambda_coverage_multi=0.02,
                 lambda_consistency_single=0.005,
                 lambda_consistency_multi=0.02,
                 closed_loop_warmup_steps=2000,
                 consistency_mode="seg_align_set",
                 setpp_enable=True,
                 setpp_regroup_set_loss=True):
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.device = device
        self.pos_weight = torch.tensor([99.0])
        self.union_weight = union_weight
        self.union_warmup_steps = union_warmup_steps
        self.setpp_enable = setpp_enable
        self.setpp_regroup_set_loss = setpp_regroup_set_loss
        self.setpp_closed_loop = setpp_closed_loop
        self.lambda_union_single = lambda_union_single
        self.lambda_union_multi = lambda_union_multi
        self.lambda_coverage_single = lambda_coverage_single
        self.lambda_coverage_multi = lambda_coverage_multi
        self.lambda_consistency_single = lambda_consistency_single
        self.lambda_consistency_multi = lambda_consistency_multi
        self.closed_loop_warmup_steps = closed_loop_warmup_steps
        self.consistency_mode = consistency_mode
        self._global_step = 0


    def loss_labels(self, outputs, targets, indices):
        pass

    

    def soft_union(self, seg_logits, valid_seg_mask):
        """
        seg_logits: [B, Kmax, H, W]
        valid_seg_mask: [B, Kmax], bool, True=valid
        return: [B, 1, H, W]
        """
        seg_prob = seg_logits.float().sigmoid()
        seg_prob = seg_prob * valid_seg_mask[:, :, None, None].float()
        one_minus = (1.0 - seg_prob).clamp(min=1e-6, max=1.0)
        return 1.0 - torch.prod(one_minus, dim=1, keepdim=True)

    def dice_prob_loss(self, pred_prob, tgt_prob, sample_weight=None, eps=1.0):
        """
        pred_prob/tgt_prob: [B, 1, H, W], already probabilities
        """
        pred = pred_prob.float().flatten(1)
        tgt = tgt_prob.float().flatten(1)

        numerator = 2 * (pred * tgt).sum(dim=1) + eps
        denominator = pred.sum(dim=1) + tgt.sum(dim=1) + eps
        loss = 1.0 - numerator / denominator

        if sample_weight is not None:
            return (loss * sample_weight).mean()

        return loss.mean()

    def build_union_target_tensor(self, targets, size, device):
        union_list = []
        for t in targets:
            masks = t["masks"].float().to(device)  # [K,H,W]
            union = (masks.sum(dim=0, keepdim=True) > 0).float()  # [1,H,W]
            if union.shape[-2:] != size:
                union = F.interpolate(
                    union.unsqueeze(0),
                    size=size,
                    mode="nearest",
                ).squeeze(0)
            union_list.append(union)
        return torch.stack(union_list, dim=0)  # [B,1,H,W]

    def build_sample_weights(self, targets, single_weight, multi_weight, device):
        weights = []
        for t in targets:
            k = len(t["labels"])
            weights.append(single_weight if k <= 1 else multi_weight)
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def build_sample_weights_from_mask_num(self, mask_num, single_weight, multi_weight, device):
        weights = []
        for k in mask_num:
            k = int(k)
            weights.append(single_weight if k <= 1 else multi_weight)
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def _use_regrouped_set_loss(self, outputs):
        if outputs.get("grouped_setpp_mode", False):
            return False
        return (
            self.setpp_enable
            and self.setpp_regroup_set_loss
            and bool(outputs.get("per_target_mode", False))
            and outputs.get("target_to_image", None) is not None
            and outputs.get("mask_num", None) is not None
        )

    def build_flat_gt_masks_tensor(self, targets_flat, size, device):
        """Flat per-target targets -> gt_masks_flat [T, 1, H, W]."""
        if isinstance(size, torch.Size):
            size = tuple(size[-2:])
        elif len(size) > 2:
            size = tuple(size[-2:])
        rows = []
        for t in targets_flat:
            m = t["masks"].float().to(device)
            if m.ndim == 3:
                m = m.squeeze(0)
            if m.shape[-2:] != size:
                m = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0),
                    size=size,
                    mode="nearest",
                ).squeeze(0).squeeze(0)
            rows.append(m)
        if not rows:
            return torch.zeros(0, 1, size[0], size[1], device=device)
        return torch.stack(rows, dim=0).unsqueeze(1)

    def build_union_target_from_flat_targets(self, targets_flat, mask_num, size, device):
        """Flat per-target targets -> GT union [B, 1, H, W]."""
        if isinstance(size, torch.Size):
            size = tuple(size[-2:])
        elif len(size) > 2:
            size = tuple(size[-2:])
        unions = []
        offset = 0
        for k in mask_num:
            k = int(k)
            if k > 0:
                mask_list = []
                for j in range(k):
                    m = targets_flat[offset + j]["masks"].float().to(device)
                    if m.ndim == 3:
                        m = m.squeeze(0)
                    if m.shape[-2:] != size:
                        m = F.interpolate(
                            m.unsqueeze(0).unsqueeze(0),
                            size=size,
                            mode="nearest",
                        ).squeeze(0).squeeze(0)
                    mask_list.append(m)
                stacked = torch.stack(mask_list, dim=0)
                union = (stacked.sum(dim=0, keepdim=True) > 0).float()
            else:
                union = torch.zeros(1, size[0], size[1], device=device)
            unions.append(union)
            offset += k
        return torch.stack(unions, dim=0)

    def loss_target_aligned_masks(self, pred_seg_masks_flat, gt_masks_flat):
        """Slot-aligned target mask loss without matcher. Shapes [T,1,H,W]."""
        assert pred_seg_masks_flat.shape == gt_masks_flat.shape
        num_masks = max(int(pred_seg_masks_flat.shape[0]), 1)
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                pred_seg_masks_flat.float(),
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                gt_masks_flat.float(),
                point_coords,
                align_corners=False,
            ).squeeze(1)
        point_logits = point_sample(
            pred_seg_masks_flat.float(),
            point_coords,
            align_corners=False,
        ).squeeze(1)
        return {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

    def loss_target_aligned_seg_logits(self, pred_seg_logits_flat, gt_labels_flat):
        """Optional SEG label loss on flat aligned slots (binary match, no matcher)."""
        if pred_seg_logits_flat is None:
            return {}
        src_logits = pred_seg_logits_flat.float().reshape(-1)
        target = torch.ones_like(src_logits)
        loss_func = nn.BCEWithLogitsLoss()
        return {"loss_SEG_class": loss_func(src_logits, target)}

    def forward_grouped_setpp(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        pred_seg_flat = outputs_without_aux["pred_seg_masks"]
        size = pred_seg_flat.shape[-2:]
        device = pred_seg_flat.device
        gt_masks_flat = self.build_flat_gt_masks_tensor(targets, size, device)
        assert pred_seg_flat.shape[0] == gt_masks_flat.shape[0], (
            f"pred/gt target count mismatch: {pred_seg_flat.shape[0]} vs {gt_masks_flat.shape[0]}"
        )

        mask_num = outputs_without_aux["mask_num"]
        warmup = self.closed_loop_warmup()
        weights = self.build_sample_weights_from_mask_num(
            mask_num,
            self.lambda_union_single,
            self.lambda_union_multi,
            device,
        )
        gt_union_b = self.build_union_target_from_flat_targets(
            targets, mask_num, size, device
        )

        losses = {}
        if "masks" in self.losses:
            losses.update(self.loss_target_aligned_masks(pred_seg_flat, gt_masks_flat))
        if "SEG_labels" in self.losses:
            gt_labels = torch.zeros(
                gt_masks_flat.shape[0], 1, 1, device=device, dtype=pred_seg_flat.dtype
            )
            seg_logits = outputs_without_aux.get("pred_seg_logits")
            if seg_logits is not None:
                if seg_logits.shape[0] == gt_masks_flat.shape[0]:
                    seg_logits_flat = seg_logits.reshape(gt_masks_flat.shape[0], 1, 1)
                else:
                    grouped = outputs_without_aux.get("pred_seg_logits_grouped", seg_logits)
                    mask_num = outputs_without_aux["mask_num"]
                    from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import (
                        flatten_grouped_seg_logits,
                    )
                    seg_logits_flat = flatten_grouped_seg_logits(grouped, mask_num)
                losses.update(
                    self.loss_target_aligned_seg_logits(seg_logits_flat, gt_labels)
                )

        if "union" in self.losses:
            pred_set = outputs_without_aux["pred_set_union_mask"].float()
            gt_union_fp32 = gt_union_b.float()
            bce = F.binary_cross_entropy_with_logits(
                pred_set, gt_union_fp32, reduction="none",
            ).flatten(1).mean(dim=1)
            bce = (bce * weights).mean()
            dice = self.dice_prob_loss(pred_set.sigmoid(), gt_union_fp32, sample_weight=weights)
            if self.setpp_closed_loop:
                losses["loss_union_mask"] = bce * warmup
                losses["loss_union_dice"] = dice * warmup
            else:
                warmup_factor = self._get_union_warmup_factor()
                k1_scale = weights.mean()
                losses["loss_union_mask"] = bce * self.union_weight * warmup_factor * k1_scale
                losses["loss_union_dice"] = dice * self.union_weight * warmup_factor * k1_scale

        if self.setpp_closed_loop and self.setpp_enable and "valid_seg_mask" in outputs_without_aux:
            seg_grouped = outputs_without_aux["pred_seg_masks_grouped"]
            valid = outputs_without_aux["valid_seg_mask"]
            seg_union = self.soft_union(seg_grouped, valid)
            coverage = self.dice_prob_loss(seg_union, gt_union_b, sample_weight=weights)
            losses["loss_setpp_coverage"] = coverage * warmup

            set_prob = outputs_without_aux["pred_set_union_mask"].float().sigmoid()
            if self.consistency_mode == "seg_align_set":
                consistency = self.dice_prob_loss(seg_union, set_prob.detach(), sample_weight=weights)
            elif self.consistency_mode == "set_align_seg":
                consistency = self.dice_prob_loss(set_prob, seg_union.detach(), sample_weight=weights)
            elif self.consistency_mode == "bidirectional":
                consistency = (
                    self.dice_prob_loss(seg_union, set_prob.detach(), sample_weight=weights)
                    + 0.5 * self.dice_prob_loss(set_prob, seg_union.detach(), sample_weight=weights)
                )
            else:
                raise ValueError(f"Unknown consistency_mode: {self.consistency_mode}")
            losses["loss_setpp_consistency"] = consistency * warmup

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_full = dict(aux_outputs)
                for key in ("grouped_setpp_mode", "per_target_mode", "target_to_image", "mask_num", "valid_seg_mask"):
                    if key in outputs_without_aux and key not in aux_full:
                        aux_full[key] = outputs_without_aux[key]
                if "pred_seg_masks" in aux_full and "masks" in self.losses:
                    aux_losses = self.loss_target_aligned_masks(
                        aux_full["pred_seg_masks"], gt_masks_flat
                    )
                    losses.update({k + f"_{i}": v for k, v in aux_losses.items()})
        from segearth_r2.utils.nan_debug import audit_criterion_grouped, current_step, enabled as nan_debug_enabled
        if nan_debug_enabled():
            audit_criterion_grouped(
                outputs_without_aux, losses, gt_masks_flat, gt_union_b, current_step()
            )
        return losses

    def regroup_per_target_union(self, pred_logits_t, target_to_image, batch_size):
        """pred_logits_t: [T, 1, H, W] logits -> [B, 1, H, W] probability union."""
        prob = pred_logits_t.float().sigmoid()
        unions = []
        for b in range(batch_size):
            idx = target_to_image == b
            if idx.any():
                p = prob[idx]
                one_minus = (1.0 - p).clamp(min=1e-6, max=1.0)
                union = 1.0 - torch.prod(one_minus, dim=0)
            else:
                union = torch.zeros_like(prob[0])
            unions.append(union)
        return torch.stack(unions, dim=0)

    def build_regrouped_union_target_tensor(self, targets, target_to_image, batch_size, size, device):
        """Flat per-target targets -> [B, 1, H, W] image-level GT union."""
        if isinstance(size, torch.Size):
            size = tuple(size[-2:])
        elif len(size) > 2:
            size = tuple(size[-2:])

        unions = []
        for b in range(batch_size):
            idx = target_to_image == b
            if idx.any():
                mask_list = []
                for t_idx in torch.where(idx)[0].tolist():
                    m = targets[t_idx]["masks"].float().to(device)
                    if m.ndim == 2:
                        m = m.unsqueeze(0)
                    for j in range(m.shape[0]):
                        mask_j = m[j:j + 1]
                        if mask_j.shape[-2:] != size:
                            mask_j = F.interpolate(
                                mask_j.unsqueeze(0),
                                size=size,
                                mode="nearest",
                            ).squeeze(0)
                        mask_list.append(mask_j)
                stacked_masks = torch.cat(mask_list, dim=0)
                union = (stacked_masks.sum(dim=0, keepdim=True) > 0).float()
            else:
                union = torch.zeros(1, size[0], size[1], device=device)
            unions.append(union)
        return torch.stack(unions, dim=0)

    def closed_loop_warmup(self):
        if self.closed_loop_warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self._global_step) / float(self.closed_loop_warmup_steps))

    def loss_set_union(self, outputs, targets):
        pred_set = outputs.get("pred_set_union_mask", outputs["pred_masks"][:, 0:1])
        warmup = self.closed_loop_warmup()

        if self._use_regrouped_set_loss(outputs):
            target_to_image = outputs["target_to_image"].to(pred_set.device)
            batch_size = len(outputs["mask_num"])
            pred_set_union_b = self.regroup_per_target_union(pred_set, target_to_image, batch_size)
            gt_union_b = self.build_regrouped_union_target_tensor(
                targets, target_to_image, batch_size, pred_set.shape[-2:], pred_set.device,
            )
            weights = self.build_sample_weights_from_mask_num(
                outputs["mask_num"],
                self.lambda_union_single,
                self.lambda_union_multi,
                pred_set.device,
            )
            bce = F.binary_cross_entropy(
                pred_set_union_b.float().clamp(1e-6, 1 - 1e-6),
                gt_union_b.float(),
                reduction="none",
            ).flatten(1).mean(dim=1)
            bce = (bce * weights).mean()
            dice = self.dice_prob_loss(pred_set_union_b, gt_union_b, sample_weight=weights)
            return {
                "loss_union_mask": bce * warmup,
                "loss_union_dice": dice * warmup,
            }

        gt_union = self.build_union_target_tensor(
            targets,
            size=pred_set.shape[-2:],
            device=pred_set.device,
        )

        weights = self.build_sample_weights(
            targets,
            self.lambda_union_single,
            self.lambda_union_multi,
            pred_set.device,
        )

        bce = F.binary_cross_entropy_with_logits(
            pred_set,
            gt_union,
            reduction="none",
        ).flatten(1).mean(dim=1)
        bce = (bce * weights).mean()

        dice = self.dice_prob_loss(
            pred_set.sigmoid(),
            gt_union,
            sample_weight=weights,
        )

        return {
            "loss_union_mask": bce * warmup,
            "loss_union_dice": dice * warmup,
        }

    def loss_setpp_coverage(self, outputs, targets):
        if self._use_regrouped_set_loss(outputs):
            pred_seg = outputs.get("pred_seg_masks", outputs["pred_masks"][:, 1:])
            target_to_image = outputs["target_to_image"].to(pred_seg.device)
            batch_size = len(outputs["mask_num"])
            seg_union_b = self.regroup_per_target_union(pred_seg, target_to_image, batch_size)
            gt_union_b = self.build_regrouped_union_target_tensor(
                targets, target_to_image, batch_size, pred_seg.shape[-2:], pred_seg.device,
            )
            weights = self.build_sample_weights_from_mask_num(
                outputs["mask_num"],
                self.lambda_coverage_single,
                self.lambda_coverage_multi,
                pred_seg.device,
            )
            loss = self.dice_prob_loss(seg_union_b, gt_union_b, sample_weight=weights)
            return {"loss_setpp_coverage": loss * self.closed_loop_warmup()}

        pred_seg = outputs.get("pred_seg_masks", outputs["pred_masks"][:, 1:])
        valid = outputs["valid_seg_mask"]

        gt_union = self.build_union_target_tensor(
            targets,
            size=pred_seg.shape[-2:],
            device=pred_seg.device,
        )

        seg_union = self.soft_union(pred_seg, valid)

        weights = self.build_sample_weights(
            targets,
            self.lambda_coverage_single,
            self.lambda_coverage_multi,
            pred_seg.device,
        )

        loss = self.dice_prob_loss(
            seg_union,
            gt_union,
            sample_weight=weights,
        )

        return {"loss_setpp_coverage": loss * self.closed_loop_warmup()}

    def loss_setpp_consistency(self, outputs, targets):
        if self._use_regrouped_set_loss(outputs):
            pred_set = outputs.get("pred_set_union_mask", outputs["pred_masks"][:, 0:1])
            pred_seg = outputs.get("pred_seg_masks", outputs["pred_masks"][:, 1:])
            target_to_image = outputs["target_to_image"].to(pred_set.device)
            batch_size = len(outputs["mask_num"])
            set_union_b = self.regroup_per_target_union(pred_set, target_to_image, batch_size)
            seg_union_b = self.regroup_per_target_union(pred_seg, target_to_image, batch_size)
            weights = self.build_sample_weights_from_mask_num(
                outputs["mask_num"],
                self.lambda_consistency_single,
                self.lambda_consistency_multi,
                pred_set.device,
            )
            if self.consistency_mode == "seg_align_set":
                loss = self.dice_prob_loss(seg_union_b, set_union_b.detach(), sample_weight=weights)
            elif self.consistency_mode == "set_align_seg":
                loss = self.dice_prob_loss(set_union_b, seg_union_b.detach(), sample_weight=weights)
            elif self.consistency_mode == "bidirectional":
                loss = (
                    self.dice_prob_loss(seg_union_b, set_union_b.detach(), sample_weight=weights)
                    + 0.5 * self.dice_prob_loss(set_union_b, seg_union_b.detach(), sample_weight=weights)
                )
            else:
                raise ValueError(f"Unknown consistency_mode: {self.consistency_mode}")
            return {"loss_setpp_consistency": loss * self.closed_loop_warmup()}

        pred_set = outputs.get("pred_set_union_mask", outputs["pred_masks"][:, 0:1])
        pred_seg = outputs.get("pred_seg_masks", outputs["pred_masks"][:, 1:])
        valid = outputs["valid_seg_mask"]

        set_prob = pred_set.sigmoid()
        seg_union = self.soft_union(pred_seg, valid)

        weights = self.build_sample_weights(
            targets,
            self.lambda_consistency_single,
            self.lambda_consistency_multi,
            pred_set.device,
        )

        if self.consistency_mode == "seg_align_set":
            # SET constrains SEG union; set_prob.detach() blocks consistency grad to SET
            loss = self.dice_prob_loss(
                seg_union,
                set_prob.detach(),
                sample_weight=weights,
            )
        elif self.consistency_mode == "set_align_seg":
            loss = self.dice_prob_loss(
                set_prob,
                seg_union.detach(),
                sample_weight=weights,
            )
        elif self.consistency_mode == "bidirectional":
            loss = (
                self.dice_prob_loss(seg_union, set_prob.detach(), sample_weight=weights)
                + 0.5 * self.dice_prob_loss(set_prob, seg_union.detach(), sample_weight=weights)
            )
        else:
            raise ValueError(f"Unknown consistency_mode: {self.consistency_mode}")

        return {"loss_setpp_consistency": loss * self.closed_loop_warmup()}

    def loss_union_mask(self, outputs, union_targets, num_masks):
        """Dice + sigmoid CE loss on query 0 (SET union mask)."""
        pred_masks = outputs["pred_masks"]  # [B, Q, H, W]
        union_mask = pred_masks[:, 0:1, :, :]  # [B, 1, H, W] -- query 0
        pred_h, pred_w = union_mask.shape[-2:]

        target_masks = []
        for t in union_targets:
            tgt = t["union_mask"].to(union_mask.device)
            if tgt.ndim == 2:
                tgt = tgt.unsqueeze(0)  # [1, H, W]
            if tgt.shape[-2:] != (pred_h, pred_w):
                tgt = F.interpolate(
                    tgt.unsqueeze(0).float(),
                    size=(pred_h, pred_w),
                    mode="nearest",
                ).squeeze(0)
            target_masks.append(tgt)
        target_masks = torch.stack(target_masks, dim=0)  # [B, 1, H, W]

        src_masks = union_mask.flatten(0, 1)  # [B, H, W]
        tgt_masks = target_masks.flatten(0, 1)  # [B, H, W]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.float().unsqueeze(1),
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                tgt_masks.float().unsqueeze(1),
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks.float().unsqueeze(1),
            point_coords,
            align_corners=False,
        ).squeeze(1)

        num_union = max(union_mask.shape[0], 1)
        losses = {
            "loss_union_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_union),
            "loss_union_dice": dice_loss_jit(point_logits, point_labels, num_union),
        }
        return losses

    def build_union_target(self, targets):
        """Build union mask targets from GT instance masks."""
        union_targets = []
        for t in targets:
            masks = t["masks"]  # [K, H, W]
            union = (masks.float().sum(dim=0) > 0).float().unsqueeze(0)  # [1, H, W]
            union_targets.append({"union_mask": union})
        return union_targets

    def _get_union_warmup_factor(self):
        if self.union_warmup_steps <= 0:
            return 1.0
        return min(1.0, self._global_step / self.union_warmup_steps)

    def set_global_step(self, step):
        self._global_step = step

    def loss_SEG_labels(self, outputs, targets, indices, num_masks):
        assert "pred_SEG_logits" in outputs
        # src_logits [batch_size, num_query, 1]
        pred_SEG_logits = outputs['pred_SEG_logits']
        if pred_SEG_logits is None:
            return {"loss_SEG_class": None}
        src_logits = pred_SEG_logits.float()
        src_logits = src_logits
        target_query = torch.zeros_like(src_logits).to(src_logits.device)
        for i, (index_i, _) in enumerate(indices):
            target_query[i, index_i] = 1
        num_sample = src_logits.shape[0] * src_logits.shape[1]
        neg = num_sample - num_masks
        if neg <= 0:
            pos_weight = 1.0
        else:
            pos_weight = neg / num_masks
        loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=src_logits.device))
        # loss_func = nn.BCELoss()
        loss_SEG_class = loss_func(src_logits, target_query)
        # loss_SEG_class = -pos_weight * (target_query * torch.log(src_logits)) - neg_weight * ((1 - target_query) * torch.log(1 - src_logits))
        # loss_SEG_class = loss_SEG_class.mean()
        losses = {"loss_SEG_class": loss_SEG_class}
        return losses
    

    def loss_masks(self, outputs, targets, indices, num_masks):
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            data_type = src_masks.dtype
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.float(),
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks.float(),
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks.float(),
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses



    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_binary_mask(self, target):
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot

    def get_loss(self, loss, outputs, targets, indices, num_masks, union_targets=None):
        loss_map = {
            'SEG_labels': self.loss_SEG_labels,
            'masks': self.loss_masks,
            'union': self.loss_set_union if self.setpp_closed_loop else self.loss_union_mask,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss == 'union':
            if self.setpp_closed_loop:
                return loss_map[loss](outputs, targets)
            return loss_map[loss](outputs, union_targets, num_masks)
        return loss_map[loss](outputs, targets, indices, num_masks)

    def get_indices(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        return indices

    def build_image_union_tensor_from_flat_targets(self, targets, target_to_image, mask_num, size, device):
        """Build per-target-row union GT by repeating each image's union onto its target rows."""
        image_unions = []
        offset = 0
        for k in mask_num:
            k = int(k)
            if k > 0:
                masks = torch.stack([targets[offset + j]["masks"][0].float() for j in range(k)], dim=0)
                union = (masks.sum(dim=0) > 0).float()
            else:
                union = torch.zeros(size, device=device)
            if union.shape[-2:] != size:
                union = F.interpolate(
                    union.unsqueeze(0).unsqueeze(0),
                    size=size,
                    mode="nearest",
                ).squeeze(0).squeeze(0)
            image_unions.append(union)
            offset += k
        union_rows = []
        for t_idx, img_idx in enumerate(target_to_image.tolist()):
            union_rows.append(image_unions[int(img_idx)])
        return torch.stack(union_rows, dim=0).unsqueeze(1)  # [T, 1, H, W]

    def build_per_target_sample_weights(self, target_to_image, mask_num, single_weight, multi_weight, device):
        image_weights = []
        for k in mask_num:
            k = int(k)
            image_weights.append(single_weight if k <= 1 else multi_weight)
        row_weights = [image_weights[int(i)] for i in target_to_image.tolist()]
        return torch.tensor(row_weights, dtype=torch.float32, device=device)

    def forward_per_target(self, outputs, targets):
        """Target-level batch T: each row has Q=2 (SET + one SEG)."""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        target_to_image = outputs_without_aux["target_to_image"]
        mask_num = outputs_without_aux["mask_num"]

        pred_masks_full = outputs_without_aux["pred_masks"]  # [T, 2, H, W]
        instance_masks = pred_masks_full[:, 1:2, :, :]
        pred_SEG_logits_full = outputs_without_aux.get("pred_SEG_logits", None)
        instance_SEG_logits = pred_SEG_logits_full[:, 1:2, :] if pred_SEG_logits_full is not None else None

        instance_outputs = {
            "pred_masks": instance_masks,
            "pred_SEG_logits": instance_SEG_logits,
        }

        indices = [
            (torch.tensor([0], device=pred_masks_full.device), torch.tensor([0], device=pred_masks_full.device))
            for _ in targets
        ]
        num_masks = max(len(targets), 1)

        size = pred_masks_full.shape[-2:]
        device = pred_masks_full.device
        gt_union_rows = self.build_image_union_tensor_from_flat_targets(
            targets, target_to_image, mask_num, size, device
        )
        row_weights = self.build_per_target_sample_weights(
            target_to_image, mask_num,
            self.lambda_union_single, self.lambda_union_multi, device,
        )

        warmup = self.closed_loop_warmup()
        losses = {}

        if 'masks' in self.losses:
            l_dict = self.get_loss('masks', instance_outputs, targets, indices, num_masks)
            losses.update(l_dict)
        if 'SEG_labels' in self.losses:
            l_dict = self.get_loss('SEG_labels', instance_outputs, targets, indices, num_masks)
            losses.update(l_dict)

        if 'union' in self.losses:
            if self.setpp_closed_loop and self._use_regrouped_set_loss(outputs_without_aux):
                l_dict = self.loss_set_union(outputs_without_aux, targets)
                losses.update(l_dict)
            else:
                pred_set = outputs_without_aux.get("pred_set_union_mask", pred_masks_full[:, 0:1])
                bce = F.binary_cross_entropy_with_logits(
                    pred_set, gt_union_rows, reduction="none",
                ).flatten(1).mean(dim=1)
                bce = (bce * row_weights).mean()
                dice = self.dice_prob_loss(pred_set.sigmoid(), gt_union_rows, sample_weight=row_weights)
                if self.setpp_closed_loop:
                    losses["loss_union_mask"] = bce * warmup
                    losses["loss_union_dice"] = dice * warmup
                else:
                    warmup_factor = self._get_union_warmup_factor()
                    k1_scale = row_weights.mean()
                    losses["loss_union_mask"] = bce * self.union_weight * warmup_factor * k1_scale
                    losses["loss_union_dice"] = dice * self.union_weight * warmup_factor * k1_scale

        if self.setpp_closed_loop and self.setpp_enable and "valid_seg_mask" in outputs_without_aux:
            if self._use_regrouped_set_loss(outputs_without_aux):
                losses.update(self.loss_setpp_coverage(outputs_without_aux, targets))
                losses.update(self.loss_setpp_consistency(outputs_without_aux, targets))
            else:
                pred_seg = outputs_without_aux.get("pred_seg_masks", pred_masks_full[:, 1:2])
                seg_union = pred_seg.sigmoid()
                coverage = self.dice_prob_loss(seg_union, gt_union_rows, sample_weight=row_weights)
                losses["loss_setpp_coverage"] = coverage * warmup

                set_prob = outputs_without_aux.get(
                    "pred_set_union_mask", pred_masks_full[:, 0:1]
                ).sigmoid()
                if self.consistency_mode == "seg_align_set":
                    consistency = self.dice_prob_loss(seg_union, set_prob.detach(), sample_weight=row_weights)
                elif self.consistency_mode == "set_align_seg":
                    consistency = self.dice_prob_loss(set_prob, seg_union.detach(), sample_weight=row_weights)
                elif self.consistency_mode == "bidirectional":
                    consistency = (
                        self.dice_prob_loss(seg_union, set_prob.detach(), sample_weight=row_weights)
                        + 0.5 * self.dice_prob_loss(set_prob, seg_union.detach(), sample_weight=row_weights)
                    )
                else:
                    raise ValueError(f"Unknown consistency_mode: {self.consistency_mode}")
                losses["loss_setpp_consistency"] = consistency * warmup

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_full = dict(aux_outputs)
                aux_full["target_to_image"] = target_to_image
                aux_full["mask_num"] = mask_num
                aux_full["per_target_mode"] = True
                if "valid_seg_mask" in outputs_without_aux:
                    aux_full["valid_seg_mask"] = outputs_without_aux["valid_seg_mask"]
                aux_losses = self.forward_per_target({"aux_outputs": [], **aux_full}, targets)
                losses.update({k + f"_{i}": v for k, v in aux_losses.items()})

        return losses

    def forward(self, outputs, targets):
        if outputs.get("grouped_setpp_mode", False):
            return self.forward_grouped_setpp(outputs, targets)
        if outputs.get("per_target_mode", False):
            return self.forward_per_target(outputs, targets)

        # Hard assertion: targets must be per-image with [K, H, W] masks
        for b, t in enumerate(targets):
            assert t["masks"].ndim == 3, \
                f"[BUG] target[{b}] masks should be [K, H, W], got shape {t['masks'].shape}. "
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # --- Strip query 0 (SET) before matcher ---
        pred_masks_full = outputs_without_aux["pred_masks"]  # [B, Q, H, W]
        instance_masks = pred_masks_full[:, 1:, :, :]  # [B, Q-1, H, W] -- only SEG queries
        pred_SEG_logits_full = outputs_without_aux.get("pred_SEG_logits", None)
        if pred_SEG_logits_full is not None:
            instance_SEG_logits = pred_SEG_logits_full[:, 1:, :]  # [B, Q-1, 1]
        else:
            instance_SEG_logits = None

        instance_outputs = {
            "pred_masks": instance_masks,
            "pred_SEG_logits": instance_SEG_logits,
        }

        indices = self.matcher(instance_outputs, targets)

        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=outputs['pred_masks'].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # --- Union target ---
        union_targets = self.build_union_target(targets)
        warmup_factor = self._get_union_warmup_factor()

        # Per-sample union weight: reduce for K=1 samples where union == instance
        per_sample_union_weight = []
        for t in targets:
            k = len(t["labels"])
            if k <= 1:
                per_sample_union_weight.append(0.1)  # union == instance, redundant
            else:
                per_sample_union_weight.append(1.0)
        per_sample_union_weight = torch.tensor(per_sample_union_weight, device=outputs['pred_masks'].device)

        # Compute K=1 scaling factor: union == instance for single-target samples
        k1_scale = per_sample_union_weight.mean()

        losses = {}
        for loss in self.losses:
            if loss == 'union':
                l_dict = self.get_loss(loss, outputs_without_aux, targets, None, num_masks, union_targets=union_targets)
                for k, v in l_dict.items():
                    if v is not None:
                        if self.setpp_closed_loop:
                            # lambda already applied inside loss_set_union; only weight_dict scales BCE/Dice
                            losses[k] = v
                        else:
                            losses[k] = v * self.union_weight * warmup_factor * k1_scale
            else:
                l_dict = self.get_loss(loss, instance_outputs, targets, indices, num_masks)
                losses.update(l_dict)

        if self.setpp_closed_loop and self.setpp_enable and "valid_seg_mask" in outputs_without_aux:
            losses.update(self.loss_setpp_coverage(outputs_without_aux, targets))
            losses.update(self.loss_setpp_consistency(outputs_without_aux, targets))

        # --- Auxiliary losses: strip query 0 from each aux output ---
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_full = dict(aux_outputs)
                if outputs.get("per_target_mode", False):
                    for key in ("per_target_mode", "target_to_image", "mask_num", "valid_seg_mask"):
                        if key in outputs and key not in aux_full:
                            aux_full[key] = outputs[key]
                aux_masks_full = aux_full["pred_masks"]
                aux_instance_masks = aux_masks_full[:, 1:, :, :]
                aux_SEG_logits_full = aux_full.get("pred_SEG_logits", None)
                if aux_SEG_logits_full is not None:
                    aux_instance_SEG_logits = aux_SEG_logits_full[:, 1:, :]
                else:
                    aux_instance_SEG_logits = None

                aux_instance = {
                    "pred_masks": aux_instance_masks,
                    "pred_SEG_logits": aux_instance_SEG_logits,
                }
                aux_indices = self.matcher(aux_instance, targets)
                for loss in self.losses:
                    if loss == 'union':
                        l_dict = self.get_loss(loss, aux_full, targets, None, num_masks, union_targets=union_targets)
                        for k, v in l_dict.items():
                            if v is not None:
                                if self.setpp_closed_loop:
                                    losses[k + f"_{i}"] = v
                                else:
                                    losses[k + f"_{i}"] = v * self.union_weight * warmup_factor * k1_scale
                    else:
                        l_dict = self.get_loss(loss, aux_instance, targets, aux_indices, num_masks)
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses

    def _get_targets(self, gt_masks):
        targets = []
        for mask in gt_masks:
            binary_masks = self._get_binary_mask(mask)
            cls_label = torch.unique(mask)
            labels = cls_label[1:]
            binary_masks = binary_masks[labels]
            targets.append({'masks': binary_masks, 'labels': labels})
        return targets

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)







class hungarian_matcher_InstructSeg(HungarianMatcher):

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0,
                ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points


    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_masks"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            # out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            # tgt_ids = targets[b]["labels"]
            #
            # # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # # but approximate it in 1 - proba[target class].
            # # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]


            # if 'pred_boxes' in outputs and outputs['pred_boxes'] is not None:
            #     out_bbox = outputs["pred_boxes"][b].float()
            #     tgt_bbox = targets[b]["boxes"].float()
            #     cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            #     cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox), box_ops.box_cxcywh_to_xyxy(tgt_bbox))
            # else:
            #     cost_bbox = 0
            #     cost_giou = 0



            cost_class = 0
                


            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask.float(),
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask.float(),
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                    # + self.cost_box * cost_bbox
                    # + self.cost_giou * cost_giou
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


