"""Hard Context Mining Loss (HCML) for CS-DEG-core."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def lambda_hcml_schedule(
    step: int,
    warmup_steps: int,
    ramp_steps: int,
    lambda_max: float,
) -> float:
    if step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return lambda_max
    return lambda_max * min(1.0, max(0.0, (step - warmup_steps) / ramp_steps))


def _resize_masks(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbor resize for binary masks [B,1,H,W] or [B,H,W]."""
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    return F.interpolate(masks.float(), size=size, mode="nearest")


def _mine_hard_pixels(
    pred_prob: torch.Tensor,
    valid: torch.Tensor,
    hard_topk_ratio: float,
    hard_topk_min: int,
) -> torch.Tensor:
    """Return bool mask [B,1,H,W] of mined hard context pixels."""
    b, _, h, w = pred_prob.shape
    k = max(hard_topk_min, int(hard_topk_ratio * h * w))
    hard_masks = []
    for i in range(b):
        prob = pred_prob[i, 0]
        valid_i = valid[i, 0].bool()
        hard = (~valid_i) & (prob > 0.5)
        if hard.sum() == 0:
            hard_masks.append(torch.zeros(1, h, w, dtype=torch.bool, device=pred_prob.device))
            continue
        prob_hard = prob.masked_fill(~hard, -1.0)
        kk = min(int(hard.sum().item()), k)
        _, top_idx = torch.topk(prob_hard.flatten(), k=kk)
        hard_flat = torch.zeros(h * w, dtype=torch.bool, device=pred_prob.device)
        hard_flat[top_idx] = True
        hard_masks.append(hard_flat.view(1, h, w))
    return torch.stack(hard_masks, dim=0)


def compute_hcml_losses(
    outputs: dict,
    targets: list[dict],
    cs_deg_config,
    global_step: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Compute HCML sub-losses on final-layer evidence logits.

    Returns dict with loss_cs_context, loss_cs_evidence, loss_cs_rank, loss_cs_sibling_aux.
    """
    device = outputs["pred_context_logits"].device
    pred_ctx = outputs["pred_context_logits"]
    pred_evi = outputs["pred_evidence_logits"]
    pred_masks = outputs["pred_masks"]

    b, q, hc, wc = pred_ctx.shape
    size = (hc, wc)

    ctx_losses = []
    evi_losses = []
    rank_losses = []
    sib_losses = []
    n_ctx = 0
    n_evi = 0
    n_rank = 0
    n_sib = 0

    rank_margin = float(getattr(cs_deg_config, "RANK_MARGIN", 0.2))
    hard_topk_ratio = float(getattr(cs_deg_config, "HARD_TOPK_RATIO", 0.01))
    hard_topk_min = int(getattr(cs_deg_config, "HARD_TOPK_MIN", 100))
    rank_samples = 512

    with torch.no_grad():
        pred_prob = pred_masks.sigmoid()
        if pred_prob.dim() == 3:
            pred_prob = pred_prob.unsqueeze(1)
        if pred_prob.shape[-2:] != size:
            pred_prob = F.interpolate(pred_prob, size=size, mode="bilinear", align_corners=False)

    for i in range(b):
        tgt_mask = targets[i]["masks"]
        if tgt_mask.dim() == 2:
            tgt_mask = tgt_mask.unsqueeze(0)
        tgt_r = _resize_masks(tgt_mask, size)

        sib = targets[i].get("sibling_masks")
        if sib is None:
            sib_r = torch.zeros_like(tgt_r)
        else:
            if sib.dim() == 2:
                sib = sib.unsqueeze(0)
            sib_r = _resize_masks(sib, size)

        valid = ((tgt_r > 0.5) | (sib_r > 0.5)).float()

        ctx_logit = pred_ctx[i, 0] if q == 1 else pred_ctx[i].mean(dim=0)
        evi_logit = pred_evi[i, 0] if q == 1 else pred_evi[i].mean(dim=0)

        hard_mask = _mine_hard_pixels(
            pred_prob[i : i + 1],
            valid,
            hard_topk_ratio,
            hard_topk_min,
        )[0, 0]

        if hard_mask.any():
            ctx_losses.append(
                F.binary_cross_entropy_with_logits(ctx_logit[hard_mask], torch.ones_like(ctx_logit[hard_mask]))
            )
            n_ctx += 1

            tgt_idx = (tgt_r[0, 0] > 0.5).nonzero(as_tuple=False)
            hard_idx = hard_mask.nonzero(as_tuple=False)
            if tgt_idx.numel() > 0 and hard_idx.numel() > 0:
                n_pairs = min(rank_samples, tgt_idx.shape[0], hard_idx.shape[0])
                ti = tgt_idx[torch.randint(0, tgt_idx.shape[0], (n_pairs,), device=device)]
                hi = hard_idx[torch.randint(0, hard_idx.shape[0], (n_pairs,), device=device)]
                score_t = evi_logit[ti[:, 0], ti[:, 1]]
                score_h = ctx_logit[hi[:, 0], hi[:, 1]]
                rank_losses.append(F.relu(rank_margin + score_h - score_t).mean())
                n_rank += 1

        tgt_pos = tgt_r[0, 0] > 0.5
        if tgt_pos.any():
            evi_losses.append(
                F.binary_cross_entropy_with_logits(evi_logit[tgt_pos], torch.ones_like(evi_logit[tgt_pos]))
            )
            n_evi += 1

        sib_pos = sib_r[0, 0] > 0.5
        if sib_pos.any():
            sib_losses.append(
                F.binary_cross_entropy_with_logits(ctx_logit[sib_pos], torch.zeros_like(ctx_logit[sib_pos]))
            )
            n_sib += 1

    zero = pred_ctx.sum() * 0.0

    loss_ctx = torch.stack(ctx_losses).mean() if ctx_losses else zero
    loss_evi = torch.stack(evi_losses).mean() if evi_losses else zero
    loss_rank = torch.stack(rank_losses).mean() if rank_losses else zero
    loss_sib = torch.stack(sib_losses).mean() if sib_losses else zero

    return {
        "loss_cs_context": loss_ctx,
        "loss_cs_evidence": loss_evi,
        "loss_cs_rank": loss_rank,
        "loss_cs_sibling_aux": loss_sib,
    }


def compute_evidence_consistency_loss(
    pred_evi: torch.Tensor,
    pred_masks: torch.Tensor,
    targets: list[dict],
    size: tuple[int, int],
) -> torch.Tensor:
    """Align final target evidence with GT mask (BCE) and detached pred mask (MSE)."""
    b, q, _, _ = pred_evi.shape
    device = pred_evi.device
    bce_terms = []
    mse_terms = []

    pred_prob = pred_masks.sigmoid()
    if pred_prob.dim() == 3:
        pred_prob = pred_prob.unsqueeze(1)
    if pred_prob.shape[-2:] != size:
        pred_prob = F.interpolate(pred_prob, size=size, mode="bilinear", align_corners=False)

    for i in range(b):
        tgt_mask = targets[i]["masks"]
        if tgt_mask.dim() == 2:
            tgt_mask = tgt_mask.unsqueeze(0)
        tgt_r = _resize_masks(tgt_mask, size)[0, 0]
        evi = pred_evi[i, 0] if q == 1 else pred_evi[i].mean(dim=0)
        tgt_pos = tgt_r > 0.5
        if tgt_pos.any():
            bce_terms.append(
                F.binary_cross_entropy_with_logits(evi[tgt_pos], torch.ones_like(evi[tgt_pos]))
            )
        mse_terms.append(
            F.mse_loss(evi.sigmoid(), pred_prob[i, 0].detach())
        )

    zero = pred_evi.sum() * 0.0
    bce = torch.stack(bce_terms).mean() if bce_terms else zero
    mse = torch.stack(mse_terms).mean() if mse_terms else zero
    return 0.5 * bce + 0.5 * mse


def compute_boundary_loss(
    pred_evi: torch.Tensor,
    targets: list[dict],
    size: tuple[int, int],
) -> torch.Tensor:
    """Optional boundary alignment via Sobel edge overlap."""
    b, q, _, _ = pred_evi.shape
    terms = []
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=pred_evi.device, dtype=pred_evi.dtype).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)

    def _edges(x: torch.Tensor) -> torch.Tensor:
        x4 = x.unsqueeze(0).unsqueeze(0)
        gx = F.conv2d(x4, kernel_x, padding=1)
        gy = F.conv2d(x4, kernel_y, padding=1)
        return (gx ** 2 + gy ** 2).sqrt().squeeze()

    for i in range(b):
        tgt_mask = targets[i]["masks"]
        if tgt_mask.dim() == 2:
            tgt_mask = tgt_mask.unsqueeze(0)
        tgt_r = _resize_masks(tgt_mask, size)[0, 0]
        evi = pred_evi[i, 0] if q == 1 else pred_evi[i].mean(dim=0)
        tgt_edge = _edges(tgt_r.float())
        evi_edge = _edges(evi.sigmoid())
        if tgt_edge.sum() > 0:
            terms.append(F.l1_loss(evi_edge, tgt_edge / (tgt_edge.max() + 1e-6)))
    zero = pred_evi.sum() * 0.0
    return torch.stack(terms).mean() if terms else zero


def compute_cs_deg_losses(
    outputs: dict,
    targets: list[dict],
    cs_deg_config,
    global_step: int = 0,
) -> dict[str, torch.Tensor]:
    """HCML + optional consistency / boundary losses."""
    hcml = compute_hcml_losses(outputs, targets, cs_deg_config, global_step=global_step)
    pred_evi = outputs["pred_evidence_logits"]
    size = (pred_evi.shape[-2], pred_evi.shape[-1])
    zero = pred_evi.sum() * 0.0

    if getattr(cs_deg_config, "EVIDENCE_CONSISTENCY", True):
        hcml["loss_cs_consistency"] = compute_evidence_consistency_loss(
            pred_evi, outputs["pred_masks"], targets, size
        )
    else:
        hcml["loss_cs_consistency"] = zero

    if getattr(cs_deg_config, "BOUNDARY_LOSS", False):
        hcml["loss_cs_boundary"] = compute_boundary_loss(pred_evi, targets, size)
    else:
        hcml["loss_cs_boundary"] = zero

    return hcml
