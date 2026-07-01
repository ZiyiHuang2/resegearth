#!/usr/bin/env python3
"""End-to-end smoke: SegEarthR2.forward + grouped SET++ loss + backward."""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import json
import transformers

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEG_ROOT = os.path.abspath(os.path.join(REPO, ".."))
sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import (  # noqa: E402
    DataCollatorForCOCODatasetV2,
    LaSeRSDataset,
    get_mask_config,
)
from segearth_r2.model.language_model.llava_phi import SegEarthR2  # noqa: E402

FULL_CFG = f"{REPO}/segearth_r2/model/mask_decoder/mask_config/ours_full_enhanced_tgswin_setpp.yaml"
DEFAULT_MODEL = os.environ.get(
    "SMOKE_E2E_MODEL",
    os.path.join(RESEG_ROOT, "output/setpp/setpp-lasers-warmstart-8w-gd4/merged_model"),
)
FALLBACK_MODEL = os.path.join(
    RESEG_ROOT, "output/base/standard-base-lasers-siglip1-8w-gd4/merged_model"
)
VISION_TOWER = os.path.join(RESEG_ROOT, "pretrained_model/CLIP/siglip-so400m-patch14-384")
VISION_TOWER_MASK = os.path.join(
    RESEG_ROOT, "pretrained_model/mask2former/model_final_54b88a.pkl"
)
LASERS_PATH = os.path.join(RESEG_ROOT.replace("/reseg", ""), "data/LaSeRS")
if not os.path.isdir(LASERS_PATH):
    LASERS_PATH = os.path.join(RESEG_ROOT, "data/LaSeRS")


def _grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm().item() ** 2)
    return math.sqrt(total) if total > 0 else 0.0


def _max_grad(module: torch.nn.Module) -> float:
    peak = 0.0
    for p in module.parameters():
        if p.grad is not None:
            peak = max(peak, float(p.grad.detach().abs().max().item()))
    return peak


def _has_nan_inf(module: torch.nn.Module) -> Tuple[bool, bool]:
    nan = inf = False
    for p in module.parameters():
        if p.grad is None:
            continue
        nan = nan or bool(torch.isnan(p.grad).any().item())
        inf = inf or bool(torch.isinf(p.grad).any().item())
    return nan, inf


def _assert_cfg(cfg) -> None:
    tg = cfg.TG_SWIN
    assert bool(getattr(tg, "ENABLED", False)), "TG_SWIN.ENABLED must be True"
    assert bool(getattr(tg, "GROUPED_SETPP_DECODER", False)), "GROUPED_SETPP_DECODER must be True"
    assert not bool(getattr(tg, "USE_COARSE_EVIDENCE", False)), "USE_COARSE_EVIDENCE must be False"
    assert not bool(getattr(tg, "USE_DR_EWTI", False)), "USE_DR_EWTI must be False"
    print(
        f"[PASS] cfg: ENABLED={tg.ENABLED} GROUPED={tg.GROUPED_SETPP_DECODER} "
        f"coarse={getattr(tg, 'USE_COARSE_EVIDENCE', False)} dr={getattr(tg, 'USE_DR_EWTI', False)}"
    )


def _pick_lasers_batch(dataset: LaSeRSDataset, want: Tuple[int, int]) -> List[int]:
    buckets: Dict[int, int] = {}
    for i, rec in enumerate(dataset.reason_file):
        k = rec["answer"].count("[SEG]")
        if k not in buckets and k in want:
            buckets[k] = i
        if all(v in buckets for v in want):
            break
    if all(v in buckets for v in want):
        return [buckets[want[0]], buckets[want[1]]]
    # fallback: first two samples with >=2 targets
    picked = []
    for i, rec in enumerate(dataset.reason_file):
        if rec["answer"].count("[SEG]") >= 2:
            picked.append(i)
        if len(picked) == 2:
            return picked
    raise RuntimeError("Could not find LaSeRS samples with multi-target answers")


def _load_mask_state_from_merged(model_path: str, prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        keys = [k for k in weight_map if k.startswith(prefixes)]
        shard_files = sorted({weight_map[k] for k in keys})
        shards: Dict[str, torch.Tensor] = {}
        for shard in shard_files:
            shards.update(load_file(os.path.join(model_path, shard)))
        return {k: shards[k] for k in keys if k in shards}
    if os.path.isfile(single_path):
        shards = load_file(single_path)
        return {k: v for k, v in shards.items() if k.startswith(prefixes)}
    raise FileNotFoundError(f"No safetensors weights under {model_path}")


def main():
    os.chdir(REPO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"[info] device={device} dtype={dtype}")

    if os.path.isdir(DEFAULT_MODEL):
        model_path = DEFAULT_MODEL
    elif os.path.isdir(FALLBACK_MODEL):
        print(f"[warn] preferred smoke model missing, falling back to {FALLBACK_MODEL}")
        model_path = FALLBACK_MODEL
    else:
        raise FileNotFoundError(
            f"merged model not found: {DEFAULT_MODEL} (fallback: {FALLBACK_MODEL})"
        )

    cfg = get_mask_config(FULL_CFG)
    _assert_cfg(cfg)

    model_args = SimpleNamespace(
        vision_tower=VISION_TOWER,
        vision_tower_mask=VISION_TOWER_MASK,
        swin_type="base",
        load_mask2former=True,
        mask_config=FULL_CFG,
        setpp_enable=True,
        setpp_closed_loop=True,
        setpp_regroup_set_loss=True,
        setpp_csqr_enable=True,
        setpp_consistency_mode="seg_align_set",
        setpp_closed_loop_warmup_steps=2000,
        enable_attention_loss=False,
        debug_batch_semantics=False,
    )
    data_args = SimpleNamespace(
        lasers_holdout_ratio=0.05,
        lasers_holdout_seed=42,
        image_aspect_ratio="square",
        image_grid_pinpoints=None,
    )

    print(f"[info] loading model from {model_path}")
    mask_prefixes = (
        "predictor.",
        "pixel_decoder.",
        "SEG_token_projector.",
        "SET_token_projector.",
        "tg_swin_",
    )
    saved_mask_state = _load_mask_state_from_merged(model_path, mask_prefixes)
    model = SegEarthR2.from_pretrained(
        model_path,
        mask_decoder_cfg=cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.initial_mask_module(VISION_TOWER_MASK, model_args)
    model.load_state_dict(saved_mask_state, strict=False)
    model.persist_setpp_config(model_args)
    model.get_model().initialize_vision_modules(model_args=model_args)
    model.to(device=device, dtype=dtype)
    if model.get_vision_tower() is not None:
        model.get_vision_tower().to(device=device, dtype=dtype)
    model.get_model().get_vision_tower_mask().to(device=device, dtype=dtype)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, use_fast=False, model_max_length=2048, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for tok in ("[SEG]", "[SET]"):
        if tok not in tokenizer.get_vocab():
            tokenizer.add_tokens([tok])
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    clip_processor = transformers.SiglipImageProcessor.from_pretrained(VISION_TOWER)
    dataset = LaSeRSDataset(
        LASERS_PATH,
        tokenizer,
        data_args,
        split="train_data.json",
        holdout_mode="train",
    )
    idxs = _pick_lasers_batch(dataset, (2, 3))
    instances = [dataset[i] for i in idxs]
    mask_num = [inst["mask_num"] for inst in instances]
    print(f"[info] LaSeRS batch indices={idxs} mask_num={mask_num} T={sum(mask_num)}")

    collator = DataCollatorForCOCODatasetV2(
        tokenizer=tokenizer, clip_image_processor=clip_processor
    )
    batch = collator(instances)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            batch[k] = [t.to(device) for t in v]

    # Trainable modules for gradient check
    for p in model.parameters():
        p.requires_grad_(False)
    train_modules = {
        "SEG_token_projector": model.SEG_token_projector,
        "SET_token_projector": model.SET_token_projector,
        "tg_swin_tcf": model.tg_swin_tcf,
        "tg_swin_controller": model.tg_swin_controller,
        "tg_swin_set_control": model.tg_swin_set_control,
        "pixel_decoder": model.pixel_decoder,
        "predictor": model.predictor,
    }
    for mod in train_modules.values():
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(True)
    model.eval()
    for mod in train_modules.values():
        if mod is not None:
            mod.train()
    model.get_model().get_vision_tower_mask().train()

    captured: Dict[str, object] = {}
    loss_log: Dict[str, float] = {}
    tensor_grads: Dict[str, float] = {}

    _orig_prepare = model._prepare_full_tg_swin_inputs
    _orig_banks = model.predictor.prepare_grouped_banks
    _orig_criterion = model.criterion.forward
    _orig_predictor = model.predictor.forward

    def _capture_prepare(*_args, **_kwargs):
        out = _orig_prepare(*_args, **_kwargs)
        captured["tg_prep"] = out
        if out.text_cond is not None:
            out.text_cond.retain_grad()
        if out.set_gate_b is not None:
            out.set_gate_b.retain_grad()
        return out

    def _capture_banks(*_args, **_kwargs):
        out = _orig_banks(*_args, **_kwargs)
        captured["banks"] = out
        return out

    def _capture_criterion(outputs, targets):
        losses = _orig_criterion(outputs, targets)
        for k, v in losses.items():
            if v is not None:
                loss_log[k] = float(v.detach().item())
        return losses

    def _capture_predictor(*args, **kwargs):
        mo = _orig_predictor(*args, **kwargs)
        if kwargs.get("grouped_setpp_mode"):
            captured["mask_outputs"] = mo
        return mo

    model._prepare_full_tg_swin_inputs = _capture_prepare  # type: ignore[method-assign]
    model.predictor.prepare_grouped_banks = _capture_banks  # type: ignore[method-assign]
    model.criterion.forward = _capture_criterion  # type: ignore[method-assign]
    model.predictor.forward = _capture_predictor  # type: ignore[method-assign]

    model.criterion.set_global_step(5000)

    out = model.forward(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        images=batch["images"],
        images_clip=batch["images_clip"],
        seg_info=batch["seg_info"],
        token_refer_id=batch.get("token_refer_id"),
        SEG_token_embedding_indices=batch["SEG_token_embedding_indices"],
        SET_token_embedding_indices=batch["SET_token_embedding_indices"],
        global_step=5000,
        mask_num=batch["mask_num"],
        dataset_type=batch.get("dataset_type"),
    )

    tg = captured.get("tg_prep")
    assert tg is not None, "missing _prepare_full_tg_swin_inputs capture"
    banks = captured.get("banks")
    assert banks is not None, "missing prepare_grouped_banks capture"
    mem_banks, pos_banks, mf_bank, slot_valid = banks

    B = len(mask_num)
    T = sum(mask_num)
    Kmax = max(mask_num)
    S = int(tg.set_gate_b.shape[1]) if tg.set_gate_b is not None else 0
    C = mem_banks[0].shape[-1]
    HW0 = mem_banks[0].shape[2]

    print(f"[shape] set_gate_b={tuple(tg.set_gate_b.shape)}")
    print(f"[shape] set_gate_t={tuple(tg.set_control.shape)}")
    print(f"[shape] memory_bank_0={tuple(mem_banks[0].shape)}")
    print(f"[shape] grouped_pos_bank_0={tuple(pos_banks[0].shape)}")
    print(f"[shape] mask_features_bank={tuple(mf_bank.shape)}")
    print(f"[shape] valid_seg_mask={tuple(slot_valid.shape)}")

    assert tg.set_gate_b.shape == (B, S, 1)
    assert tg.set_control.shape == (T, S, 1)
    assert mem_banks[0].shape == (B, Kmax, HW0, C)
    assert pos_banks[0].shape == (B, Kmax, HW0, C)
    assert mf_bank.shape[0] == B and mf_bank.shape[1] == Kmax
    assert slot_valid.shape == (B, Kmax)

    mask_outputs = captured.get("mask_outputs")
    assert mask_outputs is not None, "missing grouped predictor outputs"
    H, W = mask_outputs["pred_set_union_mask"].shape[-2:]
    print(f"[shape] decoder queries implied=({B}, {1 + Kmax}, {C})")
    print(f"[shape] pred_set_union_mask={tuple(mask_outputs['pred_set_union_mask'].shape)}")
    print(f"[shape] pred_seg_masks_grouped={tuple(mask_outputs['pred_seg_masks_grouped'].shape)}")
    print(f"[shape] pred_seg_masks={tuple(mask_outputs['pred_seg_masks'].shape)}")
    print(f"[shape] pred_masks={tuple(mask_outputs['pred_masks'].shape)}")

    assert mask_outputs.get("grouped_setpp_mode") is True
    assert mask_outputs.get("per_target_mode") is False
    assert mask_outputs["pred_set_union_mask"].shape == (B, 1, H, W)
    assert mask_outputs["pred_seg_masks_grouped"].shape == (B, Kmax, H, W)
    assert mask_outputs["pred_seg_masks"].shape == (T, 1, H, W)
    assert mask_outputs["pred_masks"].shape == (B, 1 + Kmax, H, W)
    assert not torch.isnan(mask_outputs["pred_masks"]).any()
    assert not torch.isinf(mask_outputs["pred_masks"]).any()
    assert torch.allclose(
        mask_outputs["pred_masks"][:, 0:1], mask_outputs["pred_set_union_mask"], atol=0, rtol=0
    )
    assert torch.allclose(
        mask_outputs["pred_masks"][:, 1:], mask_outputs["pred_seg_masks_grouped"], atol=0, rtol=0
    )
    print("[PASS] pred_masks semantics: [:,0]=set_union, [:,1:]=seg_grouped")
    assert model.criterion._use_regrouped_set_loss(
        {"grouped_setpp_mode": True, "per_target_mode": False}
    ) is False
    print("[PASS] grouped_setpp_mode=True per_target_mode=False _use_regrouped_set_loss=False")

    assert out.loss is not None and torch.isfinite(out.loss).all()
    print(f"[info] total forward loss={float(out.loss.detach().item()):.6e}")

    main_keys = [
        "loss_mask",
        "loss_dice",
        "loss_union_mask",
        "loss_union_dice",
        "loss_setpp_coverage",
        "loss_setpp_consistency",
    ]
    for k in main_keys:
        assert k in loss_log, f"missing main loss key: {k}"
    print(f"[loss] main keys: {[k for k in main_keys if k in loss_log]}")
    for k in main_keys:
        print(f"  {k}={loss_log[k]:.6e}")

    aux_keys = sorted(k for k in loss_log if k not in main_keys)
    seg_aux = [k for k in aux_keys if k.startswith("loss_mask_") or k.startswith("loss_dice_")]
    set_aux = [
        k for k in aux_keys
        if k.startswith("loss_union") or "coverage" in k or "consistency" in k
    ]
    other_aux = [k for k in aux_keys if k not in seg_aux and k not in set_aux]
    print(f"[loss] aux target keys ({len(seg_aux)}): {seg_aux}")
    print(f"[loss] aux set/closure keys ({len(set_aux)}): {set_aux} (expected empty — final-only contract)")
    if other_aux:
        print(f"[loss] aux other keys ({len(other_aux)}): {other_aux}")
    assert not set_aux, "aux set/closure losses should not be computed"

    model.zero_grad(set_to_none=True)
    out.loss.backward()
    backward_ok = True
    grad_report = {}
    nan_any = inf_any = False
    grad_eps = 1e-12
    for name, mod in train_modules.items():
        if mod is None:
            grad_report[name] = "N/A"
            continue
        peak = _max_grad(mod)
        grad_report[name] = peak
        n, i = _has_nan_inf(mod)
        nan_any = nan_any or n
        inf_any = inf_any or i
        if name in (
            "SEG_token_projector",
            "SET_token_projector",
            "predictor",
            "tg_swin_tcf",
            "tg_swin_controller",
            "tg_swin_set_control",
        ) and peak <= grad_eps:
            backward_ok = False
    if model.predictor is not None and model.predictor.set_union_head is not None:
        sun = _max_grad(model.predictor.set_union_head)
        grad_report["SetUnionMaskHead"] = sun
        if sun <= grad_eps:
            backward_ok = False

    tg_after = captured.get("tg_prep")
    if tg_after is not None:
        if tg_after.text_cond is not None and tg_after.text_cond.grad is not None:
            tensor_grads["text_cond"] = float(tg_after.text_cond.grad.abs().max().item())
        if tg_after.set_gate_b is not None and tg_after.set_gate_b.grad is not None:
            tensor_grads["set_gate_b"] = float(tg_after.set_gate_b.grad.abs().max().item())
    if tensor_grads:
        print(f"[grad] intermediate tensors: {tensor_grads}")

    print(f"[grad] modules (max-abs): {grad_report}")
    print(f"[info] backward_ok={backward_ok} nan={nan_any} inf={inf_any}")

    if not backward_ok or nan_any or inf_any:
        sys.exit(1)
    print("[PASS] smoke_e2e_grouped_setpp_forward")


if __name__ == "__main__":
    main()
