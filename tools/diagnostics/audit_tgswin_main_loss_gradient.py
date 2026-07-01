#!/usr/bin/env python3
"""No-probe TG-Swin main-loss gradient audit for grouped SET++ Full path."""

from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
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

MAIN_LOSS_KEYS = (
    "loss_mask",
    "loss_dice",
    "loss_union_mask",
    "loss_union_dice",
    "loss_setpp_coverage",
    "loss_setpp_consistency",
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


def _tensor_grad_report(name: str, t: Optional[torch.Tensor]) -> Dict[str, Any]:
    if t is None:
        return {"name": name, "present": False}
    rep = {
        "name": name,
        "present": True,
        "shape": tuple(t.shape),
        "requires_grad": bool(t.requires_grad),
        "grad_is_none": t.grad is None,
    }
    if t.grad is not None:
        rep["grad_max"] = float(t.grad.detach().abs().max().item())
        rep["grad_mean"] = float(t.grad.detach().abs().mean().item())
    else:
        rep["grad_max"] = None
        rep["grad_mean"] = None
    return rep


def _print_tensor_report(rep: Dict[str, Any]) -> None:
    if not rep.get("present"):
        print(f"[tensor] {rep['name']}: MISSING")
        return
    gmax = rep["grad_max"]
    gmean = rep["grad_mean"]
    print(
        f"[tensor] {rep['name']} shape={rep['shape']} "
        f"requires_grad={rep['requires_grad']} grad_is_none={rep['grad_is_none']} "
        f"grad_max={gmax} grad_mean={gmean}"
    )


def _max_grad(module: Optional[torch.nn.Module]) -> float:
    if module is None:
        return 0.0
    peak = 0.0
    for p in module.parameters():
        if p.grad is not None:
            peak = max(peak, float(p.grad.detach().abs().max().item()))
    return peak


def _retain(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is not None and t.requires_grad:
        t.retain_grad()
    return t


def _build_model(device: torch.device, dtype: torch.dtype):
    if os.path.isdir(DEFAULT_MODEL):
        model_path = DEFAULT_MODEL
    elif os.path.isdir(FALLBACK_MODEL):
        print(f"[warn] preferred model missing, fallback {FALLBACK_MODEL}")
        model_path = FALLBACK_MODEL
    else:
        raise FileNotFoundError(f"model not found: {DEFAULT_MODEL}")

    cfg = get_mask_config(FULL_CFG)
    tg = cfg.TG_SWIN
    assert tg.ENABLED and tg.GROUPED_SETPP_DECODER
    assert not getattr(tg, "USE_COARSE_EVIDENCE", False)
    assert not getattr(tg, "USE_DR_EWTI", False)

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
    prefixes = (
        "predictor.",
        "pixel_decoder.",
        "SEG_token_projector.",
        "SET_token_projector.",
        "tg_swin_",
    )
    saved = _load_mask_state_from_merged(model_path, prefixes)
    model = SegEarthR2.from_pretrained(
        model_path,
        mask_decoder_cfg=cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.initial_mask_module(VISION_TOWER_MASK, model_args)
    model.load_state_dict(saved, strict=False)
    model.persist_setpp_config(model_args)
    model.get_model().initialize_vision_modules(model_args=model_args)
    model.to(device=device, dtype=dtype)
    if model.get_vision_tower() is not None:
        model.get_vision_tower().to(device=device, dtype=dtype)
    model.get_model().get_vision_tower_mask().to(device=device, dtype=dtype)
    return model, cfg, model_path, model_args


def _setup_trainable(model: SegEarthR2) -> Dict[str, torch.nn.Module]:
    for p in model.parameters():
        p.requires_grad_(False)
    modules = {
        "tg_swin_tcf": model.tg_swin_tcf,
        "tg_swin_controller": model.tg_swin_controller,
        "tg_swin_set_control": model.tg_swin_set_control,
        "SEG_token_projector": model.SEG_token_projector,
        "SET_token_projector": model.SET_token_projector,
        "pixel_decoder": model.pixel_decoder,
        "predictor": model.predictor,
    }
    for mod in modules.values():
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(True)
    model.eval()
    for mod in modules.values():
        if mod is not None:
            mod.train()
    model.get_model().get_vision_tower_mask().train()
    return modules


def _load_batch(model: SegEarthR2, model_args, device: torch.device):
    data_args = SimpleNamespace(
        lasers_holdout_ratio=0.05,
        lasers_holdout_seed=42,
        image_aspect_ratio="square",
        image_grid_pinpoints=None,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args if isinstance(model_args, str) else DEFAULT_MODEL,
        use_fast=False,
        model_max_length=2048,
        padding_side="right",
    )
    return tokenizer, data_args


def run_main_loss_audit(model: SegEarthR2, batch: Dict[str, Any], global_step: int = 5000):
    captured: Dict[str, Any] = {}
    raw_losses: Dict[str, torch.Tensor] = {}

    _orig_prepare = model._prepare_full_tg_swin_inputs
    _orig_vision = model.get_vision_tower_feature
    _orig_pixel = model.pixel_decoder.forward_features
    _orig_banks = model.predictor.prepare_grouped_banks
    _orig_compute_bias = model.tg_swin_controller.compute_bias
    _orig_criterion = model.criterion.forward

    def _capture_prepare(*args, **kwargs):
        out = _orig_prepare(*args, **kwargs)
        _retain(out.text_cond)
        _retain(out.reliability)
        _retain(out.set_gate_b)
        _retain(out.set_control)
        captured["tg_prep"] = out
        return out

    def _capture_vision(images, **kwargs):
        feats = _orig_vision(images, **kwargs)
        res2 = feats["res2"]
        _retain(res2)
        captured["swin_res2"] = res2
        captured["vision_feats_dict"] = feats
        return feats

    def _capture_pixel(features):
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                _retain(v)
        out = _orig_pixel(features)
        mask_features, _, multi_scale = out
        _retain(mask_features)
        if multi_scale:
            _retain(multi_scale[0])
        captured["pixel_in"] = features
        captured["pixel_mask_features"] = mask_features
        captured["pixel_ms0"] = multi_scale[0] if multi_scale else None
        return out

    def _capture_banks(ms, mf, mask_num):
        out = _orig_banks(ms, mf, mask_num)
        mem0 = out[0][0]
        mf_bank = out[2]
        _retain(mem0)
        _retain(mf_bank)
        captured["memory_bank_0"] = mem0
        captured["mask_features_bank"] = mf_bank
        captured["banks"] = out
        return out

    def _capture_compute_bias(*args, **kwargs):
        bias = _orig_compute_bias(*args, **kwargs)
        if bias is not None:
            _retain(bias)
            captured["attn_text_bias"] = bias
        return bias

    def _capture_criterion(outputs, targets):
        losses = _orig_criterion(outputs, targets)
        for k, v in losses.items():
            if v is not None and isinstance(v, torch.Tensor):
                raw_losses[k] = v
        return losses

    model._prepare_full_tg_swin_inputs = _capture_prepare  # type: ignore[method-assign]
    model.get_vision_tower_feature = _capture_vision  # type: ignore[method-assign]
    model.pixel_decoder.forward_features = _capture_pixel  # type: ignore[method-assign]
    model.predictor.prepare_grouped_banks = _capture_banks  # type: ignore[method-assign]
    model.tg_swin_controller.compute_bias = _capture_compute_bias  # type: ignore[method-assign]
    model.criterion.forward = _capture_criterion  # type: ignore[method-assign]
    model.criterion.set_global_step(global_step)

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
        global_step=global_step,
        mask_num=batch["mask_num"],
        dataset_type=batch.get("dataset_type"),
    )

    mask_only = torch.tensor(0.0, device=out.loss.device if out.loss is not None else batch["input_ids"].device)
    for k in MAIN_LOSS_KEYS:
        if k not in raw_losses:
            raise RuntimeError(f"missing main loss key: {k}")
        mask_only = mask_only + raw_losses[k]

    model.zero_grad(set_to_none=True)
    backward_ok = True
    try:
        mask_only.backward(retain_graph=False)
    except Exception as exc:
        backward_ok = False
        captured["backward_error"] = str(exc)

    tg = captured.get("tg_prep")
    tensor_reports = [
        _tensor_grad_report("text_cond", tg.text_cond if tg else None),
        _tensor_grad_report("reliability", tg.reliability if tg else None),
        _tensor_grad_report("set_gate_b", tg.set_gate_b if tg else None),
        _tensor_grad_report("set_control(set_gate_t)", tg.set_control if tg else None),
        _tensor_grad_report("attn_text_bias", captured.get("attn_text_bias")),
        _tensor_grad_report("swin_res2", captured.get("swin_res2")),
    ]
    pixel_in = captured.get("pixel_in") or {}
    if "res2" in pixel_in:
        tensor_reports.append(_tensor_grad_report("pixel_in_res2", pixel_in["res2"]))
    tensor_reports.extend(
        [
            _tensor_grad_report("pixel_mask_features", captured.get("pixel_mask_features")),
            _tensor_grad_report("pixel_ms0", captured.get("pixel_ms0")),
            _tensor_grad_report("memory_bank_0", captured.get("memory_bank_0")),
            _tensor_grad_report("mask_features_bank", captured.get("mask_features_bank")),
        ]
    )

    # restore hooks before returning
    model._prepare_full_tg_swin_inputs = _orig_prepare  # type: ignore[method-assign]
    model.get_vision_tower_feature = _orig_vision  # type: ignore[method-assign]
    model.pixel_decoder.forward_features = _orig_pixel  # type: ignore[method-assign]
    model.predictor.prepare_grouped_banks = _orig_banks  # type: ignore[method-assign]
    model.tg_swin_controller.compute_bias = _orig_compute_bias  # type: ignore[method-assign]
    model.criterion.forward = _orig_criterion  # type: ignore[method-assign]

    return {
        "out": out,
        "mask_only_loss": float(mask_only.detach().item()),
        "raw_losses": {k: float(v.detach().item()) for k, v in raw_losses.items()},
        "backward_ok": backward_ok,
        "tensor_reports": tensor_reports,
        "captured": captured,
    }


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    d = (a.detach().float() - b.detach().float()).abs()
    return {"max": float(d.max().item()), "mean": float(d.mean().item())}


def run_sensitivity(model: SegEarthR2, batch: Dict[str, Any], eps: float = 0.05) -> Dict[str, Any]:
    """Functional sensitivity: perturb text_cond / set_gate_t before vision forward."""

    baseline_preds: Dict[str, torch.Tensor] = {}
    perturbed: Dict[str, Dict[str, float]] = {}

    _orig_vision = model.get_vision_tower_feature
    hook_state: Dict[str, Any] = {"mode": "baseline", "delta": None}

    def _vision_hook(images, text_cond=None, reliability=None, set_control=None, **kwargs):
        tc = text_cond
        sc = set_control
        if hook_state["mode"] == "text_cond" and tc is not None:
            tc = tc + hook_state["delta"]
        elif hook_state["mode"] == "set_gate" and sc is not None:
            sc = sc + hook_state["delta"]
        elif hook_state["mode"] == "set_gate_zero":
            sc = torch.zeros_like(sc)
        elif hook_state["mode"] == "set_gate_one":
            sc = torch.ones_like(sc)
        return _orig_vision(
            images,
            text_cond=tc,
            reliability=reliability,
            set_control=sc,
            **kwargs,
        )

    def _forward_preds() -> Dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)
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
        mo = out  # we need mask outputs — hook predictor
        return {}

    model.get_vision_tower_feature = _vision_hook  # type: ignore[method-assign]

    captured_mo: Dict[str, Any] = {}
    _orig_predictor = model.predictor.forward

    def _cap_pred(*args, **kwargs):
        mo = _orig_predictor(*args, **kwargs)
        if kwargs.get("grouped_setpp_mode"):
            captured_mo["mo"] = {k: mo[k].detach().clone() for k in (
                "pred_set_union_mask", "pred_seg_masks_grouped", "pred_seg_masks"
            )}
        return mo

    model.predictor.forward = _cap_pred  # type: ignore[method-assign]

    def _run_once(mode: str, delta=None):
        hook_state["mode"] = mode
        hook_state["delta"] = delta
        captured_mo.clear()
        model.forward(
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
        return captured_mo.get("mo")

    base = _run_once("baseline")
    if base is None:
        raise RuntimeError("failed to capture baseline predictions")

    tg_prep = None
    _orig_prep = model._prepare_full_tg_swin_inputs

    def _prep(*a, **k):
        nonlocal tg_prep
        tg_prep = _orig_prep(*a, **k)
        return tg_prep

    model._prepare_full_tg_swin_inputs = _prep  # type: ignore[method-assign]
    _run_once("baseline")
    assert tg_prep is not None

    tc_delta = eps * torch.randn_like(tg_prep.text_cond)
    sc_delta = eps * torch.randn_like(tg_prep.set_control)

    for mode, delta, label in (
        ("text_cond", tc_delta, "text_cond+noise"),
        ("set_gate", sc_delta, "set_gate_t+noise"),
        ("set_gate_zero", None, "set_gate_t=0"),
        ("set_gate_one", None, "set_gate_t=1"),
    ):
        pred = _run_once(mode, delta)
        if pred is None:
            continue
        perturbed[label] = {
            "pred_set_union_mask": _diff_stats(base["pred_set_union_mask"], pred["pred_set_union_mask"]),
            "pred_seg_masks_grouped": _diff_stats(base["pred_seg_masks_grouped"], pred["pred_seg_masks_grouped"]),
            "pred_seg_masks": _diff_stats(base["pred_seg_masks"], pred["pred_seg_masks"]),
        }

    model.get_vision_tower_feature = _orig_vision  # type: ignore[method-assign]
    model.predictor.forward = _orig_predictor  # type: ignore[method-assign]
    model._prepare_full_tg_swin_inputs = _orig_prep  # type: ignore[method-assign]
    return perturbed


def main():
    os.chdir(REPO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"[audit] yaml={FULL_CFG}")
    model, cfg, model_path, model_args = _build_model(device, dtype)
    modules = _setup_trainable(model)
    print(f"[audit] model={model_path}")

    data_args = SimpleNamespace(
        lasers_holdout_ratio=0.05,
        lasers_holdout_seed=42,
        image_aspect_ratio="square",
        image_grid_pinpoints=None,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, use_fast=False, model_max_length=2048, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for tok in ("[SEG]", "[SET]"):
        if tok not in tokenizer.get_vocab():
            tokenizer.add_tokens([tok])
    clip_processor = transformers.SiglipImageProcessor.from_pretrained(VISION_TOWER)
    dataset = LaSeRSDataset(
        LASERS_PATH, tokenizer, data_args, split="train_data.json", holdout_mode="train"
    )
    idxs = _pick_lasers_batch(dataset, (2, 3))
    instances = [dataset[i] for i in idxs]
    mask_num = [inst["mask_num"] for inst in instances]
    B, T, Kmax = len(mask_num), sum(mask_num), max(mask_num)
    print(f"[audit] batch B={B} mask_num={mask_num} T={T} Kmax={Kmax} indices={idxs}")

    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_processor)
    batch = collator(instances)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            batch[k] = [t.to(device) for t in v]

    result = run_main_loss_audit(model, batch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[audit] mask-only loss={result['mask_only_loss']:.6e}")
    print("[audit] loss components:")
    for k in MAIN_LOSS_KEYS:
        print(f"  {k}={result['raw_losses'][k]:.6e}")

    print("[audit] tensor gradients (main loss only, no probe):")
    for rep in result["tensor_reports"]:
        _print_tensor_report(rep)

    mod_grad = {name: _max_grad(mod) for name, mod in modules.items()}
    if model.predictor is not None and model.predictor.set_union_head is not None:
        mod_grad["SetUnionMaskHead"] = _max_grad(model.predictor.set_union_head)
    print("[audit] module max-abs grad:")
    for name, g in mod_grad.items():
        print(f"  {name}={g}")

    nan_inf = False
    for mod in list(modules.values()) + ([model.predictor.set_union_head] if model.predictor else []):
        if mod is None:
            continue
        for p in mod.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    nan_inf = True
    print(f"[audit] backward_ok={result['backward_ok']} nan_or_inf_grad={nan_inf}")

    print("[audit] sensitivity checks:")
    sens = run_sensitivity(model, batch)
    for label, stats in sens.items():
        print(f"  [{label}]")
        for k, v in stats.items():
            print(f"    {k}: max={v['max']:.6e} mean={v['mean']:.6e}")

    tg_pass = (
        (result["tensor_reports"][0]["grad_is_none"] is False and (result["tensor_reports"][0]["grad_max"] or 0) > 0)
        or mod_grad["tg_swin_tcf"] > 0
    )
    set_pass = (
        (result["tensor_reports"][2]["grad_is_none"] is False and (result["tensor_reports"][2]["grad_max"] or 0) > 0)
        or mod_grad["tg_swin_set_control"] > 0
    )
    ctrl_pass = mod_grad["tg_swin_controller"] > 0
    sens_pass = any(v["pred_set_union_mask"]["max"] > 1e-6 for v in sens.values())
    overall = (
        result["backward_ok"]
        and not nan_inf
        and tg_pass
        and set_pass
        and ctrl_pass
        and sens_pass
    )
    print(f"[audit] TG-Swin main-loss gradient: {'PASS' if overall else 'FAIL'}")
    if not overall:
        print("[audit] failure hints:")
        if not tg_pass:
            print("  - text_cond / tg_swin_tcf: no main-loss gradient")
        if not set_pass:
            print("  - set_gate / tg_swin_set_control: no main-loss gradient")
        if not ctrl_pass:
            print("  - tg_swin_controller: no main-loss gradient")
        if not sens_pass:
            print("  - functional sensitivity: predictions unchanged under perturbation")


if __name__ == "__main__":
    main()
