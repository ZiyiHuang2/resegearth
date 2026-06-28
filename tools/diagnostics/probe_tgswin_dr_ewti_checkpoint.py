#!/usr/bin/env python3
"""Checkpoint-level DR-EWTI tests: real v1.5 merged weights, save/reload, config."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEG_ROOT = os.path.abspath(os.path.join(REPO, ".."))
DEFAULT_V15_MERGED = os.path.join(
    RESEG_ROOT,
    "output/tgswin/tgswin-wti-v15-lasers-warmstart-2w-bs2-gd4/merged_model",
)
sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.mask_encoder.tg_swin import TGSwimController

DR_EWTI_CFG = "segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_dr_ewti.yaml"
V15_CFG = "segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_v15.yaml"


def _load_sharded_safetensors(model_dir: str) -> dict:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"no safetensors index: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    state = {}
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("safetensors required for checkpoint probe") from exc
    for shard in shard_files:
        shard_path = os.path.join(model_dir, shard)
        state.update(load_file(shard_path))
    return state


def test_real_v15_wti_load_readonly(merged_dir: str):
    full_sd = _load_sharded_safetensors(merged_dir)
    wti_sd = {k: v for k, v in full_sd.items() if k.startswith("tg_swin_controller.wti_blocks.")}
    assert len(wti_sd) == 15, f"expected 15 v1.5 wti keys, got {len(wti_sd)}"
    assert not any("dr_wti_blocks" in k for k in full_sd)
    assert not any("base_wti" in k for k in full_sd)

    dr_cfg = get_mask_config(DR_EWTI_CFG)
    ctrl = TGSwimController(
        cond_dim=int(dr_cfg.TG_SWIN.COND_DIM),
        wti_rank=int(dr_cfg.TG_SWIN.WTI_RANK),
        wti_stages=list(dr_cfg.TG_SWIN.WTI_STAGES),
        head_aware=True,
        window_size=int(dr_cfg.MODEL.SWIN.WINDOW_SIZE),
        use_dr_ewti=True,
    )
    stripped = {k.split("tg_swin_controller.wti_blocks.")[1]: v for k, v in wti_sd.items()}
    missing, unexpected = ctrl.wti_blocks.load_state_dict(stripped, strict=True)
    assert not missing and not unexpected
    controller_subset = {
        k.split("tg_swin_controller.", 1)[1]: v
        for k, v in full_sd.items()
        if k.startswith("tg_swin_controller.")
    }
    missing_dr, unexpected_dr = ctrl.load_state_dict(controller_subset, strict=False)
    assert all(m.startswith("dr_wti_blocks.") for m in missing_dr)
    assert not unexpected_dr
    print(f"[PASS] real v1.5 merged checkpoint: {len(wti_sd)} wti keys loaded; missing={len(missing_dr)} dr-only")


def test_dr_save_reload_evidence_keys(merged_dir: str):
    dr_cfg = get_mask_config(DR_EWTI_CFG)
    v15_sd = _load_sharded_safetensors(merged_dir)
    wti_sd = {
        k.split("tg_swin_controller.wti_blocks.")[1]: v
        for k, v in v15_sd.items()
        if k.startswith("tg_swin_controller.wti_blocks.")
    }

    ctrl = TGSwimController(
        cond_dim=int(dr_cfg.TG_SWIN.COND_DIM),
        wti_rank=int(dr_cfg.TG_SWIN.WTI_RANK),
        wti_stages=list(dr_cfg.TG_SWIN.WTI_STAGES),
        head_aware=True,
        window_size=int(dr_cfg.MODEL.SWIN.WINDOW_SIZE),
        use_dr_ewti=True,
    )
    ctrl.wti_blocks.load_state_dict(wti_sd, strict=True)

    evidence_sd = {k: v.clone() for k, v in ctrl.state_dict().items() if k.startswith("dr_wti_blocks.")}
    assert len(evidence_sd) == 39

    tmp = tempfile.mkdtemp(prefix="dr_ewti_ckpt_")
    try:
        torch.save(ctrl.state_dict(), os.path.join(tmp, "tg_swin_controller.pt"))
        meta = {
            "TG_SWIN_VERSION": "dr-ewti",
            "USE_COARSE_EVIDENCE": True,
            "mask_config": DR_EWTI_CFG,
        }
        with open(os.path.join(tmp, "tg_swin_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        reloaded_cfg = get_mask_config(os.path.join(REPO, DR_EWTI_CFG))
        with open(os.path.join(tmp, "tg_swin_meta.json"), "r", encoding="utf-8") as f:
            loaded_meta = json.load(f)
        assert loaded_meta["TG_SWIN_VERSION"] == "dr-ewti"
        assert loaded_meta["USE_COARSE_EVIDENCE"] is True
        assert str(reloaded_cfg.TG_SWIN.VERSION) == "dr-ewti"
        assert bool(reloaded_cfg.TG_SWIN.USE_COARSE_EVIDENCE)

        fresh = TGSwimController(
            cond_dim=int(reloaded_cfg.TG_SWIN.COND_DIM),
            wti_rank=int(reloaded_cfg.TG_SWIN.WTI_RANK),
            wti_stages=list(reloaded_cfg.TG_SWIN.WTI_STAGES),
            head_aware=True,
            window_size=int(reloaded_cfg.MODEL.SWIN.WINDOW_SIZE),
            use_dr_ewti=True,
        )
        fresh.load_state_dict(torch.load(os.path.join(tmp, "tg_swin_controller.pt"), map_location="cpu"), strict=True)
        for k, v in evidence_sd.items():
            torch.testing.assert_close(v.float(), fresh.state_dict()[k].float())
        for k, v in wti_sd.items():
            torch.testing.assert_close(v.float(), fresh.state_dict()[f"wti_blocks.{k}"].float())
        print(f"[PASS] checkpoint merge round-trip: v1.5 wti + {len(evidence_sd)} evidence keys + dr-ewti yaml")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_merge_script_import_and_arch_resolve():
    from segearth_r2.train.merge_lora_weights_and_save_hf_model import resolve_merge_arch_path

    ckpt = os.path.join(RESEG_ROOT, "output/tgswin/tgswin-wti-v15-lasers-warmstart-2w-bs2-gd4/checkpoint-20000")
    if os.path.isdir(ckpt):
        resolved = resolve_merge_arch_path(ckpt)
        assert os.path.isdir(resolved)
        print(f"[PASS] merge script resolve_merge_arch_path: {resolved}")
    else:
        print("[SKIP] merge script arch resolve — checkpoint-20000 not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15-merged", default=DEFAULT_V15_MERGED)
    args = parser.parse_args()
    os.chdir(REPO)

    if not os.path.isdir(args.v15_merged):
        print(f"[ERROR] v1.5 merged model not found: {args.v15_merged}")
        sys.exit(1)

    test_real_v15_wti_load_readonly(args.v15_merged)
    test_dr_save_reload_evidence_keys(args.v15_merged)
    test_merge_script_import_and_arch_resolve()
    print("[PASS] probe_tgswin_dr_ewti_checkpoint")


if __name__ == "__main__":
    main()
