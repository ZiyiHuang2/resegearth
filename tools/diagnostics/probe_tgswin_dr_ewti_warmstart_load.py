#!/usr/bin/env python3
"""DR-EWTI init load: base-8w or v1.5 merged + dr-ewti config must succeed."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEG_ROOT = os.path.abspath(os.path.join(REPO, ".."))
DEFAULT_BASE_8W = os.path.join(
    RESEG_ROOT,
    "output/base/standard-base-lasers-siglip1-8w-gd4/merged_model",
)
DEFAULT_V15_10W = os.path.join(
    RESEG_ROOT,
    "output/tgswin/tgswin-wti-v15-lasers-warmstart-10w-bs2-gd4/merged_model",
)
sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2

DR_EWTI_CFG = "segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_dr_ewti.yaml"


def _load_sharded_safetensors(model_dir: str) -> dict:
    from safetensors.torch import load_file

    with open(os.path.join(model_dir, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update(load_file(os.path.join(model_dir, shard)))
    return state


def _classify_init_ckpt(ckpt_sd: dict) -> str:
    tcf_ckpt = {k: v for k, v in ckpt_sd.items() if k.startswith("tg_swin_tcf.")}
    wti_ckpt = {k: v for k, v in ckpt_sd.items() if k.startswith("tg_swin_controller.wti_blocks.")}
    dr_ckpt = {k: v for k, v in ckpt_sd.items() if "dr_wti_blocks" in k}
    if dr_ckpt:
        raise AssertionError("checkpoint already contains dr_wti_blocks; use a base or v1.5 ckpt")
    if tcf_ckpt or wti_ckpt:
        assert len(wti_ckpt) == 15, f"expected 15 wti keys, got {len(wti_ckpt)}"
        assert len(tcf_ckpt) > 0, "partial tg_swin checkpoint (wti without tcf)"
        return "v15"
    return "base"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_BASE_8W)
    args = parser.parse_args()
    os.chdir(REPO)

    if not os.path.isdir(args.model_path):
        print(f"[ERROR] init checkpoint not found: {args.model_path}")
        sys.exit(1)

    dr_cfg = get_mask_config(DR_EWTI_CFG)
    assert int(dr_cfg.TG_SWIN.NUM_STAGES) == 4
    assert list(dr_cfg.TG_SWIN.WTI_STAGES) == [1, 2, 3]

    ckpt_sd = _load_sharded_safetensors(args.model_path)
    init_mode = _classify_init_ckpt(ckpt_sd)
    tcf_ckpt = {k: v for k, v in ckpt_sd.items() if k.startswith("tg_swin_tcf.")}
    wti_ckpt = {k: v for k, v in ckpt_sd.items() if k.startswith("tg_swin_controller.wti_blocks.")}
    print(f"[info] init mode={init_mode} (tcf={len(tcf_ckpt)} wti={len(wti_ckpt)} keys)")

    print(f"[info] loading SegEarthR2.from_pretrained({args.model_path}) with dr-ewti cfg")
    model = SegEarthR2.from_pretrained(
        args.model_path,
        mask_decoder_cfg=dr_cfg,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )

    tcf = model.tg_swin_tcf.router
    assert tcf.stage_embed.shape == (4, int(dr_cfg.TG_SWIN.COND_DIM)), tcf.stage_embed.shape
    assert tcf.stage_router[2].weight.shape[0] == 12
    print(f"[PASS] TCF shapes: stage_embed={tuple(tcf.stage_embed.shape)}, router_out=12")

    ctrl_sd = model.tg_swin_controller.state_dict()
    wti_model = {k: v for k, v in ctrl_sd.items() if k.startswith("wti_blocks.")}
    dr_model = {k: v for k, v in ctrl_sd.items() if k.startswith("dr_wti_blocks.")}
    assert len(wti_model) == 15
    assert len(dr_model) == 39

    if init_mode == "v15":
        for ckpt_key, ckpt_val in wti_ckpt.items():
            local_key = ckpt_key.split("tg_swin_controller.", 1)[1]
            assert local_key in wti_model, f"missing wti param {local_key}"
            torch.testing.assert_close(
                ckpt_val.float(),
                wti_model[local_key].float(),
                msg=f"wti weight mismatch: {local_key}",
            )
        print("[PASS] 15 WTI params match v1.5 checkpoint values")

        for k in dr_model:
            assert k not in {x.split("tg_swin_controller.", 1)[-1] for x in ckpt_sd if "tg_swin_controller" in x}
        print("[PASS] 39 evidence params newly initialized (not in v1.5 ckpt)")

        subset = {
            k.split("tg_swin_controller.", 1)[1]: v
            for k, v in ckpt_sd.items()
            if k.startswith("tg_swin_controller.wti_blocks.")
        }
        probe = type(model.tg_swin_controller)(
            cond_dim=int(dr_cfg.TG_SWIN.COND_DIM),
            wti_rank=int(dr_cfg.TG_SWIN.WTI_RANK),
            wti_stages=list(dr_cfg.TG_SWIN.WTI_STAGES),
            head_aware=True,
            window_size=int(dr_cfg.MODEL.SWIN.WINDOW_SIZE),
            use_dr_ewti=True,
        )
        missing, unexpected = probe.load_state_dict(subset, strict=False)
        assert not unexpected
        assert all(m.startswith("dr_wti_blocks.") for m in missing)
        assert len(missing) == 39
        print(f"[PASS] controller load_state_dict: missing={len(missing)} dr-only, unexpected=0")
    else:
        print("[PASS] base init: TCF + WTI + DR-EWTI modules randomly initialized in-model")
        assert len(tcf_ckpt) == 0 and len(wti_ckpt) == 0

    print("[PASS] probe_tgswin_dr_ewti_warmstart_load")


if __name__ == "__main__":
    main()
