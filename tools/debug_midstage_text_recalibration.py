import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.model.mipha.model.language_model.configuration_mipha import MiphaPhiConfig
from segearth_r2.utils.builder import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser("Debug MidStage Text Recalibration")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--mask_config",
        type=str,
        default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
    )
    parser.add_argument(
        "--vision_tower",
        type=str,
        default="pretrained_model/CLIP/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--vision_tower_mask",
        type=str,
        default="pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl",
    )
    parser.add_argument("--swin_type", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_names", type=int, default=8)
    parser.add_argument("--init_only", action="store_true", help="Build model from config without loading HF weights")
    parser.add_argument("--skip_vision_weights", action="store_true", help="Initialize Swin without loading vision ckpt")
    parser.add_argument("--norm_only", action="store_true", help="Apply trainable recipe with stage3 norm only")
    parser.add_argument(
        "--apply_trainable_recipe",
        action="store_true",
        help="Freeze Swin first, then unfreeze midstage + stage3 partial params for verification",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def to_device(model, device, dtype):
    model.to(device=device, dtype=dtype)
    return model


def load_local_mipha_config(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    model_type = cfg_dict.get("model_type")
    if model_type != "mipha_phi":
        print(f"[WARN] expected model_type=mipha_phi but got: {model_type}")

    cfg = MiphaPhiConfig(**cfg_dict)
    if not hasattr(cfg, "mm_vision_tower") or cfg.mm_vision_tower is None:
        cfg.mm_vision_tower = cfg_dict.get("mm_vision_tower", "")
        print("[WARN] mm_vision_tower missing in config object, fallback to config.json value")
    if not hasattr(cfg, "swin_type") or cfg.swin_type is None:
        cfg.swin_type = cfg_dict.get("swin_type", "base")
        print("[WARN] swin_type missing in config object, fallback to 'base' or config.json value")
    return cfg


def build_model(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")

    if args.init_only:
        cfg = load_local_mipha_config(args.model_path)
        if args.vision_tower:
            cfg.mm_vision_tower = args.vision_tower
        if args.swin_type:
            cfg.swin_type = args.swin_type
        mask_cfg = get_mask_config(args.mask_config)
        model = SegEarthR2(cfg, mask_decoder_cfg=mask_cfg, add_cross_attn=True)
        tokenizer = None
    else:
        model_args = SimpleNamespace(
            seg_task="instance",
            vision_tower=args.vision_tower,
            vision_tower_mask=args.vision_tower_mask,
            swin_type=args.swin_type,
        )
        tokenizer, model, _, _ = load_pretrained_model(
            model_path=args.model_path,
            model_args=model_args,
            mask_config=args.mask_config,
            device=args.device,
        )

    init_vision_ckpt = None if args.skip_vision_weights else args.vision_tower_mask
    init_args = SimpleNamespace(
        vision_tower=args.vision_tower,
        vision_tower_mask=init_vision_ckpt,
        swin_type=args.swin_type,
    )
    model.get_model().initialize_vision_modules(init_args)
    model.eval()
    return tokenizer, model


def apply_midstage_trainable_recipe(model, norm_only=False):
    vision_tower_mask = model.get_model().get_vision_tower_mask()
    if vision_tower_mask is None:
        return
    vision_tower_mask.requires_grad_(False)
    if hasattr(vision_tower_mask, "mid_stage_text_recalibration"):
        for p in vision_tower_mask.mid_stage_text_recalibration.parameters():
            p.requires_grad = True
    if hasattr(vision_tower_mask, "layers") and len(vision_tower_mask.layers) > 2:
        stage3 = vision_tower_mask.layers[2]
        for module in stage3.modules():
            if isinstance(module, torch.nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True
        if hasattr(vision_tower_mask, "norm2"):
            for p in vision_tower_mask.norm2.parameters():
                p.requires_grad = True
        if not norm_only and hasattr(stage3, "blocks") and len(stage3.blocks) > 0:
            for p in stage3.blocks[-1].parameters():
                p.requires_grad = True


def print_tensor_info(name, value):
    if value is None:
        print(f"{name}: None")
        return
    print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")


def check_build_text_condition(model, device):
    ok = True
    print("\n=== Check: build_text_condition compatibility ===")
    pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is None and hasattr(model, "_resolve_pad_token_id"):
        pad_id = model._resolve_pad_token_id()
    if pad_id is None:
        pad_id = 0
    print(f"pad_id used in test input: {pad_id}")

    test_cases = {
        "none": (None, 2),
        "1d_tensor": (torch.tensor([1, 2, 3], dtype=torch.long), 1),
        "2d_padded_tensor": (torch.tensor([[1, 2, pad_id, pad_id], [3, 4, 5, pad_id]], dtype=torch.long), 2),
        "list_tensor": ([torch.tensor([1, 2], dtype=torch.long), torch.tensor([], dtype=torch.long)], 2),
    }

    for case_name, (token_refer_id, batch_size) in test_cases.items():
        try:
            out = model.build_text_condition(token_refer_id=token_refer_id, batch_size=batch_size, device=device)
            if out is None:
                print(f"[PASS] {case_name}: output=None")
            else:
                print(f"[PASS] {case_name}: output=tensor")
                print_tensor_info(f"  {case_name}", out)
        except Exception as e:
            ok = False
            print(f"[FAIL] {case_name}: {repr(e)}")

    return ok


def _feature_shapes(feats):
    if isinstance(feats, (list, tuple)):
        return [tuple(x.shape) for x in feats]
    return None


def check_vision_tower_forward(model, device, dtype, image_size, batch_size):
    ok_none = True
    ok_text = True
    print("\n=== Check: vision_tower_mask forward ===")
    vision_tower_mask = model.get_model().get_vision_tower_mask()
    images = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)

    outs_none = None
    try:
        with torch.no_grad():
            outs_none = vision_tower_mask(images, text_cond=None)
        shapes_none = _feature_shapes(outs_none)
        if not isinstance(outs_none, (list, tuple)) or len(outs_none) != 4:
            ok_none = False
            print(f"[FAIL] text_cond=None: expected 4 features, got {type(outs_none)} len={len(outs_none) if isinstance(outs_none, (list, tuple)) else 'n/a'}")
        else:
            print(f"[PASS] text_cond=None: 4 features, shapes={shapes_none}")
    except Exception as e:
        ok_none = False
        print(f"[FAIL] text_cond=None forward error: {repr(e)}")

    try:
        text_cond = torch.randn(batch_size, model.config.hidden_size, device=device, dtype=dtype)
        with torch.no_grad():
            outs_text = vision_tower_mask(images, text_cond=text_cond)
        shapes_text = _feature_shapes(outs_text)
        if not isinstance(outs_text, (list, tuple)) or len(outs_text) != 4:
            ok_text = False
            print(f"[FAIL] text_cond=random: expected 4 features, got {type(outs_text)} len={len(outs_text) if isinstance(outs_text, (list, tuple)) else 'n/a'}")
        else:
            print(f"[PASS] text_cond=random: 4 features, shapes={shapes_text}")
            if outs_none is not None:
                same_shapes = _feature_shapes(outs_none) == shapes_text
                print(f"shape consistency with text_cond=None: {same_shapes}")
                if not same_shapes:
                    ok_text = False
    except Exception as e:
        ok_text = False
        print(f"[FAIL] text_cond=random forward error: {repr(e)}")

    return ok_none, ok_text


def check_alpha_init(model):
    ok = True
    print("\n=== Check: alpha initialization ===")
    vision_tower_mask = model.get_model().get_vision_tower_mask()
    module = getattr(vision_tower_mask, "mid_stage_text_recalibration", None)
    if module is None:
        print("[FAIL] mid_stage_text_recalibration not found")
        return False
    print("[PASS] mid_stage_text_recalibration found")
    alpha = getattr(module, "alpha", None)
    if alpha is None:
        print("[FAIL] alpha not found in mid_stage_text_recalibration")
        return False
    alpha_value = float(alpha.detach().float().cpu().view(-1)[0].item())
    is_zero_like = abs(alpha_value) < 1e-8
    print(f"alpha value: {alpha_value}")
    if is_zero_like:
        print("[PASS] alpha initialized to zero (or numerically zero)")
    else:
        ok = False
        print("[FAIL] alpha is not zero-initialized")
    return ok


def _count_trainable(named_params):
    trainable = [n for n, p in named_params if p.requires_grad]
    return len(named_params), len(trainable), trainable


def check_trainable_params(model, show_names):
    ok = True
    print("\n=== Check: trainable params visibility ===")
    named_params = list(model.named_parameters())
    vision_tower_mask = model.get_model().get_vision_tower_mask()
    stage3 = vision_tower_mask.layers[2]
    last_idx = len(stage3.blocks) - 1

    mid_params = [(n, p) for n, p in named_params if "vision_tower_mask.mid_stage_text_recalibration." in n]
    last_block_params = [(n, p) for n, p in named_params if f"vision_tower_mask.layers.2.blocks.{last_idx}." in n]
    norm_params = [
        (n, p) for n, p in named_params
        if ("vision_tower_mask.layers.2." in n and ".norm" in n) or "vision_tower_mask.norm2." in n
    ]

    for name, params in [
        ("mid_stage_text_recalibration", mid_params),
        ("stage3_last_block", last_block_params),
        ("stage3_norm", norm_params),
    ]:
        total, trainable, trainable_names = _count_trainable(params)
        print(f"{name}: total={total}, trainable={trainable}")
        if trainable == 0:
            ok = False
            print(f"[WARNING] {name} has 0 trainable params")
        if trainable_names:
            print(f"{name} sample trainable names: {trainable_names[:show_names]}")

    return ok


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print("[INFO] CPU detected: forcing dtype to float32 for compatibility")
        dtype = torch.float32

    print(f"device={device}, dtype={dtype}, init_only={args.init_only}, skip_vision_weights={args.skip_vision_weights}")

    results = {}
    try:
        _, model = build_model(args)
        model = to_device(model, device, dtype)
        if args.apply_trainable_recipe:
            apply_midstage_trainable_recipe(model, norm_only=args.norm_only)
        results["model_load"] = True
        print("[PASS] model load/init")
    except Exception as e:
        results["model_load"] = False
        print(f"[FAIL] model load/init: {repr(e)}")
        print("\n=== Summary ===")
        print("[FAIL] build_text_condition compatibility")
        print("[FAIL] vision_tower_mask text_cond=None")
        print("[FAIL] vision_tower_mask text_cond=random")
        print("[FAIL] alpha initialized to zero")
        print("[FAIL] trainable params visible")
        return

    try:
        results["build_text_condition"] = check_build_text_condition(model, device)
    except Exception as e:
        results["build_text_condition"] = False
        print(f"[FAIL] build_text_condition compatibility: {repr(e)}")

    try:
        none_ok, text_ok = check_vision_tower_forward(model, device, dtype, args.image_size, args.batch_size)
        results["vision_none"] = none_ok
        results["vision_text"] = text_ok
    except Exception as e:
        results["vision_none"] = False
        results["vision_text"] = False
        print(f"[FAIL] vision_tower_mask checks: {repr(e)}")

    try:
        results["alpha"] = check_alpha_init(model)
    except Exception as e:
        results["alpha"] = False
        print(f"[FAIL] alpha check: {repr(e)}")

    try:
        results["trainable"] = check_trainable_params(model, args.show_names)
    except Exception as e:
        results["trainable"] = False
        print(f"[FAIL] trainable params check: {repr(e)}")

    print("\n=== Optional check ===")
    print("[SKIP] minimal full loss forward (requires full multimodal batch assembly).")

    print("\n=== Summary ===")
    print(f"[{'PASS' if results.get('build_text_condition', False) else 'FAIL'}] build_text_condition compatibility")
    print(f"[{'PASS' if results.get('vision_none', False) else 'FAIL'}] vision_tower_mask text_cond=None")
    print(f"[{'PASS' if results.get('vision_text', False) else 'FAIL'}] vision_tower_mask text_cond=random")
    print(f"[{'PASS' if results.get('alpha', False) else 'FAIL'}] alpha initialized to zero")
    print(f"[{'PASS' if results.get('trainable', False) else 'FAIL'}] trainable params visible")


if __name__ == "__main__":
    main()
