import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class PriorOutput:
    patch_text_sim: torch.Tensor
    patch_prior: torch.Tensor
    patch_prob: torch.Tensor
    peak_prob: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    global_conf: torch.Tensor
    text_embed: torch.Tensor
    patch_embed: torch.Tensor
    patch_grid_size: Tuple[int, int]
    input_image_size: Tuple[int, int]
    expression_truncated: bool


class RemoteCLIPPriorBranch(torch.nn.Module):
    def __init__(self, model_name: str = "ViT-B-32", weight_path: Optional[str] = None, device: str = "cuda", unfreeze_last_layer: bool = False, temperature: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.model_name = model_name
        self.weight_path = weight_path
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.context_length = 77

        try:
            import open_clip  # type: ignore
        except ImportError as e:
            raise ImportError("open_clip is required for RemoteCLIPPriorBranch. Install open_clip_torch.") from e

        self.open_clip = open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=None)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        if self.weight_path is None:
            raise ValueError("weight_path is required for RemoteCLIPPriorBranch")
        self._load_remoteclip_weights(self.weight_path)

        self.model.to(self.device)
        # 与主模型 dtype 解耦：训练侧可能把整网 cast 成 bf16，OpenCLIP 在 fp32 下更稳
        self.model.float()
        self.model.eval()
        self._freeze_all()
        if unfreeze_last_layer:
            self._unfreeze_last_visual_block()

    def _preprocess_image(self, image: Image.Image, clip_input_size: int) -> torch.Tensor:
        tfm = transforms.Compose([
            transforms.Resize(clip_input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(clip_input_size),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
        ])
        return tfm(image).unsqueeze(0)

    def _freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def _unfreeze_last_visual_block(self) -> None:
        visual = getattr(self.model, "visual", None)
        if visual is None:
            raise RuntimeError("open_clip model has no visual module")
        blocks = None
        if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            blocks = visual.transformer.resblocks
        elif hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
            blocks = visual.trunk.blocks
        if blocks is None or len(blocks) == 0:
            raise RuntimeError("Cannot locate visual transformer blocks for unfreezing last layer")
        for p in blocks[-1].parameters():
            p.requires_grad = True

    def _remap_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        remapped = {}
        for k, v in state_dict.items():
            nk = "visual.transformer." + k[len("visual.trunk."):] if k.startswith("visual.trunk.") else k
            remapped[nk] = v
        return remapped

    def _assert_critical_remoteclip_loaded(self, final_missing: List[str]) -> None:
        """Fail if OpenCLIP skeleton would still be missing RemoteCLIP-critical weights."""
        miss = set(final_missing)
        model_sd = self.model.state_dict()
        critical_keys = [
            "visual.conv1.weight",
            "visual.transformer.resblocks.0.attn.in_proj_weight",
            "token_embedding.weight",
            "text_projection",
            "visual.proj",
        ]
        for k in critical_keys:
            if k in miss:
                raise RuntimeError(
                    f"Critical RemoteCLIP weight still listed in missing_keys after load: {k}. "
                    f"(missing_keys_count={len(final_missing)})"
                )
            if k not in model_sd:
                raise RuntimeError(f"Critical key absent from model.state_dict(): {k}")
            t = model_sd[k]
            if not torch.isfinite(t).all():
                raise RuntimeError(f"Critical tensor has non-finite values: {k}")
            if float(t.detach().abs().sum()) <= 1e-12:
                raise RuntimeError(f"Critical tensor is all-zero (likely not loaded): {k}")

    def _load_remoteclip_weights(self, weight_path: str) -> None:
        p = Path(weight_path)
        if not p.exists():
            raise FileNotFoundError(f"RemoteCLIP weight not found: {weight_path}")
        raw = torch.load(str(p), map_location="cpu")
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict) else raw
        if not isinstance(sd, dict):
            raise RuntimeError("Unexpected RemoteCLIP checkpoint format: expected dict/state_dict")

        def _report(tag: str, r) -> Tuple[List[str], List[str]]:
            mk, uk = list(r.missing_keys), list(r.unexpected_keys)
            print(f"[RemoteCLIP] {tag} missing_keys_count={len(mk)} unexpected_keys_count={len(uk)}")
            print(f"[RemoteCLIP] {tag} missing_keys_first10={mk[:10]}")
            print(f"[RemoteCLIP] {tag} unexpected_keys_first10={uk[:10]}")
            return mk, uk

        r1 = self.model.load_state_dict(sd, strict=False)
        mk1, uk1 = _report("initial_strict_false", r1)
        final_missing, final_unexpected = mk1, uk1

        if final_missing or final_unexpected:
            print(
                "[RemoteCLIP] Key mapping rule: checkpoint keys prefixed with 'visual.trunk.' "
                "are remapped to 'visual.transformer.' for OpenCLIP ViT layout compatibility."
            )
            sd2 = self._remap_state_dict_keys(sd)
            r2 = self.model.load_state_dict(sd2, strict=False)
            final_missing, final_unexpected = _report("after_key_remap_strict_false", r2)

        self._assert_critical_remoteclip_loaded(final_missing)
        print("[RemoteCLIP] remoteclip_weight_loaded=True")

    def _build_pos_embed(self, visual, x):
        pe = visual.positional_embedding.to(x.dtype)
        if pe.shape[0] == x.shape[1]:
            return pe
        cls_pe = pe[:1]
        patch_pe = pe[1:]
        old_n = patch_pe.shape[0]
        old_g = int(round(math.sqrt(old_n)))
        if old_g * old_g != old_n:
            raise RuntimeError(f"Cannot reshape original positional embedding: {old_n}")
        new_n = x.shape[1] - 1
        new_g = int(round(math.sqrt(new_n)))
        if new_g * new_g != new_n:
            raise RuntimeError(f"Cannot infer new grid from token count: {new_n}")
        patch_pe = patch_pe.reshape(old_g, old_g, -1).permute(2, 0, 1).unsqueeze(0)
        patch_pe = F.interpolate(patch_pe, size=(new_g, new_g), mode="bicubic", align_corners=False)
        patch_pe = patch_pe.squeeze(0).permute(1, 2, 0).reshape(new_g * new_g, -1)
        return torch.cat([cls_pe, patch_pe], dim=0)

    def _extract_patch_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual
        if not hasattr(visual, "conv1"):
            raise RuntimeError("Unsupported visual encoder structure: cannot extract patch tokens")

        x = visual.conv1(image_tensor)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        if hasattr(visual, "class_embedding"):
            cls = visual.class_embedding.to(x.dtype)
            cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
        if hasattr(visual, "positional_embedding"):
            x = x + self._build_pos_embed(visual, x)
        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)
        if hasattr(visual, "ln_pre"):
            x = visual.ln_pre(x)

        if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        elif hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
            x = visual.trunk(x)
        else:
            raise RuntimeError("Cannot locate visual transformer for patch token extraction")

        if hasattr(visual, "ln_post"):
            x = visual.ln_post(x)
        if hasattr(visual, "proj") and visual.proj is not None:
            x = x @ visual.proj
        if x.shape[1] <= 1:
            raise RuntimeError("Patch tokens not found: sequence length <= 1")
        return x[:, 1:, :]

    def _compute_text_embed(self, expression: str) -> Tuple[torch.Tensor, bool]:
        tokenized = self.tokenizer([expression])
        if not torch.is_tensor(tokenized):
            tokenized = torch.tensor(tokenized)
        tokenized = tokenized.to(self.device)

        truncated = False
        encode_fn = getattr(self.tokenizer, "encode", None)
        if callable(encode_fn):
            try:
                raw_ids = encode_fn(expression)
                if isinstance(raw_ids, (list, tuple)):
                    truncated = len(raw_ids) + 2 > self.context_length
            except Exception:
                truncated = False

        if truncated:
            print("[WARNING] Expression truncated")

        return self.model.encode_text(tokenized), truncated

    @torch.no_grad()
    def forward_image_expression(self, image_path: str, expression: str, clip_input_size: int = 224) -> PriorOutput:
        # DeepSpeed / Trainer 可能把整网参数 cast 成 bf16；此处强制 fp32 前向，结束后再恢复，避免 conv 输入/权重 dtype 不一致。
        try:
            p0 = next(self.model.parameters())
        except StopIteration:
            p0 = None
        orig_dtype = p0.dtype if p0 is not None else torch.float32

        self.model.float()
        amp_off = (
            torch.autocast(device_type="cuda", enabled=False)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        try:
            image_tensor = self._preprocess_image(Image.open(image_path).convert("RGB"), clip_input_size).to(
                self.device, dtype=torch.float32
            )
            with amp_off:
                patch_embed = self._extract_patch_tokens(image_tensor)
                text_embed, truncated = self._compute_text_embed(expression)

                patch_sim = torch.matmul(
                    F.normalize(patch_embed, dim=-1), F.normalize(text_embed, dim=-1).unsqueeze(-1)
                ).squeeze(-1)
                n = patch_sim.shape[-1]
                g = int(round(math.sqrt(n)))
                if g * g != n:
                    raise RuntimeError(f"Patch count {n} is not a square; cannot infer 2D grid")

                patch_prob = torch.softmax(patch_sim / self.temperature, dim=-1)
                peak = patch_prob.max(dim=-1, keepdim=True).values
                entropy = -(patch_prob * torch.log(patch_prob + self.eps)).sum(dim=-1, keepdim=True)
                normalized_entropy = entropy / math.log(max(n, 2))
                global_conf = peak * (1.0 - normalized_entropy)

                return PriorOutput(
                    patch_text_sim=patch_sim,
                    patch_prior=patch_sim,
                    patch_prob=patch_prob,
                    peak_prob=peak,
                    entropy=entropy,
                    normalized_entropy=normalized_entropy,
                    global_conf=global_conf,
                    text_embed=text_embed,
                    patch_embed=patch_embed,
                    patch_grid_size=(g, g),
                    input_image_size=(clip_input_size, clip_input_size),
                    expression_truncated=truncated,
                )
        finally:
            if orig_dtype != torch.float32:
                self.model.to(dtype=orig_dtype)


def topk_patch_info(patch_text_sim: torch.Tensor, patch_prob: torch.Tensor, patch_grid_size: Tuple[int, int], original_hw: Tuple[int, int], prior_hw: Tuple[int, int], topk: int = 5) -> List[Dict]:
    sim = patch_text_sim.detach().cpu().flatten()
    prob = patch_prob.detach().cpu().flatten()
    h, w = patch_grid_size
    oh, ow = original_hw
    ph, pw = prior_hw
    k = min(int(topk), sim.numel())
    vals, idxs = torch.topk(sim, k=k)
    out: List[Dict] = []
    for rank, (v, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        r = int(idx // w)
        c = int(idx % w)
        x1n, x2n = c / w, (c + 1) / w
        y1n, y2n = r / h, (r + 1) / h

        bx1p, by1p, bx2p, by2p = int(round(x1n * pw)), int(round(y1n * ph)), int(round(x2n * pw)), int(round(y2n * ph))
        bx1o, by1o, bx2o, by2o = int(round(x1n * ow)), int(round(y1n * oh)), int(round(x2n * ow)), int(round(y2n * oh))
        cxp, cyp = (bx1p + bx2p) / 2.0, (by1p + by2p) / 2.0
        cxo, cyo = (bx1o + bx2o) / 2.0, (by1o + by2o) / 2.0

        out.append({
            "rank": rank,
            "row": r,
            "col": c,
            "flat_index": int(idx),
            "similarity": float(v),
            "probability": float(prob[idx].item()),
            "bbox_in_prior_space": [bx1p, by1p, bx2p, by2p],
            "bbox_in_original_space": [bx1o, by1o, bx2o, by2o],
            "center_in_prior_space": [cxp, cyp],
            "center_in_original_space": [cxo, cyo],
        })
    return out
