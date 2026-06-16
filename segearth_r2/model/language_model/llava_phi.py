from typing import Dict, List, Optional, Tuple, Union
from addict import Dict
from dataclasses import dataclass
import os
from pathlib import Path
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom

from ..mipha.model.language_model.mipha_phi import (MiphaPhiForCausalLM, MiphaPhiModel)

from segearth_r2.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, REFER_TOKEN_INDEX

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoderForOPTPreTrain
from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.qdti import QuerySpecificTextMemoryBias
from ..mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from ..mask_encoder.swin_trans import build_swin_b, build_swin_l

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

from ..datasets_mapper.IVS_mapper import IVSDatasetMapper
from segearth_r2.model.mask_decoder.mask_criterion.Mask_Criterion import Criterion, hungarian_matcher_InstructSeg
from segearth_r2.model.language_model.prompt_query_fusion import (
    DualGranularityPromptAdapter,
    PromptAwareQueryRefiner,
    pack_seg_hidden_states_bq,
    expand_bq_for_mask_num,
    expand_bp_for_mask_num,
)
from transformers import PhiModel, PhiForCausalLM, PhiConfig
from fvcore.nn import FlopCountAnalysis

@dataclass
class CausalOutputWithMask(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_mask: Optional[torch.FloatTensor] = None
    loss_dice: Optional[torch.FloatTensor] = None
    loss_llm: Optional[torch.FloatTensor] = None
    loss_attention: Optional[torch.FloatTensor] = None

class AttentionLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(AttentionLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, model_attention_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        device = model_attention_logits.device
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)  # Make sure the tensor is on the correct device
        epsilon = 1e-8  # To avoid log(0)
        for idx in range(model_attention_logits.shape[0]):
            # Extract the attention map values based on the mask
            attention_map_target = model_attention_logits[idx][gt_mask[idx] == 1]
            attention_map_else = model_attention_logits[idx][gt_mask[idx] == 0]
            if attention_map_target.numel() == 0:
                continue
            mean = torch.mean(attention_map_else) if attention_map_else.numel() > 0 else torch.tensor(0.0, device=device)
            mse = torch.mean((attention_map_target - mean) ** 2)
            loss += -torch.log(mse + epsilon)
        if self.reduction == 'batchmean':
            loss = loss / model_attention_logits.shape[0]
        elif self.reduction == 'sum':
            pass  # Use the raw sum of losses
        elif self.reduction == 'mean':
            loss = loss / model_attention_logits.numel()  # Overall mean loss
        return loss

class SegEarthR2Model(MiphaPhiModel):

    def __init__(self, config: PhiConfig, mask_decoder_cfg=None):
        super(SegEarthR2Model, self).__init__(config)
        self.cfg = mask_decoder_cfg
        self.projector_outdim = config.hidden_size

        if hasattr(config, "mm_vision_tower"):
            swin_type = getattr(config,'swin_type','base')
            if swin_type == 'base':
                self.vision_tower_mask = build_swin_b(None)
            else:
                self.vision_tower_mask = build_swin_l(None)

            self.vision_tower_mask.image_processor = IVSDatasetMapper(self.cfg)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_vision_tower_mask(self):
        vision_tower = getattr(self, 'vision_tower_mask', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower if hasattr(model_args, 'vision_tower') else model_args.mm_vision_tower
        vision_tower_mask = model_args.vision_tower_mask if hasattr(model_args, 'vision_tower_mask') else model_args.mm_vision_tower_mask

        self.config.mm_vision_tower = vision_tower
        swin_type = getattr(model_args,'swin_type','base')
        self.config.swin_type = swin_type
        if swin_type == 'base':
            vision_tower_mask = build_swin_b(vision_tower_mask)
        else:
            print('current visual encoder is swin large')
            vision_tower_mask = build_swin_l(vision_tower_mask)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower_mask = [vision_tower_mask]
        else:
            self.vision_tower_mask = vision_tower_mask

        self.config.use_mm_proj = True
        vision_tower_mask.hidden_size = 256
        vision_tower_mask.image_processor = IVSDatasetMapper(self.cfg)

class SegEarthR2(MiphaPhiForCausalLM):
    DGP_QDTI_STATE_KEY_MARKERS = (
        "prompt_adapter",
        "query_refiner",
        "query_specific_text_memory_bias",
    )
    DGP_PROMPT_KEY_MARKERS = (
        "prompt_adapter",
        "query_refiner",
    )
    DGP_STAGE_A_WEIGHTS_BIN = "dgp_stage_a_weights.bin"
    DGP_STAGE_A_WEIGHTS_SAFE = "dgp_stage_a_weights.safetensors"
    DGP_CONFIG_FIELDS = (
        "dgp_version",
        "dgp_training_stage",
        "use_dgp_qdti",
        "use_qdti_bias",
        "gate_init",
        "gate_g_init",
        "gate_l_init",
        "use_sigmoid_gate",
        "dgp_fuse_dim",
        "dgp_refiner_hidden_dim",
        "dgp_pg_tokens",
        "qdti_bias_dim",
        "qdti_init_std",
        "qdti_max_abs",
        "qdti_apply_layers",
        "qdti_scale_init",
        "scale_hard_loss_weight",
    )

    STAGE_A_TRAINABLE_MARKERS = ("prompt_adapter", "query_refiner")
    STAGE_A_FORBIDDEN_TRAINABLE_MARKERS = (
        "SEG_token_projector",
        "pixel_decoder",
        "predictor",
        "vision_tower",
        "vision_tower_mask",
        "lm_head",
        "lora_",
        "query_specific_text_memory_bias",
        "qdti",
    )

    @classmethod
    def _normalize_dgp_param_key(cls, name: str) -> str:
        key = name
        for prefix in ("base_model.model.", "model.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        return key

    @classmethod
    def apply_v61_stage_defaults(cls, model_args, stage: Optional[str]) -> None:
        stage = (stage or "").strip().lower()
        if stage not in ("a", "b"):
            return
        if not getattr(model_args, "dgp_version", None):
            model_args.dgp_version = "v6.1"
        model_args.use_dgp_qdti = True
        if getattr(model_args, "gate_g_init", None) is None:
            model_args.gate_g_init = 0.01
        if getattr(model_args, "gate_l_init", None) is None:
            model_args.gate_l_init = 0.02
        if getattr(model_args, "gate_init", None) is None:
            model_args.gate_init = float(model_args.gate_l_init)
        if stage == "a":
            model_args.dgp_training_stage = "a"
            model_args.use_qdti_bias = False
            model_args.qdti_apply_layers = "disabled"
            if getattr(model_args, "qdti_scale_init", None) is None:
                model_args.qdti_scale_init = 0.0
        elif stage == "b":
            model_args.dgp_training_stage = "b"
            model_args.use_qdti_bias = True
            model_args.qdti_apply_layers = "last1"
            model_args.qdti_scale_init = 1e-3

    @classmethod
    def merge_dgp_config_from_hf(cls, hf_config, model_args) -> None:
        """Fill unset CLI args from saved HF config (eval/merge load path)."""
        inherit_if_none = (
            "dgp_version",
            "dgp_training_stage",
            "gate_init",
            "gate_g_init",
            "gate_l_init",
            "use_sigmoid_gate",
            "qdti_apply_layers",
            "qdti_scale_init",
            "dgp_fuse_dim",
            "dgp_refiner_hidden_dim",
            "dgp_pg_tokens",
            "qdti_bias_dim",
            "qdti_init_std",
            "qdti_max_abs",
            "scale_hard_loss_weight",
        )
        for field in inherit_if_none:
            cli_val = getattr(model_args, field, None)
            hf_val = getattr(hf_config, field, None)
            if cli_val is None and hf_val is not None:
                setattr(model_args, field, hf_val)

        if not bool(getattr(model_args, "use_dgp_qdti", False)):
            if bool(getattr(hf_config, "use_dgp_qdti", False)):
                model_args.use_dgp_qdti = True

        if getattr(model_args, "use_qdti_bias", None) is None:
            if hasattr(hf_config, "use_qdti_bias"):
                model_args.use_qdti_bias = bool(hf_config.use_qdti_bias)

    @classmethod
    def extract_stage_a_state_dict(cls, model) -> Dict[str, torch.Tensor]:
        state = {}
        for name, param in model.named_parameters():
            if any(marker in name for marker in cls.DGP_PROMPT_KEY_MARKERS):
                key = cls._normalize_dgp_param_key(name)
                state[key] = param.detach().cpu().clone()
        return state

    @classmethod
    def export_stage_a_weights(cls, model, directory: str, rank0_log: bool = True) -> str:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        state = cls.extract_stage_a_state_dict(model)
        if not state:
            raise RuntimeError(f"No Stage A DGP keys found to export under {directory}")

        bin_path = out_dir / cls.DGP_STAGE_A_WEIGHTS_BIN
        torch.save(state, bin_path)
        try:
            from safetensors.torch import save_file
            save_file(state, str(out_dir / cls.DGP_STAGE_A_WEIGHTS_SAFE))
        except Exception:
            pass

        if rank0_log:
            keys = sorted(state.keys())
            print(f"[DGP] Exported Stage A weights: {len(keys)} tensors -> {bin_path}")
            print(f"[DGP] Stage A keys: {keys}")
        return str(bin_path)

    @classmethod
    def _load_stage_a_file(cls, path: Path) -> Dict[str, torch.Tensor]:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return dict(load_file(str(path)))
        return torch.load(str(path), map_location="cpu")

    @classmethod
    def _filter_stage_a_state(cls, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            cls._normalize_dgp_param_key(k): v
            for k, v in state.items()
            if any(marker in k for marker in cls.DGP_PROMPT_KEY_MARKERS)
        }

    @classmethod
    def _remap_stage_a_state_for_model(cls, model, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_sd = model.state_dict()
        norm_to_full = {cls._normalize_dgp_param_key(k): k for k in model_sd.keys()}
        remapped = {}
        for k, v in state.items():
            nk = cls._normalize_dgp_param_key(k)
            if nk in norm_to_full:
                remapped[norm_to_full[nk]] = v
            elif k in model_sd:
                remapped[k] = v
        return remapped

    @classmethod
    def load_dgp_stage_checkpoint(
        cls,
        model,
        checkpoint_root: str,
        markers=None,
        rank0_log: bool = True,
    ) -> Dict[str, object]:
        markers = markers or cls.DGP_PROMPT_KEY_MARKERS
        checked_paths: List[str] = []
        root = Path(checkpoint_root)
        if not root.exists():
            raise FileNotFoundError(f"DGP checkpoint root not found: {checkpoint_root}")

        ckpt_dir = root
        if root.is_dir():
            ckpts = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if ckpts:
                ckpt_dir = ckpts[-1]

        search_dirs = []
        for d in (ckpt_dir, root):
            if d not in search_dirs:
                search_dirs.append(d)

        for base in search_dirs:
            for fname in (cls.DGP_STAGE_A_WEIGHTS_BIN, cls.DGP_STAGE_A_WEIGHTS_SAFE):
                path = base / fname
                checked_paths.append(str(path))
                if path.is_file():
                    state = cls._filter_stage_a_state(cls._load_stage_a_file(path))
                    return cls._apply_stage_a_state(model, state, str(path), markers, rank0_log)

        for base in search_dirs:
            bin_path = base / "pytorch_model.bin"
            checked_paths.append(str(bin_path))
            if bin_path.is_file():
                state = cls._filter_stage_a_state(torch.load(bin_path, map_location="cpu"))
                return cls._apply_stage_a_state(model, state, str(bin_path), markers, rank0_log)

        for base in search_dirs:
            for st_path in sorted(base.glob("*.safetensors")):
                if "model" not in st_path.name:
                    continue
                checked_paths.append(str(st_path))
                try:
                    from safetensors.torch import load_file
                    state = cls._filter_stage_a_state(load_file(str(st_path)))
                    if state:
                        return cls._apply_stage_a_state(model, state, str(st_path), markers, rank0_log)
                except Exception:
                    continue

        zero_dirs = [d for d in search_dirs if (d / "global_step0").exists() or list(d.glob("global_step*"))]
        for zdir in zero_dirs:
            checked_paths.append(str(zdir))
            try:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                raw = get_fp32_state_dict_from_zero_checkpoint(str(zdir))
                state = cls._filter_stage_a_state(raw)
                if state:
                    return cls._apply_stage_a_state(model, state, str(zdir), markers, rank0_log)
            except Exception:
                continue

        raise FileNotFoundError(
            "Failed to load Stage A DGP weights. Checked paths:\n  - "
            + "\n  - ".join(checked_paths)
        )

    @classmethod
    def _apply_stage_a_state(
        cls,
        model,
        state: Dict[str, torch.Tensor],
        source: str,
        markers,
        rank0_log: bool,
    ) -> Dict[str, object]:
        if not state:
            raise RuntimeError(f"No DGP keys {markers} found in {source}")

        remapped = cls._remap_stage_a_state_for_model(model, state)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        missing_markers = [
            k for k in missing if any(m in k for m in markers)
        ]
        if missing_markers:
            raise RuntimeError(
                f"Failed to load DGP stage checkpoint from {source}; missing keys: {missing_markers[:12]}"
            )

        gate_g_after = None
        gate_l_after = None
        for name, param in model.named_parameters():
            if name.endswith("query_refiner.gate_g") or name.endswith("query_refiner.gate_g_logit"):
                gate_g_after = float(param.detach().float().cpu().reshape(-1)[0].item())
            if name.endswith("query_refiner.gate_l") or name.endswith("query_refiner.gate_l_logit"):
                gate_l_after = float(param.detach().float().cpu().reshape(-1)[0].item())
        if gate_l_after is None and hasattr(model, "query_refiner"):
            refiner = model.query_refiner
            if getattr(refiner, "use_sigmoid_gate", False):
                gate_l_after = float(torch.sigmoid(refiner.gate_l_logit).detach().cpu().reshape(-1)[0].item())
            elif hasattr(refiner, "gate_l"):
                gate_l_after = float(refiner.gate_l.detach().float().cpu().reshape(-1)[0].item())
            elif hasattr(refiner, "gate"):
                gate_l_after = float(refiner.gate.detach().float().cpu().reshape(-1)[0].item())

        gate_g_src = None
        gate_l_src = None
        for k, v in state.items():
            nk = cls._normalize_dgp_param_key(k)
            if nk.endswith("query_refiner.gate_g") or nk.endswith("query_refiner.gate_g_logit"):
                gate_g_src = float(v.detach().float().cpu().reshape(-1)[0].item())
            if nk.endswith("query_refiner.gate_l") or nk.endswith("query_refiner.gate_l_logit") or nk == "query_refiner.gate":
                gate_l_src = float(v.detach().float().cpu().reshape(-1)[0].item())

        report = {
            "source": source,
            "loaded_keys": sorted(state.keys()),
            "loaded_count": len(state),
            "missing": list(missing),
            "unexpected": list(unexpected),
            "gate_g_src": gate_g_src,
            "gate_l_src": gate_l_src,
            "gate_g_after": gate_g_after,
            "gate_l_after": gate_l_after,
        }
        if rank0_log:
            print(
                f"[DGP] Loaded {report['loaded_count']} Stage A tensors from {source}; "
                f"gate_g_src={gate_g_src} gate_l_src={gate_l_src} "
                f"gate_g_after={gate_g_after} gate_l_after={gate_l_after}"
            )
            print(f"[DGP] Loaded keys: {report['loaded_keys']}")
            if unexpected:
                print(f"[DGP] Unexpected keys (ignored): {report['unexpected'][:8]}")
        return report

    @staticmethod
    def resolve_use_qdti_bias(model_args) -> bool:
        if not bool(getattr(model_args, "use_dgp_qdti", False)):
            return False
        if getattr(model_args, "use_qdti_bias", None) is None:
            return True
        return bool(getattr(model_args, "use_qdti_bias"))

    @classmethod
    def sync_dgp_config_from_args(cls, config, model_args):
        use_dgp = bool(getattr(model_args, "use_dgp_qdti", False))
        config.use_dgp_qdti = use_dgp
        config.use_qdti_bias = cls.resolve_use_qdti_bias(model_args)
        config.dgp_fuse_dim = int(getattr(model_args, "dgp_fuse_dim", 256))
        config.dgp_refiner_hidden_dim = int(getattr(model_args, "dgp_refiner_hidden_dim", 512))
        config.dgp_pg_tokens = int(getattr(model_args, "dgp_pg_tokens", 1))
        config.qdti_bias_dim = int(getattr(model_args, "qdti_bias_dim", 128))
        config.qdti_init_std = float(getattr(model_args, "qdti_init_std", 1e-3))
        config.qdti_max_abs = float(getattr(model_args, "qdti_max_abs", 0.01))
        qdti_layers = getattr(model_args, "qdti_apply_layers", None)
        stage = (getattr(model_args, "dgp_training_stage", None) or "").strip().lower()
        version = (getattr(model_args, "dgp_version", None) or "").strip().lower()
        if qdti_layers is None and version != "v6.1" and stage not in ("a", "b"):
            qdti_layers = "last3"
        if qdti_layers is not None:
            config.qdti_apply_layers = str(qdti_layers)
        qdti_scale = getattr(model_args, "qdti_scale_init", None)
        if qdti_scale is not None or version == "v6.1" or stage in ("a", "b"):
            config.qdti_scale_init = float(0.0 if qdti_scale is None else qdti_scale)
        config.scale_hard_loss_weight = float(getattr(model_args, "scale_hard_loss_weight", 0.0))
        if getattr(model_args, "dgp_version", None) is not None:
            config.dgp_version = str(model_args.dgp_version)
        if getattr(model_args, "dgp_training_stage", None) is not None:
            config.dgp_training_stage = str(model_args.dgp_training_stage)
        if getattr(model_args, "gate_init", None) is not None:
            config.gate_init = float(model_args.gate_init)
        if getattr(model_args, "gate_g_init", None) is not None:
            config.gate_g_init = float(model_args.gate_g_init)
        if getattr(model_args, "gate_l_init", None) is not None:
            config.gate_l_init = float(model_args.gate_l_init)
        if getattr(model_args, "use_sigmoid_gate", None) is not None:
            config.use_sigmoid_gate = bool(model_args.use_sigmoid_gate)
        return config

    def __init__(self, config, model_args=None, mask_decoder_cfg=None, add_cross_attn=True, cross_attn_index=None):
        super(SegEarthR2, self).__init__(config)

        self.model = SegEarthR2Model(config, mask_decoder_cfg)
        self.init_config = config
        self.mask_decoder_cfg = mask_decoder_cfg
        self.cross_attn_index = cross_attn_index

        self.lm_head = nn.Linear(config.hidden_size, 51200, bias=False)

        is_train_mask_decode = getattr(config, 'mask_decode_train', False)
        self.is_train_mask_decode = is_train_mask_decode

        if is_train_mask_decode:
            print('Mask Decoder has been trained, init directly')
            self.initial_mask_module()
        self.post_init()

    def _dgp_qdti_enabled(self) -> bool:
        return bool(getattr(self.config, "use_dgp_qdti", False))

    def _qdti_bias_enabled(self) -> bool:
        if not self._dgp_qdti_enabled():
            return False
        return bool(getattr(self.config, "use_qdti_bias", True))

    def _resolve_pad_token_id(self):
        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is None and hasattr(self, "tokenizer") and self.tokenizer is not None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
        return pad_id

    def ensure_dgp_qdti_modules(self):
        if not self._dgp_qdti_enabled():
            return
        llm_dim = int(self.config.hidden_size)
        fuse_dim = int(getattr(self.config, "dgp_fuse_dim", 256))
        refiner_hidden = int(getattr(self.config, "dgp_refiner_hidden_dim", 512))
        pg_tokens = int(getattr(self.config, "dgp_pg_tokens", 1))
        if getattr(self, "prompt_adapter", None) is None:
            self.prompt_adapter = DualGranularityPromptAdapter(llm_dim, fuse_dim, pg_tokens=pg_tokens)
        if getattr(self, "query_refiner", None) is None:
            gate_g_init = float(getattr(self.config, "gate_g_init", 0.01))
            gate_l_init = float(getattr(self.config, "gate_l_init", 0.02))
            use_sigmoid_gate = bool(getattr(self.config, "use_sigmoid_gate", False))
            self.query_refiner = PromptAwareQueryRefiner(
                fuse_dim,
                refiner_hidden,
                gate_g_init=gate_g_init,
                gate_l_init=gate_l_init,
                use_sigmoid_gate=use_sigmoid_gate,
            )
        if not hasattr(self, "predictor") or self.predictor is None:
            return
        spec = str(getattr(self.config, "qdti_apply_layers", "last3"))
        if spec.lower() in ("disabled", "none") or not self._qdti_bias_enabled():
            self.predictor.qdti_apply_layers = None
        elif getattr(self.predictor, "qdti_apply_layers", None) is None:
            self.predictor.qdti_apply_layers = spec
        if not self._qdti_bias_enabled():
            return
        if getattr(self.predictor, "query_specific_text_memory_bias", None) is not None:
            return
        hidden_dim = int(self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.predictor.register_module(
            "query_specific_text_memory_bias",
            QuerySpecificTextMemoryBias(
                text_dim=fuse_dim,
                memory_dim=hidden_dim,
                query_dim=hidden_dim,
                bias_dim=int(getattr(self.config, "qdti_bias_dim", 128)),
                init_std=float(getattr(self.config, "qdti_init_std", 1e-3)),
                max_abs=float(getattr(self.config, "qdti_max_abs", 0.01)),
                qdti_scale_init=float(getattr(self.config, "qdti_scale_init", 0.0)),
            ),
        )

    @classmethod
    def dgp_qdti_keys_in_state_dict(cls, state_dict) -> bool:
        for key in state_dict.keys():
            if any(marker in key for marker in cls.DGP_QDTI_STATE_KEY_MARKERS):
                return True
        return False

    @classmethod
    def validate_dgp_qdti_checkpoint(cls, state_dict, context: str = "checkpoint", require_qdti_bias: bool = True):
        required = list(cls.DGP_PROMPT_KEY_MARKERS)
        if require_qdti_bias:
            required.append("query_specific_text_memory_bias")
        missing = []
        for marker in required:
            if not any(marker in k for k in state_dict.keys()):
                missing.append(marker)
        if missing:
            raise RuntimeError(
                f"use_dgp_qdti=True but {context} is missing DGP-QDTI weights for: {missing}. "
                f"Expected state_dict keys containing {required}."
            )

    def _repeat_text_memory_for_mask_num(self, text_memory, text_memory_mask, mask_num_tensor):
        if text_memory is None or text_memory_mask is None:
            return None, None
        tm = expand_bp_for_mask_num(text_memory, mask_num_tensor)
        mn = torch.as_tensor(mask_num_tensor, device=text_memory_mask.device, dtype=torch.long).flatten()
        tmm = torch.repeat_interleave(text_memory_mask, mn, dim=0)
        return tm, tmm

    def _store_dgp_health(self, adapter_health: dict, refiner_health: dict) -> None:
        payload = {}
        for key, value in adapter_health.items():
            if key in ("detail_prompt_source_name", "_q_detail"):
                if key == "detail_prompt_source_name":
                    payload[key] = value
                continue
            if torch.is_tensor(value):
                payload[key] = float(value.detach().float().cpu().item())
            else:
                payload[key] = value
        for key, value in refiner_health.items():
            if torch.is_tensor(value):
                payload[key] = float(value.detach().float().cpu().item())
            else:
                payload[key] = value
        self._dgp_last_health = payload

    def _build_instruction_text_mask(
        self,
        attention_mask: torch.Tensor,
        seg_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        *,
        is_eval: bool = False,
    ) -> Tuple[torch.Tensor, str]:
        """
        Full instruction-only KV mask for P_g / P_l text_tokens.
        refer_span_mask is never used here (diagnostic / detail_source only).
        """
        base = attention_mask.bool() & ~seg_mask.bool()
        if image_mask is not None:
            base = base & ~image_mask.bool()

        if labels is not None and labels.shape == attention_mask.shape:
            mask = base & (labels == IGNORE_INDEX)
            return mask, "labels_ignore_index"

        if is_eval:
            print(
                "[DGP] Eval has no labels; using conservative mask excluding pad/image/[SEG]/special tokens."
            )
        return base, "conservative_no_labels"

    def _maybe_audit_instruction_mask(
        self,
        text_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        seg_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor],
        refer_span_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        mode: str,
    ) -> None:
        if getattr(self, "_dgp_instruction_mask_audited", False):
            return
        self._dgp_instruction_mask_audited = True

        b = 0
        seq_len = text_mask.shape[1]
        included = text_mask[b].nonzero(as_tuple=False).flatten().tolist()
        attn_row = attention_mask[b].bool() if attention_mask is not None else text_mask[b]
        excluded = (~text_mask[b] & attn_row).nonzero(as_tuple=False).flatten().tolist()

        seg_excluded = not (text_mask[b] & seg_mask[b]).any().item()
        img_excluded = True
        if image_mask is not None:
            img_excluded = not (text_mask[b] & image_mask[b]).any().item()

        pad_excluded = True
        if labels is not None and labels.shape == text_mask.shape:
            pad_excluded = not (text_mask[b] & (labels == 0)).any().item()

        assistant_excluded = True
        if labels is not None and labels.shape == text_mask.shape:
            assistant_excluded = not (text_mask[b] & (labels != IGNORE_INDEX)).any().item()

        refer_span_present = False
        refer_span_in_text = False
        if refer_span_mask is not None:
            refer_span_present = bool(refer_span_mask[b].any().item())
            refer_span_in_text = bool((text_mask[b] & refer_span_mask[b].bool()).any().item())

        text_mask_equals_refer_only = False
        if refer_span_present:
            refer_only = attention_mask[b].bool() & refer_span_mask[b].bool()
            text_mask_equals_refer_only = bool((text_mask[b] == refer_only).all().item())

        print("[DGP][instruction_mask_audit] text_mask_mode=", mode)
        print(f"  seq_len={seq_len} included_count={len(included)} excluded_count={len(excluded)}")
        print(f"  included_token_indices_head={included[:16]}{'...' if len(included) > 16 else ''}")
        print(f"  excluded_token_indices_head={excluded[:16]}{'...' if len(excluded) > 16 else ''}")
        print(f"  [SEG]_excluded={seg_excluded} image_excluded={img_excluded} pad_excluded={pad_excluded}")
        print(f"  assistant_answer_excluded={assistant_excluded}")
        print(
            f"  refer_span_present={refer_span_present} refer_span_in_text={refer_span_in_text} "
            f"text_mask_is_refer_only={text_mask_equals_refer_only}"
        )
        if text_mask_equals_refer_only and refer_span_present:
            print("  [WARN] text_mask equals refer_span only — P_g would not be expression-level")

    def _build_instruction_mask(
        self,
        attention_mask: torch.Tensor,
        seg_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        refer_span_mask: Optional[torch.Tensor] = None,
        is_eval: bool = False,
    ) -> torch.Tensor:
        mask, mode = self._build_instruction_text_mask(
            attention_mask, seg_mask, image_mask, labels, is_eval=is_eval
        )
        self._maybe_audit_instruction_mask(
            mask, attention_mask, seg_mask, image_mask, refer_span_mask, labels, mode
        )
        return mask

    @classmethod
    def validate_stage_a_trainable_params(cls, model, rank0_log: bool = True) -> None:
        """Fail fast if any trainable parameter falls outside Stage A whitelist."""
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        if rank0_log:
            print(f"[DGP v6.1 Stage A] trainable parameter count={len(trainable_names)}")
            for name in sorted(trainable_names):
                print(f"  [TRAINABLE] {name}")

        invalid = []
        for name in trainable_names:
            if any(forbidden in name for forbidden in cls.STAGE_A_FORBIDDEN_TRAINABLE_MARKERS):
                invalid.append(name)
                continue
            if not any(marker in name for marker in cls.STAGE_A_TRAINABLE_MARKERS):
                invalid.append(name)

        if invalid:
            raise RuntimeError(
                "[DGP v6.1 Stage A] Invalid trainable parameters outside whitelist:\n  - "
                + "\n  - ".join(sorted(invalid))
            )

    def _compute_seg_embedding_with_dgp(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seg_embedding_indices: torch.Tensor,
        image_features_indices: Optional[torch.Tensor] = None,
        target_phrase_mask: Optional[torch.Tensor] = None,
        refer_span_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_hints: Optional[list] = None,
        is_eval: bool = False,
    ):
        seg_mask = seg_embedding_indices.bool()
        image_mask = image_features_indices.bool() if image_features_indices is not None else None
        instruction_mask = self._build_instruction_mask(
            attention_mask, seg_mask, image_mask, labels, refer_span_mask, is_eval=is_eval
        )

        seg_hidden_bq, seg_valid_mask = pack_seg_hidden_states_bq(hidden_states, seg_mask)
        seg_embedding = self.SEG_token_projector(seg_hidden_bq)

        p_g, p_l, prompt_tokens, prompt_mask, adapter_health = self.prompt_adapter(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            seg_mask=seg_mask,
            q_seg=seg_embedding,
            target_phrase_mask=target_phrase_mask,
            refer_span_mask=refer_span_mask,
            image_mask=image_mask,
            instruction_mask=instruction_mask,
            seg_query_mask=seg_valid_mask,
        )

        q_ref, refiner_health = self.query_refiner(
            seg_embedding,
            p_g,
            p_l,
            seg_query_mask=seg_valid_mask,
            prompt_mask=prompt_mask,
        )
        self._store_dgp_health(adapter_health, refiner_health)
        if class_hints is not None:
            self._dgp_last_health["class_hints"] = list(class_hints)
            cos_pl = float(adapter_health.get("cos_pl_qseg", torch.tensor(0.0)).detach().float().cpu().item()
                if torch.is_tensor(adapter_health.get("cos_pl_qseg")) else float(adapter_health.get("cos_pl_qseg", 0.0)))
            for hints in class_hints:
                for cls_name in hints:
                    self._dgp_last_health[f"class_{cls_name}_cos_pl_qseg"] = cos_pl

        B, Q, D = q_ref.shape
        if seg_valid_mask.shape != (B, Q):
            raise ValueError(
                f"seg_valid_mask {tuple(seg_valid_mask.shape)} != Q_ref batch/query {(B, Q)}"
            )
        if prompt_tokens.shape[:2] != prompt_mask.shape:
            raise ValueError("prompt_tokens / prompt_mask shape mismatch")

        return q_ref, prompt_tokens, prompt_mask

    def initial_mask_module(self, pretrained_path=None, model_args=None):
        if not self.is_train_mask_decode:
            print('Initialize mask modules...')
            self.config.mask_decode_train = True

        self.attention_loss = AttentionLoss()
        
        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        self.SEG_token_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

        if model_args is not None:
            self.sync_dgp_config_from_args(self.config, model_args)
        self.ensure_dgp_qdti_modules()
            
        self.mask_decoder_training_init(self.mask_decoder_cfg)
        if pretrained_path is not None:
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def change_w(weights, old_name, new_name):
                weights[new_name] = weights[old_name]
                weights.pop(old_name)

            if pretrained_path.endswith('.pkl'):
                with open(pretrained_path, 'rb') as f:
                    ckpt = pickle.load(f)
            else:
                ckpt = torch.load(pretrained_path)
            pixel_decoder_weights = get_w(ckpt['model'],'sem_seg_head.pixel_decoder')
            predictor_weights = get_w(ckpt['model'],'sem_seg_head.predictor')
            pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
            predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

            #deal some diff keys
            change_w(pixel_decoder_weights,'adapter_1.weight','adapter_1.0.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.weight','adapter_1.1.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.bias','adapter_1.1.bias')
            change_w(pixel_decoder_weights,'layer_1.weight','layer_1.0.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.weight','layer_1.1.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.bias','layer_1.1.bias')
            if 'static_query.weight' in predictor_weights:
                change_w(predictor_weights,'static_query.weight','query_feat.weight')
            if predictor_weights['query_embed.weight'].shape[0] == 200:
                predictor_weights['query_embed.weight'] = predictor_weights['query_embed.weight'][:100,:]
            diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights,strict=False)
            diff_predictor_msg = self.predictor.load_state_dict(predictor_weights,strict=False)
            print(diff_predictor_msg)
            print(diff_pixel_msg)

    def get_vision_tower_feature(self, images):
        features = self.get_model().get_vision_tower_mask()(images)
        
        features_dict = {
            'res2': features[0], # bs, 128, 256, 256
            'res3': features[1], # bs, 256, 128, 128
            'res4': features[2], # bs, 512, 64, 64
            'res5': features[3], # bs, 1024, 32, 32
        }
        return features_dict
    def mask_decoder_training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT
        
        matcher = hungarian_matcher_InstructSeg(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        
        weight_dict = {"loss_SEG_class": class_weight,  "loss_mask": mask_weight,
                       "loss_dice": dice_weight, }

        self.weight_dict = weight_dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["SEG_labels", "masks",]
        self.criterion = Criterion(
            matcher=matcher,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device
        )
        self.size_divisibility = 32
        self.sem_seg_postprocess_before_inference = True
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images) # image_features: [4, 729, 1152]
        image_features = self.get_model().mm_projector(image_features) # image_features: [4, 729, 2560]
        
        return image_features
    
    def get_text_image_tokens(self, images):
        image_features = self.get_model().get_vision_tower()(images) 
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        seg_norm = cfg.MODEL.MASK_FORMER.SEG_NORM
        seg_proj = cfg.MODEL.MASK_FORMER.SEG_PROJ
        seg_fuse_score = cfg.MODEL.MASK_FORMER.FUSE_SCORE

        qdti_apply = (
            str(getattr(self.config, "qdti_apply_layers", "last3"))
            if self._qdti_bias_enabled()
            else None
        )

        predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(in_channels,
                                                                     hidden_dim,
                                                                     num_queries,
                                                                     nheads,
                                                                     dim_feedforward,
                                                                     dec_layers,
                                                                     pre_norm,
                                                                     mask_dim,
                                                                     enforce_input_project,
                                                                     seg_norm,
                                                                     seg_proj,
                                                                     seg_fuse_score,
                                                                     qdti_apply_layers=qdti_apply,)
        if self._qdti_bias_enabled():
            predictor.register_module(
                "query_specific_text_memory_bias",
                QuerySpecificTextMemoryBias(
                    text_dim=int(getattr(self.config, "dgp_fuse_dim", 256)),
                    memory_dim=int(hidden_dim),
                    query_dim=int(hidden_dim),
                    bias_dim=int(getattr(self.config, "qdti_bias_dim", 128)),
                    init_std=float(getattr(self.config, "qdti_init_std", 1e-3)),
                    max_abs=float(getattr(self.config, "qdti_max_abs", 0.01)),
                    qdti_scale_init=float(getattr(self.config, "qdti_scale_init", 0.0)),
                ),
            )
        return predictor


    def get_model(self):
        return self.model
    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES
        out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        num_features = [int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2 ** i) for i in
                        range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))]
        out_feature_channels = {
            "res2": num_features[0],
            "res3": num_features[1],
            "res4": num_features[2],
            "res5": num_features[3],
        }
        backbone_feature_shape = dict()
        for name in out_features:
            backbone_feature_shape[name] = Dict(
                {'channel': out_feature_channels[name], 'stride': out_feature_strides[name]})
        return backbone_feature_shape

    def get_encoder_image(self, images):
        encode_image_features = self.get_model().get_vision_tower()(images)
        return encode_image_features

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ["res3", "res4", "res5"]

        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                 transformer_dropout,
                                                 transformer_nheads,
                                                 transformer_dim_feedforward,
                                                 transformer_enc_layers,
                                                 conv_dim,
                                                 mask_dim,
                                                 transformer_in_features,
                                                 common_stride)
        return pixel_decoder
    
    def prepare_targets(self, targets, images):
        
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        has_gt_ids = False
        if hasattr(targets[0], 'gt_ids'):
            has_gt_ids = True
        for targets_per_image in targets:
            if has_gt_ids:
                inst_ids = targets_per_image.gt_ids
                valid_id = inst_ids!=-1
            else:
                inst_ids = None
                valid_id = None
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "valid": valid_id,
                    "inst_id": inst_ids,
                }
            )
        return new_targets

    def get_special_token(self, SEG, EOS):
        self.SEG_id = SEG
        self.EOS_id = EOS

    def embed_refer_ids(self, refer_ids):
        if refer_ids is None:
            return None
        embedded_refer = self.get_model().embed_tokens(refer_ids)
        return embedded_refer

    @staticmethod
    def _class_hints_from_token_refer_id(token_refer_id, tokenizer=None) -> list:
        problem_classes = ("bridge", "vehicle", "ship", "tennis")
        if token_refer_id is None:
            return []
        if torch.is_tensor(token_refer_id):
            refer_items = [token_refer_id]
        elif isinstance(token_refer_id, (list, tuple)):
            refer_items = list(token_refer_id)
        else:
            refer_items = [token_refer_id]

        hints = []
        for refer_ids in refer_items:
            if refer_ids is None or (torch.is_tensor(refer_ids) and refer_ids.numel() == 0):
                hints.append([])
                continue
            if tokenizer is not None:
                text = tokenizer.decode(
                    refer_ids.detach().cpu().tolist() if torch.is_tensor(refer_ids) else refer_ids,
                    skip_special_tokens=True,
                ).lower()
            else:
                text = ""
            hints.append([cls for cls in problem_classes if cls in text])
        return hints

    def concat_image_seg_cls_embeds(self, input_id, img_feature, label, SEG_token_embedding_indices=None, refer_embedding=None):
        image_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
        assert len(image_token_indices) == 1, 'not supporting multi image index'
        
        image_features_indices = []
        cur_new_input_embeds = []
        if label is not None:
            cur_new_label = []
            assert label.shape == input_id.shape
        else:
            cur_new_label = None
        
        cur_SEG_token_embedding_indices = [] if SEG_token_embedding_indices is not None else None
        cur_refer_span_indices = []
        track_refer_span = SEG_token_embedding_indices is not None
        
        chunks = []
        current_chunk = []

        for id in input_id:
            if id >= 0:
                current_chunk.append(id.item())
            else:
                if current_chunk:
                    chunks.append(torch.tensor(current_chunk, device=input_id.device))
                    current_chunk = []
                chunks.append([id])
        if current_chunk:
            chunks.append(torch.tensor(current_chunk, device=input_id.device))

       
        for chunk in chunks:
            chunk_len = len(chunk)
            if chunk_len == 1 and chunk[0] == IMAGE_TOKEN_INDEX:
                cur_new_input_embeds.append(img_feature)
                image_features_indices.append(torch.ones(img_feature.shape[0]))
                if SEG_token_embedding_indices is not None:
                    cur_SEG_token_embedding_indices.append(torch.full((img_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if track_refer_span:
                    cur_refer_span_indices.append(torch.zeros(img_feature.shape[0], device=input_id.device, dtype=torch.bool))
                if label is not None:
                    cur_new_label.append(
                        torch.full((img_feature.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
                  
            elif chunk_len == 1 and chunk[0] == REFER_TOKEN_INDEX:
                refer_embed = refer_embedding
                if len(refer_embed.shape) == 1:
                    refer_embed = refer_embed.unsqueeze(0)
                cur_new_input_embeds.append(refer_embed)
                image_features_indices.append(torch.zeros(refer_embed.shape[0]))
                
                if SEG_token_embedding_indices is not None:
                    cur_SEG_token_embedding_indices.append(
                        torch.full((refer_embed.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if track_refer_span:
                    cur_refer_span_indices.append(torch.ones(refer_embed.shape[0], device=input_id.device, dtype=torch.bool))
                if label is not None:
                    cur_new_label.append(
                        torch.full((refer_embed.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
            
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(input_id[:chunk_len]))
                image_features_indices.append(torch.zeros(chunk_len))
                
                if SEG_token_embedding_indices is not None:
                    cur_SEG_token_embedding_indices.append(SEG_token_embedding_indices[:chunk_len])
                if track_refer_span:
                    cur_refer_span_indices.append(torch.zeros(chunk_len, device=input_id.device, dtype=torch.bool))
                if label is not None:
                    cur_new_label.append(label[:chunk_len])

            input_id = input_id[chunk_len:]
            
            if SEG_token_embedding_indices is not None:
                SEG_token_embedding_indices = SEG_token_embedding_indices[chunk_len:]
            if label is not None:
                label = label[chunk_len:]

        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        if label is not None:
            cur_new_label = [x.to(device=self.device) for x in cur_new_label]
            cur_new_label = torch.cat(cur_new_label, dim=0)
        
        if SEG_token_embedding_indices is not None:
            cur_SEG_token_embedding_indices = [x.to(device=self.device) for x in cur_SEG_token_embedding_indices]
            cur_SEG_token_embedding_indices = torch.cat(cur_SEG_token_embedding_indices, dim=0)

        cur_refer_span_mask = None
        if track_refer_span:
            cur_refer_span_indices = [x.to(device=self.device) for x in cur_refer_span_indices]
            cur_refer_span_mask = torch.cat(cur_refer_span_indices, dim=0).bool()
        
        if image_features_indices:
            image_features_indices = [x.to(device=self.device) for x in image_features_indices]
            image_features_indices = torch.cat(image_features_indices, dim=0)

        return cur_new_input_embeds, cur_new_label, cur_SEG_token_embedding_indices, image_features_indices, cur_refer_span_mask

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, token_refer_id=None, SEG_token_embedding_indices=None):

        vision_tower = self.get_vision_tower()
        
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None, None, None

        image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_image_features_indices = []
        
        new_SEG_token_embedding_indices = [] if SEG_token_embedding_indices is not None else None
        new_refer_span_mask = [] if SEG_token_embedding_indices is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_image_feature = image_features[batch_idx]
            
            cur_SEG_token_embedding_indices = SEG_token_embedding_indices[batch_idx] if SEG_token_embedding_indices is not None else None
            
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue

            if labels is not None:
                cur_label = labels[batch_idx]
            else:
                cur_label = None

            if token_refer_id is not None:
                cur_token_refer_id = token_refer_id[batch_idx]
            else:
                cur_token_refer_id = None

            cur_refer_embedding = self.embed_refer_ids(cur_token_refer_id)

            cur_input_embeds, cur_label, cur_SEG_token_embedding_indices, cur_image_features_indices, cur_refer_span_mask = self.concat_image_seg_cls_embeds(
                input_id=cur_input_ids,
                img_feature=cur_image_feature,
                label=cur_label,
                SEG_token_embedding_indices=cur_SEG_token_embedding_indices,
                refer_embedding=cur_refer_embedding
            )

            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(cur_label)

            if SEG_token_embedding_indices is not None:
                new_SEG_token_embedding_indices.append(cur_SEG_token_embedding_indices)
                new_refer_span_mask.append(cur_refer_span_mask)

            if new_image_features_indices is not None:
                new_image_features_indices.append(cur_image_features_indices)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)),
                                          dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
            
            if SEG_token_embedding_indices is not None:
                new_SEG_token_embedding_indices_align = []
                new_refer_span_mask_align = []
                for new_SEG_token_embedding_indice, new_refer_span in zip(
                    new_SEG_token_embedding_indices, new_refer_span_mask
                ):
                    new_SEG_token_embedding_indice = torch.cat(
                        (new_SEG_token_embedding_indice,
                         torch.zeros((max_len - new_SEG_token_embedding_indice.shape[0]),dtype=new_SEG_token_embedding_indice.dtype, device=new_SEG_token_embedding_indice.device)),
                        dim=0)
                    new_refer_span = torch.cat(
                        (new_refer_span,
                         torch.zeros((max_len - new_refer_span.shape[0]), dtype=torch.bool, device=new_refer_span.device)),
                        dim=0)
                    new_SEG_token_embedding_indices_align.append(new_SEG_token_embedding_indice)
                    new_refer_span_mask_align.append(new_refer_span)
                new_SEG_token_embedding_indices = torch.stack(new_SEG_token_embedding_indices_align, dim=0)
                new_refer_span_mask = torch.stack(new_refer_span_mask_align, dim=0)
            
            if new_image_features_indices is not None:
                new_image_features_indices_align = []
                for new_image_features_indice in new_image_features_indices:
                    new_image_features_indice = torch.cat(
                        (new_image_features_indice,
                         torch.zeros((max_len - new_image_features_indice.shape[0]),dtype=new_image_features_indice.dtype, device=new_image_features_indice.device)),
                        dim=0)
                    new_image_features_indices_align.append(new_image_features_indice)
                new_image_features_indices = torch.stack(new_image_features_indices_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
            
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if SEG_token_embedding_indices is not None:
                new_SEG_token_embedding_indices = torch.stack(new_SEG_token_embedding_indices, dim=0)
                new_refer_span_mask = torch.stack(new_refer_span_mask, dim=0)

            if new_image_features_indices is not None:
                new_image_features_indices = torch.stack(new_image_features_indices, dim=0)
            
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
   
        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_SEG_token_embedding_indices, new_image_features_indices, new_refer_span_mask
    
    def get_SEG_embedding(self, hidden_states, SEG_embedding_indices):
        SEG_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, SEG_embedding_indices):
            current_refer_state = current_hidden_state[current_token_indice.bool()]
            SEG_embedding_list.append(current_refer_state)
        return torch.cat(SEG_embedding_list, dim=0).unsqueeze(1)
           
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images_clip: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            token_refer_id=None,
            SEG_token_embedding_indices=None,
            global_step=None,
            mask_num=None,
            dataset_type=None,) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), f'this batch contain different dataset_type: {dataset_type}'
            batch_dataset_type = dataset_type[0]
        else:
            batch_dataset_type = []
        output_attentions = True

        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (SEG_token_embedding_indices == 1).sum() != 0:

            # for generative mode only the 1th stage need
            if input_ids.shape[1] != 1:
                image_features = self.get_vision_tower_feature(images)
                bs = input_ids.shape[0]
            
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, SEG_token_embedding_indices, image_features_indices, refer_span_mask = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images_clip,
                token_refer_id=token_refer_id, SEG_token_embedding_indices=SEG_token_embedding_indices)
        else:
            refer_span_mask = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        attentions = [attention_item.sum(dim=1) for attention_item in outputs.attentions]

        text_memory = None
        text_memory_mask = None
        if self._dgp_qdti_enabled():
            class_hints = self._class_hints_from_token_refer_id(
                token_refer_id, tokenizer=getattr(self, "tokenizer", None)
            )
            q_ref, text_memory, text_memory_mask = self._compute_seg_embedding_with_dgp(
                hidden_states,
                attention_mask,
                SEG_token_embedding_indices,
                image_features_indices=image_features_indices,
                refer_span_mask=refer_span_mask,
                labels=labels,
                class_hints=class_hints,
            )
            if not self._qdti_bias_enabled():
                text_memory = None
                text_memory_mask = None
        else:
            SEG_embedding = self.SEG_token_projector(
                self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices)
            )
        
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)
        mask_num = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num, dim=0)
            for feat in multi_scale_features
        ]

        if self._dgp_qdti_enabled():
            SEG_embedding = expand_bq_for_mask_num(q_ref, mask_num)
            if SEG_embedding.dim() != 3:
                raise ValueError(f"SEG_embedding must be [B,Q,256], got {tuple(SEG_embedding.shape)}")

        tm_rep, tmm_rep = self._repeat_text_memory_for_mask_num(text_memory, text_memory_mask, mask_num)
        if tm_rep is not None and tm_rep.shape[0] != SEG_embedding.shape[0]:
            raise ValueError(
                f"QDTI text_memory batch {tm_rep.shape[0]} != SEG_embedding batch {SEG_embedding.shape[0]}"
            )

        mask_outputs = self.predictor(
            multi_scale_features,
            mask_features,
            None,
            None,
            SEG_embedding,
            text_memory=tm_rep,
            text_memory_mask=tmm_rep,
        )

        # 开始计算loss
        loss = None

        llm_loss = None
        if labels is not None:
            # if seg_query_mask is None or batch_dataset_type in seg_llm_loss_dataset:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            vocab_size = shift_logits.shape[-1]
            shift_logits = shift_logits.view(-1, vocab_size)  # self.config.vocab_size
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            llm_loss = loss_fct(shift_logits, shift_labels)
            
        mask_loss = None
        if seg_info is not None:
            if 'padding_mask' in seg_info[0]:
                if isinstance(seg_info[0]["instances"], list):
                    gt_instances = [x["instances"][0].to(self.device) for x in seg_info]
                else:
                    gt_instances = [x["instances"].to(self.device) for x in seg_info]

                targets = self.prepare_targets(gt_instances, images)
            elif 'mask' in seg_info[0]:
                targets = []
                for gt_mask in seg_info:
                    targets.append(
                        {
                            'labels': torch.tensor([0]).to(mask_outputs['pred_masks'].device),
                            'masks': gt_mask['mask'].to(mask_outputs['pred_masks'].device),
                            'valid': None,
                            'inst_id': None
                        }
                    )
            else:
                targets = None
            mask_losses = self.criterion(mask_outputs, targets)
            weight_dict = self.weight_dict

            loss_mask = 0.0
            loss_dice = 0.0
        
            for k in list(mask_losses.keys()):
                if k in weight_dict:
                    if mask_losses[k] is not None:
                        mask_losses[k] *= weight_dict[k]
                    
                    if '_mask' in k:
                        loss_mask += mask_losses[k]
                    
                    elif '_dice' in k:
                        loss_dice += mask_losses[k]
                else:
                    mask_losses.pop(k)
            mask_loss = loss_mask + loss_dice

        loss_attention = None
        masks = [_seg_info['mask'] for _seg_info in seg_info]
        masks_resized = [
            F.interpolate(m.unsqueeze(0).float(), size=(800, 800), mode="nearest").squeeze(0)
            for m in masks
        ]
        masks = torch.stack(masks_resized, dim=0) # [4, 1, 800, 800]
        masks_down = F.interpolate(masks, size=(27, 27), mode="bilinear", align_corners=False)
        masks_down = masks_down.view(masks_down.size(0), -1)
        masks_down[masks_down > 0] = 1
        
        loss_attention = torch.tensor(0.0, device=mask_loss.device)           
        for full_attention_map in attentions:
            batch_attentions_list = []
            for batch_idx in range(bs):
                attention_map = full_attention_map[batch_idx]
                SEG_mask = SEG_token_embedding_indices[batch_idx].bool()
                image_features_mask = image_features_indices[batch_idx].bool()
                attention = attention_map[SEG_mask][:, image_features_mask] # [1, 729]
                batch_attentions_list.append(attention)
            batch_attentions = torch.cat(batch_attentions_list, dim=0) # [4, 729]
            loss_attention += self.attention_loss(batch_attentions, masks_down)
                             
        loss = llm_loss + mask_loss + 0.01 * loss_attention

        return CausalOutputWithMask(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_mask=loss_mask.detach(),
            loss_dice=loss_dice.detach(),
            loss_llm=llm_loss.detach(),
            loss_attention=0.01 * loss_attention.detach(),
        )
    
    def eval_seg(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images_clip: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            token_refer_id=None,
            SEG_token_embedding_indices=None,
            mask_num = None,
            dgp_use_refined_query: bool = True):
        
        output_attentions = False
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features = self.get_vision_tower_feature(images)

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, SEG_token_embedding_indices, image_features_indices, refer_span_mask = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images_clip,
            token_refer_id=token_refer_id, SEG_token_embedding_indices=SEG_token_embedding_indices)
    
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state   

        text_memory = None
        text_memory_mask = None
        q_seg_baseline = None
        if self._dgp_qdti_enabled():
            class_hints = self._class_hints_from_token_refer_id(
                token_refer_id, tokenizer=getattr(self, "tokenizer", None)
            )
            q_ref, text_memory, text_memory_mask = self._compute_seg_embedding_with_dgp(
                hidden_states,
                attention_mask,
                SEG_token_embedding_indices,
                image_features_indices=image_features_indices,
                refer_span_mask=refer_span_mask,
                labels=labels,
                class_hints=class_hints,
                is_eval=True,
            )
            if not dgp_use_refined_query:
                seg_mask = SEG_token_embedding_indices.bool()
                seg_hidden_bq, _ = pack_seg_hidden_states_bq(hidden_states, seg_mask)
                q_seg_baseline = self.SEG_token_projector(seg_hidden_bq)
            if not self._qdti_bias_enabled():
                text_memory = None
                text_memory_mask = None
        else:
            SEG_embedding = self.SEG_token_projector(
                self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices)
            )

        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)
    
        images = [image.repeat((num, 1, 1, 1)) for image, num in zip(images, mask_num)]
        images = [s[0] for image_repeat in images for s in torch.split(image_repeat, 1, dim=0)]
        mask_num = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num, dim=0)
            for feat in multi_scale_features
        ]

        if self._dgp_qdti_enabled():
            seg_query = q_seg_baseline if (not dgp_use_refined_query and q_seg_baseline is not None) else q_ref
            SEG_embedding = expand_bq_for_mask_num(seg_query, mask_num)
            if SEG_embedding.dim() != 3:
                raise ValueError(f"SEG_embedding must be [B,Q,256], got {tuple(SEG_embedding.shape)}")

        tm_rep, tmm_rep = self._repeat_text_memory_for_mask_num(text_memory, text_memory_mask, mask_num)
        if tm_rep is not None and tm_rep.shape[0] != SEG_embedding.shape[0]:
            raise ValueError(
                f"QDTI text_memory batch {tm_rep.shape[0]} != SEG_embedding batch {SEG_embedding.shape[0]}"
            )

        mask_outputs = self.predictor(
            multi_scale_features,
            mask_features,
            None,
            None,
            SEG_embedding,
            text_memory=tm_rep,
            text_memory_mask=tmm_rep,
        )

        
        mask_pred_results = mask_outputs["pred_masks"]
        images = ImageList.from_tensors(images, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        
        processed_results = []
        for _seg_info, mask_pred_result in zip(seg_info, mask_pred_results):
            instance_r = {
                'pred': ((mask_pred_result.detach().float().cpu().numpy() > 0) * 255).astype(np.uint8),
                'image_name': _seg_info['image_id'],
                'id': _seg_info['data_id'],
                'mask_id': _seg_info['mask_id'],
            }
            processed_results.append(instance_r)
        return processed_results