import os
import sys
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import transformers
from transformers import SiglipImageProcessor
from peft import LoraConfig, get_peft_model
import warnings
import copy
from deepspeed.profiling.flops_profiler import get_model_profile

from segearth_r2.datasets.dataset import *
from llava_trainer import LLaVATrainer
from segearth_r2.model.language_model.llava_phi import SegEarthR2

warnings.filterwarnings('ignore')
local_rank = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="pretrained_model/mllm/Mipha-3B")
   
    version: Optional[str] = field(default="phi-2")

    freeze_backbone: bool = field(default=False)
    train_clip_backbone: bool = field(default=False)
    train_swin_backbone: bool = field(default=False)

    vision_tower: str = "pretrained_model/CLIP/siglip-so400m-patch14-384"
    vision_tower_mask: str = "pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
    with_norm: bool = field(default=True)
    with_layernorm: bool = field(default=False)
    skip_init_vision: bool = field(default=False)
    swin_type: Optional[str] = field(default="base")
    projector_outdim: Optional[int] = field(default=2048)
    mm_projector_type: Optional[str] = field(default="swin_conv")
    model_version: Optional[str] = field(default="v1")
    load_mask2former: bool = field(default=True)
    mask_config: Optional[str] = field(default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    mm_use_im_patch_token: bool = field(default=False)
    mm_use_im_start_end: bool = field(default=False)

@dataclass
class DataArguments:
    lazy_preprocess: bool = True
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    base_data_path: str = '/data1/xzp/data'
    data_ratio: str = '1'  
    switch_bs: int = 4 # 16
    fix_dataset_len: int = 0
    segmentation: bool = True
    dataset_name: str = field(default="rrsisd")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    dataloader_prefetch_factor: int = field(default=2)
    dataloader_num_workers: int = field(default=4)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=False)
    deepspeed: Optional[str] = field(default='scripts/zero1.json')
    
    output_dir: Optional[str] = field(default="output/model")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=True)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = field(default=True)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    dataloader_drop_last: bool = True
    unfreeze_llm_last_n: int = field(
        default=0,
        metadata={
            "help": "Test 4: unfreeze last N LLM decoder layers (full params). 0 keeps existing behavior."
        },
    )
    freeze_pixel_decoder: bool = field(
        default=False,
        metadata={
            "help": "Test 4: freeze pixel_decoder to isolate query-side training."
        },
    )
    test4_mode: bool = field(
        default=False,
        metadata={
            "help": "Test 4 preset: unfreeze_llm_last_n=2, freeze_pixel_decoder=True, lora_enable=False."
        },
    )
    llm_lr: float = field(
        default=1e-5,
        metadata={"help": "Test 4: learning rate for unfrozen LLM last-N layer full params."},
    )
    mask_lr: float = field(
        default=1e-4,
        metadata={"help": "Test 4: learning rate for SEG_token_projector, predictor, lm_head."},
    )
    freeze_predictor: bool = field(
        default=False,
        metadata={"help": "Test 4 ablation: freeze predictor; train LLM last-N + SEG_token_projector only."},
    )
    freeze_lm_head: bool = field(
        default=False,
        metadata={
            "help": "Test 4: freeze lm_head. test4_mode defaults to True unless --no_freeze_lm_head is passed."
        },
    )
    use_tcpd: bool = field(
        default=False,
        metadata={"help": "Enable SEG-conditioned TCPD inside pixel decoder (default off)."},
    )
    tcpd_condition_source: str = field(
        default="seg",
        metadata={"help": "TCPD condition source; only 'seg' ([SEG] embedding) is supported."},
    )
    tcpd_train: bool = field(
        default=False,
        metadata={
            "help": "When use_tcpd=True, unfreeze pixel_decoder params whose names contain 'tcpd' "
            "even if freeze_pixel_decoder=True."
        },
    )
    tcpd_spatial_mode: str = field(
        default="spatial",
        metadata={"help": "TCPD MSDeformAttn mode: 'global' (v1 broadcast) or 'spatial' (v2 per-token)."},
    )
    tcpd_condition_msdeform: bool = field(
        default=True,
        metadata={"help": "Apply TCPD conditioning inside MSDeformAttn."},
    )
    tcpd_condition_fpn: bool = field(
        default=True,
        metadata={"help": "Apply TCPD conditioning on FPN top-down fusion."},
    )
    tcpd_condition_output_scale: bool = field(
        default=True,
        metadata={"help": "Apply TCPD scale fusion on pixel decoder outputs."},
    )
    freeze_seg_projector: bool = field(
        default=False,
        metadata={"help": "Freeze SEG_token_projector (required for pure TCPD-only ablation)."},
    )


def apply_test4_mode_defaults(training_args):
    if not training_args.test4_mode:
        return
    training_args.unfreeze_llm_last_n = 2
    training_args.freeze_pixel_decoder = True
    training_args.lora_enable = False
    if "--no_freeze_lm_head" not in sys.argv:
        training_args.freeze_lm_head = True


def get_llm_decoder_layers(model):
    base = model.base_model if hasattr(model, "base_model") else model
    inner = base.model if hasattr(base, "model") else base
    if not hasattr(inner, "layers"):
        raise ValueError(
            f"Cannot find LLM decoder layers; expected `.model.layers`, got {type(inner)}"
        )
    return inner.layers


def unfreeze_llm_last_n_layers(model, n: int):
    layers = get_llm_decoder_layers(model)
    if n <= 0:
        return [], []
    n = min(n, len(layers))
    total = len(layers)
    unfrozen = []
    indices = []
    for i in range(total - n, total):
        for p in layers[i].parameters():
            p.requires_grad = True
        unfrozen.append(f"model.model.layers.{i}")
        indices.append(i)
    return unfrozen, indices


def freeze_module_by_keyword(model, keyword: str):
    for name, p in model.named_parameters():
        if keyword in name:
            p.requires_grad = False


def unfreeze_module_by_keyword(model, keyword: str):
    for name, p in model.named_parameters():
        if keyword in name:
            p.requires_grad = True


def module_is_trainable(model, keyword: str) -> bool:
    params = [p for n, p in model.named_parameters() if keyword in n]
    if not params:
        return False
    return any(p.requires_grad for p in params)


def module_trainable_param_count(model, keyword: str) -> int:
    return sum(
        p.numel() for n, p in model.named_parameters()
        if keyword in n and p.requires_grad
    )


def pixel_decoder_trainable_counts(model):
    """Return (total, tcpd, non_tcpd) trainable param counts under pixel_decoder."""
    total = tcpd = non_tcpd = 0
    for name, p in model.named_parameters():
        if "pixel_decoder" not in name or not p.requires_grad:
            continue
        n = p.numel()
        total += n
        if "tcpd" in name:
            tcpd += n
        else:
            non_tcpd += n
    return total, tcpd, non_tcpd


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def is_pure_tcpd_mode(training_args) -> bool:
    """Pure TCPD-only: train only pixel_decoder *tcpd* params."""
    return (
        training_args.use_tcpd
        and training_args.tcpd_train
        and training_args.freeze_pixel_decoder
        and training_args.freeze_predictor
        and training_args.freeze_lm_head
        and getattr(training_args, "freeze_seg_projector", False)
        and training_args.unfreeze_llm_last_n == 0
        and not training_args.test4_mode
    )


def log_test4_train_config(model, training_args, unfrozen_layer_names, unfrozen_layer_indices):
    if training_args.local_rank not in (-1, 0):
        return None
    layers = get_llm_decoder_layers(model)
    total_layers = len(layers)
    trainable, total = count_trainable_params(model)
    pct = 100.0 * trainable / total if total else 0.0
    lines = [
        f"[Test4] test4_mode: {training_args.test4_mode}",
        f"[Test4] unfreeze_llm_last_n: {training_args.unfreeze_llm_last_n}",
        f"[Test4] freeze_pixel_decoder: {training_args.freeze_pixel_decoder}",
        f"[Test4] freeze_predictor: {training_args.freeze_predictor}",
        f"[Test4] freeze_seg_projector: {getattr(training_args, 'freeze_seg_projector', False)}",
        f"[Test4] freeze_lm_head: {training_args.freeze_lm_head}",
        f"[Test4] use_tcpd: {getattr(training_args, 'use_tcpd', False)}",
        f"[Test4] tcpd_spatial_mode: {getattr(training_args, 'tcpd_spatial_mode', 'spatial')}",
        f"[Test4] tcpd_condition_msdeform: {getattr(training_args, 'tcpd_condition_msdeform', True)}",
        f"[Test4] tcpd_condition_fpn: {getattr(training_args, 'tcpd_condition_fpn', True)}",
        f"[Test4] tcpd_condition_output_scale: {getattr(training_args, 'tcpd_condition_output_scale', True)}",
        f"[Test4] tcpd_train: {getattr(training_args, 'tcpd_train', False)}",
        f"[Test4] lora_enable: {training_args.lora_enable}",
        f"[Test4] llm_lr: {training_args.llm_lr}",
        f"[Test4] mask_lr: {training_args.mask_lr}",
        f"[Test4] total LLM layers: {total_layers}",
        f"[Test4] pure_tcpd_mode: {is_pure_tcpd_mode(training_args)}",
    ]
    if unfrozen_layer_indices:
        idx_str = ", ".join(str(i) for i in unfrozen_layer_indices)
        lines.append(f"[Test4] unfrozen LLM layer indices: {idx_str} / total {total_layers}")
        lines.append("[Test4] unfrozen LLM layers:")
        for name in unfrozen_layer_names:
            lines.append(f"  - {name}")
    else:
        lines.append("[Test4] unfrozen LLM layer indices: (none)")
    for kw in ("pixel_decoder", "SEG_token_projector", "predictor", "lm_head"):
        lines.append(
            f"[Test4] {kw}: trainable={module_is_trainable(model, kw)}, "
            f"trainable_params={module_trainable_param_count(model, kw):,}"
        )
    lines.append(f"[Test4] trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    if training_args.freeze_pixel_decoder:
        pd_total, pd_tcpd, pd_non_tcpd = pixel_decoder_trainable_counts(model)
        lines.append(
            f"[Test4] pixel_decoder trainable: total={pd_total:,}, "
            f"tcpd={pd_tcpd:,}, non_tcpd={pd_non_tcpd:,}"
        )
        use_tcpd = getattr(training_args, "use_tcpd", False)
        tcpd_train = getattr(training_args, "tcpd_train", False)
        if use_tcpd and tcpd_train:
            if pd_non_tcpd != 0:
                raise RuntimeError(
                    f"[Test4] ASSERT FAIL: non-TCPD pixel_decoder has {pd_non_tcpd} trainable params "
                    "when freeze_pixel_decoder=True and tcpd_train=True"
                )
            if pd_tcpd == 0:
                raise RuntimeError(
                    "[Test4] ASSERT FAIL: no TCPD trainable params when "
                    "freeze_pixel_decoder=True, use_tcpd=True, tcpd_train=True"
                )
            lines.append(
                "[Test4] ASSERT OK: only TCPD pixel_decoder params trainable "
                f"({pd_tcpd:,} params)"
            )
        else:
            if pd_total != 0:
                raise RuntimeError(
                    f"[Test4] ASSERT FAIL: pixel_decoder has {pd_total} trainable params "
                    "when freeze_pixel_decoder=True"
                )
            lines.append("[Test4] ASSERT OK: pixel_decoder trainable params = 0")
    if is_pure_tcpd_mode(training_args):
        pred_n = module_trainable_param_count(model, "predictor")
        seg_n = module_trainable_param_count(model, "SEG_token_projector")
        lm_n = module_trainable_param_count(model, "lm_head")
        if pred_n != 0:
            raise RuntimeError(
                f"[Test4] ASSERT FAIL: pure TCPD mode but predictor has {pred_n} trainable params"
            )
        if seg_n != 0:
            raise RuntimeError(
                f"[Test4] ASSERT FAIL: pure TCPD mode but SEG_token_projector has {seg_n} trainable params"
            )
        if lm_n != 0:
            raise RuntimeError(
                f"[Test4] ASSERT FAIL: pure TCPD mode but lm_head has {lm_n} trainable params"
            )
        if unfrozen_layer_indices:
            raise RuntimeError(
                "[Test4] ASSERT FAIL: pure TCPD mode but LLM layers are unfrozen"
            )
        lines.append("[Test4] ASSERT OK: pure TCPD — only TCPD pixel_decoder params trainable")
        lines.append(f"[Test4] predictor trainable: {pred_n}")
        lines.append(f"[Test4] SEG_token_projector trainable: {seg_n}")
        lines.append(f"[Test4] lm_head trainable: {lm_n}")
        lines.append("[Test4] unfrozen LLM layers: none")
    for line in lines:
        print(line)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    if training_args.output_dir:
        os.makedirs(training_args.output_dir, exist_ok=True)
        summary_path = os.path.join(training_args.output_dir, "test4_train_config.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[Test4] config summary saved to: {summary_path}")
        return summary_path
    return None


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def find_linear_layers(model, lora_target_modules=['q_proj', 'v_proj'], train_module_list=[]): 
    cur_train_module_list = copy.deepcopy(train_module_list)
    cur_train_module_list.extend(["vision_tower", "vision_tower_mask"])
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (isinstance(module, cls)
            and all(
                        [
                            x not in name
                            for x in cur_train_module_list
                        ]
                    )
                    and any([x in name for x in lora_target_modules])):
            # names = name.split('.')
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)

    return sorted(list(lora_module_names))


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""


    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def make_unify_datamodule(clip_image_processor, tokenizer, data_args, training_args):
    data_ratio = data_args.data_ratio
    data_ratio = data_ratio.split('||')
    data_ratio = [int(data_) for data_ in data_ratio]
    datasets = []

    dataset_name = data_args.dataset_name.lower()

    if data_ratio[0] != 0:
        if dataset_name == "rrsisd":
            train_dataset = RRSISDDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split="train"
            )
            eval_dataset = RRSISDDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split="val"
            )
        elif dataset_name == "lasers":
            train_dataset = LaSeRSDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split="train_data.json"
            )
            eval_split = "val_data.json"
            eval_json = os.path.join(
                data_args.base_data_path, "val", "annotations", eval_split
            )
            if not os.path.isfile(eval_json):
                eval_split = "train_data.json"
                if training_args.local_rank in (-1, 0):
                    print(
                        f"[WARN] LaSeRS val split missing at {eval_json}; "
                        f"fallback eval_dataset split={eval_split}"
                    )
            eval_dataset = LaSeRSDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split=eval_split
            )
        elif dataset_name == "refsegrs":
            train_dataset = RefSegRSDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split="train"
            )
            eval_dataset = RefSegRSDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split="val"
            )
        elif dataset_name == "risbench":
            train_dataset = RISBenchDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split="train"
            )
            eval_dataset = RISBenchDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split="val"
            )
        else:
            raise ValueError(f"Unsupported dataset_name: {data_args.dataset_name}")

        datasets += [train_dataset] * data_ratio[0]
    else:
        raise ValueError("data_ratio[0] is 0; train dataset is empty.")

    print(f'the dataset ratio is: {data_ratio}')
    print(f'the dataset name is: {data_args.dataset_name}')
    train_dataset = UnifyDatasetSingleDatasetForBatch(
        datasets, data_ratio, data_args.switch_bs, fix_dataset_len=data_args.fix_dataset_len
    )
    print(f'total unify datasest number is {len(train_dataset)}')
    data_collator = DataCollatorForCOCODatasetV2(
        tokenizer=tokenizer, clip_image_processor=clip_image_processor
    )
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    apply_test4_mode_defaults(training_args)
    if training_args.seed is None:
        training_args.seed = 42
    if training_args.data_seed is None:
        training_args.data_seed = 42
    local_rank = training_args.local_rank
    transformers.set_seed(training_args.seed)
    if training_args.local_rank in (-1, 0):
        print(f"[Seed] Set global seed before model init: {training_args.seed}")
    if training_args.test4_mode and training_args.local_rank in (-1, 0):
        print(
            "[Test4] test4_mode enabled: "
            f"unfreeze_llm_last_n={training_args.unfreeze_llm_last_n}, "
            f"freeze_pixel_decoder={training_args.freeze_pixel_decoder}, "
            f"lora_enable={training_args.lora_enable}, "
            f"freeze_lm_head={training_args.freeze_lm_head}"
        )
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)) # 用不着？

    mask_cfg = get_mask_config(config=model_args.mask_config)
    bnb_model_from_pretrained_args = {}

    model = SegEarthR2.from_pretrained(
        model_args.model_name_or_path,
        mask_decoder_cfg=mask_cfg,
        add_cross_attn=True,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
                )

    if not model.is_train_mask_decode:
        mask2former_ckpt = model_args.vision_tower_mask if model_args.load_mask2former else None
        model.initial_mask_module(mask2former_ckpt, model_args)

    model.config.use_cache = False

    model.config.use_tcpd = training_args.use_tcpd
    model.config.tcpd_condition_source = training_args.tcpd_condition_source
    model.config.tcpd_spatial_mode = training_args.tcpd_spatial_mode
    model.config.tcpd_condition_msdeform = training_args.tcpd_condition_msdeform
    model.config.tcpd_condition_fpn = training_args.tcpd_condition_fpn
    model.config.tcpd_condition_output_scale = training_args.tcpd_condition_output_scale
    if training_args.tcpd_condition_source != "seg":
        raise ValueError(f"Unsupported tcpd_condition_source: {training_args.tcpd_condition_source}")
    if training_args.tcpd_spatial_mode not in ("global", "spatial"):
        raise ValueError(f"Unsupported tcpd_spatial_mode: {training_args.tcpd_spatial_mode}")

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)


    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower_mask = model.model.get_vision_tower_mask()
        vision_tower.to(dtype=torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32), device=training_args.device)
        vision_tower_mask.to(dtype=torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32), device=training_args.device)
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints 

        if not model_args.train_clip_backbone:
            model.model.vision_tower.requires_grad_(False)
        if not model_args.train_swin_backbone:
            model.model.vision_tower_mask.requires_grad_(False)

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

    tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))
    train_module_list = [
        "lm_head", "pixel_decoder", "predictor", "SEG_token_projector",
    ]
    if training_args.freeze_pixel_decoder:
        train_module_list = [m for m in train_module_list if m != "pixel_decoder"]
    if training_args.freeze_predictor:
        train_module_list = [m for m in train_module_list if m != "predictor"]
    if getattr(training_args, "freeze_seg_projector", False):
        train_module_list = [m for m in train_module_list if m != "SEG_token_projector"]
    if training_args.freeze_lm_head:
        train_module_list = [m for m in train_module_list if m != "lm_head"]

    if model_args.train_swin_backbone:
        train_module_list.append('vision_tower_mask')

    test4_no_lora = (
        training_args.test4_mode
        or (training_args.unfreeze_llm_last_n > 0 and not training_args.lora_enable)
    )
    if training_args.lora_enable and training_args.unfreeze_llm_last_n > 0 and not training_args.test4_mode:
        if training_args.local_rank in (-1, 0):
            print(
                "[Test4] WARNING: unfreeze_llm_last_n > 0 with lora_enable=True may double-update "
                "q_proj/v_proj in unfrozen layers. Prefer --test4_mode or --no-lora_enable."
            )

    if training_args.lora_enable:
        lora_r = training_args.lora_r
        lora_alpha = training_args.lora_alpha
        lora_dropout = training_args.lora_dropout
        lora_exclude_modules = list(train_module_list)
        if training_args.freeze_pixel_decoder:
            lora_exclude_modules.append("pixel_decoder")
        if training_args.unfreeze_llm_last_n > 0:
            layers = get_llm_decoder_layers(model)
            total_layers = len(layers)
            n = min(training_args.unfreeze_llm_last_n, total_layers)
            for i in range(total_layers - n, total_layers):
                lora_exclude_modules.append(f"layers.{i}")
        lora_target_modules = find_linear_layers(model, train_module_list=lora_exclude_modules)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in train_module_list
                ]):

                p.requires_grad = True
    elif test4_no_lora:
        model.requires_grad_(False)
        for n, p in model.named_parameters():
            if any(x in n for x in train_module_list):
                p.requires_grad = True

    unfrozen_layer_names = []
    unfrozen_layer_indices = []
    if training_args.unfreeze_llm_last_n > 0:
        unfrozen_layer_names, unfrozen_layer_indices = unfreeze_llm_last_n_layers(
            model, training_args.unfreeze_llm_last_n
        )
        training_args.test4_unfrozen_layer_indices = unfrozen_layer_indices

    if training_args.freeze_pixel_decoder:
        freeze_module_by_keyword(model, "pixel_decoder")

    if training_args.use_tcpd and training_args.tcpd_train:
        unfreeze_module_by_keyword(model, "tcpd")

    if training_args.freeze_predictor:
        freeze_module_by_keyword(model, "predictor")

    if getattr(training_args, "freeze_seg_projector", False):
        freeze_module_by_keyword(model, "SEG_token_projector")

    if training_args.freeze_lm_head:
        freeze_module_by_keyword(model, "lm_head")

    if (
        training_args.test4_mode
        or training_args.unfreeze_llm_last_n > 0
        or training_args.freeze_pixel_decoder
        or training_args.freeze_predictor
        or getattr(training_args, "freeze_seg_projector", False)
        or training_args.freeze_lm_head
        or (training_args.use_tcpd and training_args.tcpd_train)
    ):
        log_test4_train_config(
            model, training_args, unfrozen_layer_names, unfrozen_layer_indices
        )

    model.get_special_token(SEG=tokenizer("[SEG]", return_tensors='pt', add_special_tokens=False)['input_ids'], EOS=tokenizer.eos_token_id)
    
    clip_image_processor = SiglipImageProcessor.from_pretrained(model_args.vision_tower)
    
    data_module = make_unify_datamodule(clip_image_processor=clip_image_processor, tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    training_args.dataloader_drop_last = True
    test4_training = (
        training_args.test4_mode or training_args.unfreeze_llm_last_n > 0
    )
    training_args.save_strategy = "steps"
    if training_args.save_steps is None or training_args.save_steps <= 0:
        training_args.save_steps = 500
    if test4_training:
        # Gate on LaSeRS test eval in scripts; skip train-time eval (val missing / huge).
        training_args.load_best_model_at_end = False
        if hasattr(training_args, "evaluation_strategy"):
            training_args.evaluation_strategy = "no"
        if hasattr(training_args, "eval_strategy"):
            training_args.eval_strategy = "no"
        if training_args.save_total_limit is None:
            training_args.save_total_limit = 6
    else:
        if hasattr(training_args, "evaluation_strategy"):
            training_args.evaluation_strategy = "steps"
        if hasattr(training_args, "eval_strategy"):
            training_args.eval_strategy = "steps"
        training_args.eval_steps = training_args.save_steps
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_score"
        training_args.greater_is_better = True
        if training_args.save_total_limit is None or training_args.save_total_limit > 2:
            training_args.save_total_limit = 2
    if test4_training and training_args.local_rank in (-1, 0):
        print(
            f"[Test4] save_total_limit={training_args.save_total_limit}, "
            f"load_best_model_at_end={training_args.load_best_model_at_end}, "
            f"evaluation_strategy=no"
        )
    
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
