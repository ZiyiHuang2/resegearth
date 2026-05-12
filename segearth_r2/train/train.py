import os
import sys
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import transformers
from transformers import SiglipImageProcessor
from peft import LoraConfig, get_peft_model
import warnings
import copy
import torch
import torch.nn as nn
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
    use_attention_loss: bool = field(default=True)
    use_midstage_gate_loss: bool = field(default=False)
    midstage_gate_loss_weight: float = field(default=0.0)
    use_mstva: bool = field(default=False)
    mstva_align_dim: int = field(default=256)
    use_mstva_loss: bool = field(default=False)
    mstva_loss_weight: float = field(default=0.0)
    mstva_scale_weights: str = field(default="0.5,0.3,0.2")
    use_text_film: bool = field(default=False)
    text_film_init_std: float = field(default=1e-3)
    text_film_branch_alpha: float = field(default=1.0)
    text_film_visual_dim: int = field(default=512)
    train_midstage_recalibration: bool = field(default=True)
    stage3_norm_only: bool = field(default=False)

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
    concept_public_semantic_library: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional path to concept_public_semantic_library_v2.json; RRSIS-D train prompts inject matched grounding priors in the user message (raw text first, then priors, then image)."
        },
    )

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
    lora_enable: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    dataloader_drop_last: bool = True


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


def _set_module_requires_grad(module, requires_grad=True):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def _enable_text_film_trainable(model):
    raw_model = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    if not hasattr(raw_model, "text_film_branch") or raw_model.text_film_branch is None:
        return
    for p in raw_model.text_film_branch.parameters():
        p.requires_grad = True


def log_text_film_trainable_params(model, local_rank_value=0):
    if not _is_rank0(local_rank_value):
        return
    raw_model = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    if not hasattr(raw_model, "text_film_branch") or raw_model.text_film_branch is None:
        print("[TextFiLM trainable] text_film_branch not present")
        return
    names = [n for n, p in raw_model.named_parameters() if "text_film_branch" in n and p.requires_grad]
    print(f"[TextFiLM trainable] trainable param count={len(names)}")
    for n in names[:16]:
        print(f"  {n}")
    if len(names) > 16:
        print(f"  ... ({len(names) - 16} more)")


def _enable_midstage_text_recalibration_trainable(model, norm_only=False):
    vision_tower_mask = model.get_model().get_vision_tower_mask()
    if vision_tower_mask is None:
        return

    if hasattr(vision_tower_mask, "mid_stage_text_recalibration"):
        _set_module_requires_grad(vision_tower_mask.mid_stage_text_recalibration, True)
    if hasattr(vision_tower_mask, "mstva") and vision_tower_mask.mstva is not None:
        _set_module_requires_grad(vision_tower_mask.mstva, True)

    if not hasattr(vision_tower_mask, "layers") or len(vision_tower_mask.layers) <= 2:
        return

    stage3 = vision_tower_mask.layers[2]
    for module in stage3.modules():
        if isinstance(module, nn.LayerNorm):
            _set_module_requires_grad(module, True)

    if hasattr(vision_tower_mask, "norm2"):
        _set_module_requires_grad(vision_tower_mask.norm2, True)

    if not norm_only and hasattr(stage3, "blocks") and len(stage3.blocks) > 0:
        _set_module_requires_grad(stage3.blocks[-1], True)


def _is_rank0(local_rank_value):
    return local_rank_value in (None, -1, 0)


def log_midstage_recalibration_trainable_params(model, local_rank_value=0, max_show=8):
    if not _is_rank0(local_rank_value):
        return

    raw_model = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    if not hasattr(raw_model, "get_model"):
        print("[MidStage trainable] unable to inspect model wrapper")
        return

    core_model = raw_model.get_model()
    vision_tower_mask = core_model.get_vision_tower_mask()
    if vision_tower_mask is None or not hasattr(vision_tower_mask, "layers") or len(vision_tower_mask.layers) <= 2:
        print("[MidStage trainable] vision_tower_mask/stage3 not available")
        return

    stage3 = vision_tower_mask.layers[2]
    last_block_idx = len(stage3.blocks) - 1 if hasattr(stage3, "blocks") and len(stage3.blocks) > 0 else None

    named_params = list(raw_model.named_parameters())
    midstage_prefix = "model.vision_tower_mask.mid_stage_text_recalibration."
    stage3_last_block_prefix = None if last_block_idx is None else f"model.vision_tower_mask.layers.2.blocks.{last_block_idx}."

    midstage_params = [(n, p) for n, p in named_params if n.startswith(midstage_prefix)]
    stage3_last_block_params = [] if stage3_last_block_prefix is None else [(n, p) for n, p in named_params if n.startswith(stage3_last_block_prefix)]
    stage3_norm_params = [
        (n, p) for n, p in named_params
        if (n.startswith("model.vision_tower_mask.layers.2.") and ".norm" in n)
        or n.startswith("model.vision_tower_mask.norm2.")
    ]

    def _count_trainable(param_items):
        trainable_items = [name for name, param in param_items if param.requires_grad]
        return len(param_items), len(trainable_items), trainable_items

    mid_total, mid_trainable, mid_names = _count_trainable(midstage_params)
    blk_total, blk_trainable, blk_names = _count_trainable(stage3_last_block_params)
    norm_total, norm_trainable, norm_names = _count_trainable(stage3_norm_params)

    print(f"[MidStage trainable] module(mid_stage_text_recalibration): total={mid_total}, trainable={mid_trainable}")
    if mid_trainable == 0:
        print("[MidStage trainable][WARNING] no trainable params in mid_stage_text_recalibration")
    print(f"[MidStage trainable] stage3 last block: total={blk_total}, trainable={blk_trainable}")
    print(f"[MidStage trainable] stage3 norm params: total={norm_total}, trainable={norm_trainable}")

    sample_names = (mid_names + blk_names + norm_names)[:max_show]
    if sample_names:
        print(f"[MidStage trainable] sample trainable params: {sample_names}")


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
            eval_dataset = LaSeRSDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split="val_data.json"
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
    if model_args.use_text_film and model_args.use_mstva:
        print("[train] use_text_film=True: forcing use_mstva=False (mutually exclusive).")
        model_args.use_mstva = False
    if training_args.seed is None:
        training_args.seed = 42
    if training_args.data_seed is None:
        training_args.data_seed = 42
    local_rank = training_args.local_rank
    transformers.set_seed(training_args.seed)
    if training_args.local_rank in (-1, 0):
        print(f"[Seed] Set global seed before model init: {training_args.seed}")
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
    model.config.use_attention_loss = model_args.use_attention_loss
    model.config.use_midstage_gate_loss = model_args.use_midstage_gate_loss
    model.config.midstage_gate_loss_weight = model_args.midstage_gate_loss_weight
    model.config.use_mstva = model_args.use_mstva
    model.config.mstva_align_dim = model_args.mstva_align_dim
    model.config.use_mstva_loss = model_args.use_mstva_loss
    model.config.mstva_loss_weight = model_args.mstva_loss_weight
    model.config.mstva_scale_weights = model_args.mstva_scale_weights
    model.config.use_text_film = model_args.use_text_film
    model.config.text_film_init_std = model_args.text_film_init_std
    model.config.text_film_branch_alpha = model_args.text_film_branch_alpha
    model.config.text_film_visual_dim = model_args.text_film_visual_dim
    model.config.text_film_eval_mode = "normal"
    model.config.text_film_force_alpha = 1.0
    model.ensure_text_film_branch()

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
        if getattr(model, "text_film_branch", None) is not None:
            model.text_film_branch.to(
                dtype=torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32),
                device=training_args.device,
            )
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints 

        if not model_args.train_clip_backbone:
            model.model.vision_tower.requires_grad_(False)
        if not model_args.train_swin_backbone:
            model.model.vision_tower_mask.requires_grad_(False)

        if model_args.train_midstage_recalibration:
            _enable_midstage_text_recalibration_trainable(
                model,
                norm_only=model_args.stage3_norm_only
            )
        if model_args.use_text_film:
            _enable_text_film_trainable(model)

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

    tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))
    train_module_list = [
        "lm_head", "pixel_decoder", "predictor", "SEG_token_projector", "mid_stage_text_recalibration", "mstva",
        "text_film_branch",
    ]

    if model_args.train_swin_backbone:
        train_module_list.append('vision_tower_mask')
        
    if training_args.lora_enable:
        lora_r = training_args.lora_r
        lora_alpha = training_args.lora_alpha
        lora_dropout = training_args.lora_dropout
        lora_target_modules = find_linear_layers(model, train_module_list=train_module_list)
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

        if model_args.train_midstage_recalibration:
            _enable_midstage_text_recalibration_trainable(
                model,
                norm_only=model_args.stage3_norm_only
            )
        if model_args.use_text_film:
            _enable_text_film_trainable(model)

    if model_args.train_midstage_recalibration:
        log_midstage_recalibration_trainable_params(model, local_rank_value=training_args.local_rank)
    if model_args.use_text_film:
        log_text_film_trainable_params(model, local_rank_value=training_args.local_rank)

    model.get_special_token(SEG=tokenizer("[SEG]", return_tensors='pt', add_special_tokens=False)['input_ids'], EOS=tokenizer.eos_token_id)
    
    clip_image_processor = SiglipImageProcessor.from_pretrained(model_args.vision_tower)
    
    data_module = make_unify_datamodule(clip_image_processor=clip_image_processor, tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    training_args.dataloader_drop_last = True
    if hasattr(training_args, "evaluation_strategy"):
        training_args.evaluation_strategy = "steps"
    if hasattr(training_args, "eval_strategy"):
        training_args.eval_strategy = "steps"
    training_args.save_strategy = "steps"
    if training_args.save_steps is None or training_args.save_steps <= 0:
        training_args.save_steps = 500
    training_args.eval_steps = training_args.save_steps
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "eval_score"
    training_args.greater_is_better = True
    if training_args.save_total_limit is None or training_args.save_total_limit > 2:
        training_args.save_total_limit = 2
    
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
