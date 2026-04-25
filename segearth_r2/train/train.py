import os
import sys
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

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
    use_precomputed_structured_maps: bool = field(default=False)
    structured_map_dir: Optional[str] = field(default=None)

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
    max_grad_norm: float = field(default=1.0)
    use_attention_loss: bool = field(default=True)
    use_structured_attention_loss: bool = field(default=False)
    attention_loss_weight: float = field(default=0.01)
    attention_loss_reduction: str = field(default="batchmean")
    attention_loss_last_k_layers: int = field(default=0)
    attention_audit_mode: bool = field(default=False)
    attention_audit_dir: Optional[str] = field(default=None)
    structured_fg_bg_weight: float = field(default=1.0)
    structured_boundary_outer_weight: float = field(default=1.0)
    structured_attention_margin: float = field(default=0.0)
    target_layers: Optional[str] = field(default=None)
    target_heads_config_path: Optional[str] = field(default=None)
    top_k_heads: int = field(default=4)
    small_weight: float = field(default=1.5)
    small_area_ratio_threshold: float = field(default=0.01)
    structured_warmup_steps: int = field(default=0)
    structured_decay_start_step: int = field(default=-1)
    structured_decay_end_step: int = field(default=-1)
    strict_attention_selection: bool = field(default=False)
    structured_log_interval: int = field(default=50)


def _parse_target_layers(target_layers_raw: Optional[str]):
    if target_layers_raw is None:
        return None
    text = str(target_layers_raw).strip()
    if text == "":
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    tokens = [x.strip() for x in text.split(",") if x.strip() != ""]
    if len(tokens) == 0:
        raise ValueError(f"Invalid --target_layers value: {target_layers_raw}")
    try:
        layers = [int(x) for x in tokens]
    except ValueError as e:
        raise ValueError(f"Invalid --target_layers value: {target_layers_raw}") from e
    if len(set(layers)) != len(layers):
        raise ValueError(f"Duplicated layer index in --target_layers: {layers}")
    if min(layers) < 0:
        raise ValueError(f"Layer indices must be non-negative: {layers}")
    return layers


def _load_target_heads_dict(config_path: Optional[str], top_k_heads: int):
    if config_path is None:
        return None
    path = os.path.abspath(config_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"target heads config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or len(payload) == 0:
        raise ValueError("target heads config must be a non-empty dict like {\"24\": [1,2,3,4]}")

    normalized = {}
    for k, v in payload.items():
        try:
            layer_idx = int(k)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid layer key in target heads config: {k}") from e
        if layer_idx < 0:
            raise ValueError(f"Layer index in target heads config must be non-negative: {layer_idx}")
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"Heads for layer {layer_idx} must be a non-empty list")
        try:
            head_indices = [int(x) for x in v]
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid head index list for layer {layer_idx}: {v}") from e
        if any(x < 0 for x in head_indices):
            raise ValueError(f"Head indices must be non-negative for layer {layer_idx}: {head_indices}")
        if len(set(head_indices)) != len(head_indices):
            raise ValueError(f"Duplicated head index for layer {layer_idx}: {head_indices}")
        if top_k_heads > 0:
            head_indices = head_indices[:top_k_heads]
            if len(head_indices) == 0:
                raise ValueError(f"Layer {layer_idx} has no valid heads after top_k_heads={top_k_heads}.")
        normalized[layer_idx] = head_indices
    return normalized


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
        tokenizer=tokenizer,
        clip_image_processor=clip_image_processor,
        use_precomputed_structured_maps=data_args.use_precomputed_structured_maps,
    )
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    parsed_target_layers = None
    parsed_target_heads_dict = None
    if training_args.use_structured_attention_loss:
        parsed_target_layers = _parse_target_layers(training_args.target_layers)
        parsed_target_heads_dict = _load_target_heads_dict(
            training_args.target_heads_config_path,
            training_args.top_k_heads,
        )
    if training_args.seed is None:
        training_args.seed = 42
    if training_args.data_seed is None:
        training_args.data_seed = 42
    local_rank = training_args.local_rank
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
    model.config.use_attention_loss = training_args.use_attention_loss
    model.config.use_structured_attention_loss = training_args.use_structured_attention_loss
    model.config.attention_loss_weight = training_args.attention_loss_weight
    model.config.attention_loss_reduction = training_args.attention_loss_reduction
    model.config.attention_loss_last_k_layers = training_args.attention_loss_last_k_layers
    model.config.attention_audit_mode = training_args.attention_audit_mode
    model.config.attention_audit_dir = training_args.attention_audit_dir
    model.config.structured_fg_bg_weight = training_args.structured_fg_bg_weight
    model.config.structured_boundary_outer_weight = training_args.structured_boundary_outer_weight
    model.config.structured_attention_margin = training_args.structured_attention_margin
    model.config.target_layers = parsed_target_layers
    model.config.target_heads_dict = parsed_target_heads_dict
    model.config.target_heads_config_path = training_args.target_heads_config_path
    model.config.top_k_heads = training_args.top_k_heads
    model.config.small_weight = training_args.small_weight
    model.config.small_area_ratio_threshold = training_args.small_area_ratio_threshold
    model.config.structured_warmup_steps = training_args.structured_warmup_steps
    model.config.structured_decay_start_step = training_args.structured_decay_start_step
    model.config.structured_decay_end_step = training_args.structured_decay_end_step
    model.config.strict_attention_selection = training_args.strict_attention_selection
    model.config.structured_log_interval = training_args.structured_log_interval
    model.config.use_precomputed_structured_maps = data_args.use_precomputed_structured_maps
    model.config.structured_map_dir = data_args.structured_map_dir

    if not model.is_train_mask_decode:
        mask2former_ckpt = model_args.vision_tower_mask if model_args.load_mask2former else None
        model.initial_mask_module(mask2former_ckpt, model_args)
    if hasattr(model, "set_attention_loss_config"):
        model.set_attention_loss_config()

    model.config.use_cache = False

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
