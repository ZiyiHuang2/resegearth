import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import transformers
from transformers import SiglipImageProcessor
from peft import LoraConfig, get_peft_model
import warnings
import copy
from deepspeed.profiling.flops_profiler import get_model_profile
import json
from segearth_r2.datasets.dataset import *
from segearth_r2.train.llava_trainer import LLaVATrainer
from segearth_r2.model.language_model.llava_qwen import SegEarthR2Qwen as SegEarthR2
from segearth_r2.utils.constants import IGNORE_INDEX
from segearth_r2.utils import conversation as conversation_lib
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
    mm_projector_type: Optional[str] = field(default="linear")
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
    base_data_path: str = '/home/wangchengjun/huangziyi/data/RRSISD'
    data_ratio: str = '1'  
    switch_bs: int = 4 # 16
    fix_dataset_len: int = 0
    segmentation: bool = True
    mask_style: str = "legacy"
    dataset_name: str = "rrsisd"

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

def ensure_single_special_token(tokenizer, model, token_text: str) -> int:
    """
    Ensure token_text is registered as a single tokenizer token and model embeddings
    are resized when needed.
    """
    before_vocab_size = len(tokenizer)

    token_id = tokenizer.convert_tokens_to_ids(token_text)
    encoded = tokenizer.encode(token_text, add_special_tokens=False)

    already_single = (
        token_id is not None
        and isinstance(token_id, int)
        and len(encoded) == 1
        and encoded[0] == token_id
    )

    if not already_single:
        num_new_tokens = tokenizer.add_special_tokens(
            {"additional_special_tokens": [token_text]}
        )
        if num_new_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))

    token_id = tokenizer.convert_tokens_to_ids(token_text)
    encoded = tokenizer.encode(token_text, add_special_tokens=False)

    if token_id is None or len(encoded) != 1 or encoded[0] != token_id:
        raise ValueError(
            f"Failed to register {token_text} as a single tokenizer token. "
            f"token_id={token_id}, encoded={encoded}, "
            f"vocab_before={before_vocab_size}, vocab_after={len(tokenizer)}"
        )

    print(f"[tokenizer] ensured special token {token_text} -> id={token_id}")
    return token_id
def make_unify_datamodule(clip_image_processor, tokenizer, data_args, training_args):
    data_ratio = data_args.data_ratio
    data_ratio = data_ratio.split('||')
    data_ratio = [int(data_) for data_ in data_ratio]
    datasets = []

    dataset_name = data_args.dataset_name.lower()

    if data_ratio[0] != 0:
        if dataset_name == "rrsisd":
            train_dataset_single = RRSISDDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split='train'
            )
            eval_dataset = RRSISDDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split='val'
            )
        elif dataset_name == "lasers":
            train_dataset_single = LaSeRSDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split='train_data.json'
            )
            eval_dataset = LaSeRSDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split='val_data.json'
            )

        elif dataset_name == "refsegrs":
            train_dataset_single = RefSegRSDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split='train'
            )
            eval_dataset = RefSegRSDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split='val'
            )
        elif dataset_name == "risbench":
            train_dataset_single = RISBenchDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split='train'
            )
            eval_dataset = RISBenchDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split='val'
            )
        else:
            raise ValueError(
                f"Unsupported dataset_name={data_args.dataset_name}. "
                f"Expected one of: rrsisd, lasers, refsegrs, risbench"
            )

        datasets += [train_dataset_single] * data_ratio[0]
    else:
        raise ValueError("data_ratio[0] is 0; train dataset is empty.")

    print(f'the dataset ratio is: {data_ratio}')
    print(f'the dataset name is: {dataset_name}')
    train_dataset = UnifyDatasetSingleDatasetForBatch(
        datasets, data_ratio, data_args.switch_bs, fix_dataset_len=data_args.fix_dataset_len
    )
    print(f'total unify datasest number is {len(train_dataset)}')
    data_collator = DataCollatorForCOCODatasetV2(
        tokenizer=tokenizer,
        clip_image_processor=clip_image_processor
    )
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def _non_ignore_spans(labels, ignore_index=IGNORE_INDEX):
    spans = []
    start = None
    for idx, value in enumerate(labels):
        if value != ignore_index and start is None:
            start = idx
        elif value == ignore_index and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(labels) - 1))
    return spans


def _format_prompt_for_dump(prompt_text, keep_chars=500):
    if len(prompt_text) <= keep_chars * 2:
        return prompt_text
    return prompt_text[:keep_chars] + "\n...[TRUNCATED]...\n" + prompt_text[-keep_chars:]


def _safe_decode_with_placeholders(tokenizer, input_ids, seg_token_id=None):
    parts = []
    valid_ids = []
    vocab_size = getattr(tokenizer, "vocab_size", None)

    def flush_valid():
        nonlocal valid_ids
        if not valid_ids:
            return
        try:
            parts.append(tokenizer.decode(valid_ids, skip_special_tokens=False))
        except Exception:
            parts.append(str(valid_ids))
        valid_ids = []

    for tid in input_ids:
        tid_int = int(tid)
        if tid_int == IMAGE_TOKEN_INDEX:
            flush_valid()
            parts.append("<image>")
        elif tid_int == REFER_TOKEN_INDEX:
            flush_valid()
            parts.append("<refer>")
        elif vocab_size is not None and (tid_int < 0 or tid_int >= vocab_size):
            flush_valid()
            parts.append(f"<tok:{tid_int}>")
        elif seg_token_id is not None and tid_int == seg_token_id:
            flush_valid()
            parts.append("[SEG]")
        else:
            valid_ids.append(tid_int)

    flush_valid()
    return "".join(parts)


def dump_prompt_diagnostics_step0(
    train_dataset,
    tokenizer,
    output_dir,
    version_name,
    mask_style,
    mm_projector_trainable,
    max_samples=10,
):
    diagnostics_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    dump_path = os.path.join(diagnostics_dir, "prompt_dump_step0.jsonl")
    sample_cnt = min(max_samples, len(train_dataset))
    seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")

    with open(dump_path, "w", encoding="utf-8") as fw:
        for idx in range(sample_cnt):
            sample = train_dataset[idx]
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            if torch.is_tensor(input_ids):
                input_ids = input_ids.tolist()
            if torch.is_tensor(labels):
                labels = labels.tolist()

            prompt_text = _safe_decode_with_placeholders(tokenizer, input_ids, seg_token_id=seg_token_id)
            spans = _non_ignore_spans(labels)
            ignore_cnt = sum(1 for x in labels if x == IGNORE_INDEX)
            total_cnt = len(labels)
            ignore_ratio = (ignore_cnt / total_cnt) if total_cnt > 0 else 1.0
            valid_ratio = 1.0 - ignore_ratio if total_cnt > 0 else 0.0
            valid_label_token_count = total_cnt - ignore_cnt

            sample_id = idx
            if "annotations" in sample and len(sample["annotations"]) > 0:
                sample_id = sample["annotations"][0].get("data_id", idx)

            key_tokens = {
                "has_image_token_id": IMAGE_TOKEN_INDEX in input_ids,
                "has_refer_token_id": REFER_TOKEN_INDEX in input_ids,
                "has_seg_token_id": (seg_token_id in input_ids) if seg_token_id is not None and seg_token_id >= 0 else False,
                "has_image_text": "<image>" in prompt_text,
                "has_refer_text": "<refer>" in prompt_text,
                "has_seg_text": ("[SEG]" in prompt_text) or ("<seg>" in prompt_text.lower()),
            }

            record = {
                "sample_id": sample_id,
                "index": idx,
                "version": version_name,
                "mask_style": mask_style,
                "mm_projector_trainable": bool(mm_projector_trainable),
                "prompt_text": _format_prompt_for_dump(prompt_text, keep_chars=500),
                "tokenized_input_ids_len": total_cnt,
                "label_valid_spans_non_neg100": spans,
                "valid_label_token_count": valid_label_token_count,
                "valid_label_token_ratio": round(valid_ratio, 6),
                "key_token_presence": key_tokens,
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[rank0] prompt diagnostics dumped to: {dump_path}")


def is_mm_projector_trainable(model):
    mm_projector = model.get_model().mm_projector
    return any(p.requires_grad for p in mm_projector.parameters())

def ensure_datasets_dataset_attr(local_rank=None):
    try:
        import datasets as hf_datasets
    except Exception:
        return

    if not hasattr(hf_datasets, "Dataset"):
        class _CompatDataset:
            pass
        hf_datasets.Dataset = _CompatDataset
        if local_rank in [None, -1, 0]:
            print("[compat] injected datasets.Dataset placeholder to avoid Trainer AttributeError")
def train():
    global local_rank

    import inspect
    import pathlib

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.seed is None:
        training_args.seed = 42
    if training_args.data_seed is None:
        training_args.data_seed = 42
    local_rank = training_args.local_rank
    transformers.set_seed(training_args.seed)
    if training_args.local_rank in (-1, 0):
        print(f"[Seed] Set global seed before model init: {training_args.seed}")

    compute_dtype = (
        torch.float16 if training_args.fp16 else
        (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

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
        conversation_lib.default_conversation = (
            conversation_lib.conv_templates.get("vicuna_v1")
            or conversation_lib.conv_templates.get("llava_v1")
            or conversation_lib.conv_templates["default"]
        )

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower_mask = model.model.get_vision_tower_mask()

        current_device = torch.device(f"cuda:{training_args.local_rank}") if torch.cuda.is_available() and training_args.local_rank != -1 else training_args.device
        vision_tower.to(dtype=compute_dtype, device=current_device)
        vision_tower_mask.to(dtype=compute_dtype, device=current_device)

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
        else:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

    seg_token_id = ensure_single_special_token(tokenizer, model, "[SEG]")
    print("SEG token id after tokenizer add:", tokenizer.convert_tokens_to_ids("[SEG]"))
    print("SEG encode after tokenizer add:", tokenizer.encode("[SEG]", add_special_tokens=False))

    train_module_list = [
        "lm_head", "pixel_decoder", "predictor", "SEG_token_projector",
    ]
    if not training_args.freeze_mm_mlp_adapter:
        train_module_list.append("mm_projector")
    if model_args.train_swin_backbone:
        train_module_list.append("vision_tower_mask")

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
            if any(x in n for x in train_module_list):
                p.requires_grad = True

    model.get_special_token(
        SEG=torch.tensor([[seg_token_id]], dtype=torch.long),
        EOS=tokenizer.eos_token_id
    )

    clip_image_processor = SiglipImageProcessor.from_pretrained(model_args.vision_tower)

    data_module = make_unify_datamodule(
        clip_image_processor=clip_image_processor,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args
    )
    mm_projector_trainable = is_mm_projector_trainable(model)
    if training_args.local_rank in [-1, 0]:
        print(f"[rank0] freeze_mm_mlp_adapter={training_args.freeze_mm_mlp_adapter}")
        print(f"[rank0] mm_projector_trainable={mm_projector_trainable}")
        dump_prompt_diagnostics_step0(
            train_dataset=data_module["train_dataset"],
            tokenizer=tokenizer,
            output_dir=training_args.output_dir,
            version_name=model_args.version,
            mask_style=data_args.mask_style,
            mm_projector_trainable=mm_projector_trainable,
            max_samples=10,
        )
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

    current_device = torch.device(f"cuda:{training_args.local_rank}") if torch.cuda.is_available() and training_args.local_rank != -1 else training_args.device
    model.to(current_device)
    if torch.cuda.is_available():
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        torch.cuda.empty_cache()
    # --------- 关键兼容点：不同 transformers 版本对 Trainer 是否支持 tokenizer= 不一致 ---------
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        **data_module
    )
    if "tokenizer" in inspect.signature(LLaVATrainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = LLaVATrainer(**trainer_kwargs)
    # 统一挂上 tokenizer，避免后续代码/保存流程用到 trainer.tokenizer
    trainer.tokenizer = tokenizer
    # ------------------------------------------------------------------------------
    ensure_datasets_dataset_attr(training_args.local_rank)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir
    )

if __name__ == "__main__":
    train()