import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import torch
import torch.distributed as distributed
import numpy as np
import zipfile
from tifffile import imwrite as imsave
from tqdm import tqdm
from transformers import SiglipImageProcessor

from segearth_r2.utils import conversation as conversation_lib
from segearth_r2.utils.builder import load_pretrained_model
from segearth_r2.datasets.dataset import (
    DataCollatorForCOCODatasetV2,
    LaSeRSDataset,
    RefSegRSDataset,
    RISBenchDataset,
    RRSISDDataset,
)

from dataclasses import dataclass, field
import transformers
from typing import Optional


@dataclass
class DataArguments:
    local_rank: int = 0

    vision_tower: str = "pretrained_model/CLIP/siglip-so400m-patch14-384"
    vision_tower_mask: str = "pretrained_model/mask2former/model_final_54b88a.pkl"

    lazy_preprocess: bool = False
    base_data_path: Optional[str] = field(default="your_data_path")
    model_path: Optional[str] = field(default="your_model_path")
    mask_config: Optional[str] = field(
        default="../segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = "segearth_r2"
    version: str = "llava_phi"
    output_dir: str = "save_folder"
    eval_batch_size: int = 1
    dataloader_num_workers: int = 8
    max_eval_samples: int = 0

    load_8bit: bool = False
    load_4bit: bool = False

    # 新增：数据集类型与 split
    dataset_name: str = "lasers"   # "lasers" or "rrsisd" or "refsegrs" or "risbench"
    split: str = "val"             # for rrsisd: train / val / test
    zip_results: bool = True       # 是否自动打包输出目录

    # Text-FiLM：仅影响 eval 前向，不写回 checkpoint；由 model.config.use_text_film 决定是否启用分支
    text_film_eval_mode: str = "normal"  # normal | bypass | force_alpha
    text_film_force_alpha: float = 1.0

    # Decoder cross-attn token bias：eval 临时覆盖，不写回 checkpoint
    decoder_attn_bias_eval_mode: str = "normal"  # normal | bypass | force_scale
    decoder_attn_bias_force_scale: float = 1.0
    allow_random_qdti_init: bool = False


def init_distributed_mode(args):
    # 多卡机器上直接 `python eval.py` 时不会设置 RANK/WORLD_SIZE；若仅按 device_count>1
    # 就 init_process_group，会触发 “RANK expected, but not set”。仅在 torchrun / 显式分布式环境下启用。
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    distributed_job = "RANK" in os.environ and world_size_env > 1

    if torch.cuda.device_count() <= 1 or not distributed_job:
        args.distributed = False
        args.local_rank = 0
        args.world_size = 1
        return

    args.distributed = True
    distributed.init_process_group(backend="nccl")
    local_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    torch.cuda.set_device(local_rank)

    print(f"I am rank {local_rank} in this world of size {world_size}!")
    args.local_rank = local_rank
    args.world_size = world_size


def zip_folder(folder_path):
    folder_path = os.path.abspath(folder_path)
    zip_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname=os.path.basename(file_path))


def build_eval_datasets(data_args, tokenizer):
    dataset_name = data_args.dataset_name.lower()

    if dataset_name == "lasers":
        json_folders = os.path.join(data_args.base_data_path, "val", "annotations")
        if not os.path.isdir(json_folders):
            raise FileNotFoundError(f"LaSeRS val annotation dir not found: {json_folders}")

        splits = sorted(os.listdir(json_folders))
        eval_sets = []

        for split in splits:
            if data_args.local_rank == 0:
                print(f"------ cur benchmark is LaSeRS {split} subset -------")

            eval_dataset = LaSeRSDataset(
                base_data_path=data_args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split=split,
            )
            eval_sets.append((split, eval_dataset))

        return eval_sets

    elif dataset_name == "rrsisd":
        split = data_args.split.lower()
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported RRSISD split: {split}. Must be train / val / test")

        if data_args.local_rank == 0:
            print(f"------ cur benchmark is RRSISD {split} subset -------")

        eval_dataset = RRSISDDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=split,
        )

        # 为输出命名保留统一形式
        split_name = f"{split}.json"
        return [(split_name, eval_dataset)]

    elif dataset_name == "refsegrs":
        split = data_args.split.lower()
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported RefSegRS split: {split}. Must be train / val / test")

        if data_args.local_rank == 0:
            print(f"------ cur benchmark is RefSegRS {split} subset -------")

        eval_dataset = RefSegRSDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=split,
        )

        split_name = f"{split}.json"
        return [(split_name, eval_dataset)]

    elif dataset_name == "risbench":
        split = data_args.split.lower()
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported RISBench split: {split}. Must be train / val / test")

        if data_args.local_rank == 0:
            print(f"------ cur benchmark is RISBench {split} subset -------")

        eval_dataset = RISBenchDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            split=split,
        )

        split_name = f"{split}.json"
        return [(split_name, eval_dataset)]
    else:
        raise ValueError(f"Unsupported dataset_name: {data_args.dataset_name}")


def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    init_distributed_mode(data_args)

    if data_args.local_rank == 0:
        print(f"[Eval] dataset_name={data_args.dataset_name}, split={data_args.split}, base_data_path={data_args.base_data_path}")

    model_path = os.path.expanduser(data_args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_args=data_args,
        mask_config=data_args.mask_config,
        device="cuda",
        load_8bit=data_args.load_8bit,
        load_4bit=data_args.load_4bit,
    )

    orig_tf_mode = getattr(model.config, "text_film_eval_mode", "normal")
    orig_tf_alpha = float(getattr(model.config, "text_film_force_alpha", 1.0))
    orig_dac_mode = getattr(model.config, "decoder_attn_bias_eval_mode", "normal")
    orig_dac_force = float(getattr(model.config, "decoder_attn_bias_force_scale", 1.0))
    try:
        model.config.text_film_eval_mode = getattr(data_args, "text_film_eval_mode", "normal")
        model.config.text_film_force_alpha = float(getattr(data_args, "text_film_force_alpha", 1.0))
        model.config.decoder_attn_bias_eval_mode = getattr(data_args, "decoder_attn_bias_eval_mode", "normal")
        model.config.decoder_attn_bias_force_scale = float(getattr(data_args, "decoder_attn_bias_force_scale", 1.0))

        device = torch.device(data_args.local_rank if torch.cuda.is_available() else "cpu")
        if not data_args.load_8bit and not data_args.load_4bit:
            model.to(dtype=torch.float16, device=device)
        else:
            model.to(device=device)

        data_args.is_multimodal = True
        conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

        clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
        data_collator = DataCollatorForCOCODatasetV2(
            tokenizer=tokenizer,
            clip_image_processor=clip_image_processor,
        )

        save_folder = data_args.output_dir
        os.makedirs(save_folder, exist_ok=True)

        eval_sets = build_eval_datasets(data_args, tokenizer)

        for split, eval_dataset in eval_sets:
            if data_args.local_rank == 0:
                print(f"[Eval] split={split}, dataset_len={len(eval_dataset)}")

            if not data_args.distributed:
                val_sampler = None
            else:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    eval_dataset,
                    shuffle=False,
                    drop_last=False,
                )

            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=data_args.eval_batch_size,
                shuffle=False,
                num_workers=data_args.dataloader_num_workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=data_collator,
            )

            do_eval(model, eval_dataloader, save_folder, split, data_args, device)

        if data_args.local_rank == 0 and data_args.zip_results:
            zip_folder(save_folder)
    finally:
        model.config.text_film_eval_mode = orig_tf_mode
        model.config.text_film_force_alpha = orig_tf_alpha
        model.config.decoder_attn_bias_eval_mode = orig_dac_mode
        model.config.decoder_attn_bias_force_scale = orig_dac_force


def do_eval(model, eval_dataloader, save_folder, split, data_args, device):
    model.eval()
    processed_samples = 0

    if data_args.distributed:
        distributed.barrier()

    infer_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for idx, inputs in tqdm(
            enumerate(eval_dataloader),
            total=len(eval_dataloader),
            disable=(data_args.local_rank != 0),
        ):
            if data_args.max_eval_samples > 0 and processed_samples >= data_args.max_eval_samples:
                break

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs["token_refer_id"] = [ids.to(device) for ids in inputs["token_refer_id"]]

            outputs = model.eval_seg(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=inputs["images"].to(dtype=infer_dtype),
                images_clip=inputs["images_clip"].to(dtype=infer_dtype),
                seg_info=inputs["seg_info"],
                token_refer_id=inputs["token_refer_id"],
                SEG_token_embedding_indices=inputs["SEG_token_embedding_indices"],
                labels=inputs["labels"],
                mask_num=inputs["mask_num"],
            )

            for output in outputs:
                pred_mask = output["pred"]
                image_name = output["image_name"]
                sample_id = output["id"]
                mask_id = output["mask_id"]

                split_stem = split.split(".")[0]
                mask_save_name = f"{image_name}_{sample_id}_{split_stem}_{mask_id}.tif"

                if pred_mask.ndim > 2:
                    pred_mask = np.squeeze(pred_mask)

                imsave(
                    os.path.join(save_folder, mask_save_name),
                    pred_mask.astype(np.uint8),
                )
                processed_samples += 1

    if data_args.distributed:
        distributed.barrier()


if __name__ == "__main__":
    evaluation()
