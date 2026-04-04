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
    # BHFM评测配置（默认None表示沿用模型内配置）
    bhfm_enable: Optional[bool] = field(default=None)
    bhfm_stages: str = field(default="2")
    bhfm_interval: int = field(default=3)
    bhfm_full_open: bool = field(default=False)

    # 新增：数据集类型与 split
    dataset_name: str = "lasers"   # "lasers" or "rrsisd"
    split: str = "val"             # for rrsisd: train / val / test
    zip_results: bool = True       # 是否自动打包输出目录


def init_distributed_mode(args):
    args.distributed = True
    if torch.cuda.device_count() <= 1:
        args.distributed = False
        args.local_rank = 0
        args.world_size = 1
        return

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
    )

    device = torch.device(data_args.local_rank if torch.cuda.is_available() else "cpu")
    model.to(dtype=torch.float32, device=device)
    model.eval_debug_rank0 = (data_args.local_rank == 0)

    # 评测侧可选覆写BHFM开关/策略（尽量不破坏原路径）
    vision_tower_mask = model.get_model().get_vision_tower_mask()
    if data_args.bhfm_enable is not None:
        vision_tower_mask.bhfm_enable = bool(data_args.bhfm_enable)
    if hasattr(vision_tower_mask, "bhfm_stages"):
        vision_tower_mask.bhfm_stages = tuple(
            int(x.strip()) for x in data_args.bhfm_stages.split(",") if x.strip() != ""
        )
    if hasattr(vision_tower_mask, "bhfm_interval"):
        vision_tower_mask.bhfm_interval = int(data_args.bhfm_interval)
    if hasattr(vision_tower_mask, "bhfm_full_open"):
        vision_tower_mask.bhfm_full_open = bool(data_args.bhfm_full_open)

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
            print(
                f"[Eval][BHFM] enable={getattr(vision_tower_mask, 'bhfm_enable', False)}, "
                f"stages={getattr(vision_tower_mask, 'bhfm_stages', ())}, "
                f"interval={getattr(vision_tower_mask, 'bhfm_interval', -1)}, "
                f"full_open={getattr(vision_tower_mask, 'bhfm_full_open', False)}"
            )

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


def do_eval(model, eval_dataloader, save_folder, split, data_args, device):
    model.eval()
    processed_samples = 0

    if data_args.distributed:
        distributed.barrier()

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
                images=inputs["images"].float(),
                images_clip=inputs["images_clip"].float(),
                seg_info=inputs["seg_info"],
                token_refer_id=inputs["token_refer_id"],
                SEG_token_embedding_indices=inputs["SEG_token_embedding_indices"],
                labels=inputs["labels"],
                mask_num=inputs["mask_num"],
            )

            if data_args.local_rank == 0 and idx == 0:
                vt = model.get_model().get_vision_tower_mask()
                print(
                    f"[Eval][BHFM] calls={getattr(vt, 'last_bhfm_calls', 0)}, "
                    f"text_in_shape={getattr(vt, 'last_bhfm_text_in_shape', None)}, "
                    f"text_out_shape={getattr(vt, 'last_bhfm_text_out_shape', None)}"
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
