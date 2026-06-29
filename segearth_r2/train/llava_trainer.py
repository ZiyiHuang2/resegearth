import os
import torch
import numpy as np
import cv2
import shutil
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch.distributed as dist
from typing import Optional
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available, is_torch_tpu_available,is_accelerate_available
if is_apex_available():
    from apex import amp
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from fvcore.nn import FlopCountAnalysis, parameter_count
from deepspeed.profiling.flops_profiler import get_model_profile

import torch

from packaging import version
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class LLaVATrainer(Trainer):

    LOSS_LOG_KEYS = (
        "loss_llm",
        "loss_mask",
        "loss_dice",
        "loss_union",
        "loss_setpp_coverage",
        "loss_setpp_consistency",
        "loss_attention",
    )

    def _iter_output_items(self, outputs):
        if isinstance(outputs, dict):
            return outputs.items()
        if hasattr(outputs, "keys"):
            return ((k, outputs[k]) for k in outputs.keys())
        return ()

    def _loss_value(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().float().item())
        if isinstance(value, (float, int)):
            return float(value)
        return None

    def _extract_loss_dict(self, outputs):
        loss_dict = {}
        for name, value in self._iter_output_items(outputs):
            if "loss" not in name or name == "loss":
                continue
            loss_value = self._loss_value(value)
            if loss_value is None:
                continue
            if loss_value == 0 and hasattr(self, "history_loss_dict") and name in self.history_loss_dict:
                loss_value = self.history_loss_dict[name]
            loss_dict[name] = loss_value
        return loss_dict

    def _log_losses_to_output(self, loss_dict):
        if not loss_dict:
            return
        if self.args.local_rank not in (-1, 0):
            return
        ordered = []
        for key in self.LOSS_LOG_KEYS:
            if key in loss_dict:
                ordered.append(f"{key}={loss_dict[key]:.6f}")
        for key in sorted(loss_dict.keys()):
            if key not in self.LOSS_LOG_KEYS:
                ordered.append(f"{key}={loss_dict[key]:.6f}")
        logger.info("[step %d] losses: %s", self.state.global_step, " | ".join(ordered))

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def update_history_loss_dict(self, outputs):
        if not hasattr(self, 'history_loss_dict'):
            self.history_loss_dict = {}
        for name, value in self._iter_output_items(outputs):
            if 'loss' in name and name != 'loss':
                loss_value = self._loss_value(value)
                if loss_value is None:
                    continue
                if name not in self.history_loss_dict:
                    self.history_loss_dict[name] = loss_value
                elif loss_value != 0:
                    self.history_loss_dict[name] = loss_value


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
                How the loss is computed by Trainer. By default, all models return the loss in the first element.

                Subclass and override for custom behavior.
                """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        global_step = self.state.global_step
        inputs['global_step'] = global_step
        
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            elif isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs[0]
            loss_dict = self._extract_loss_dict(outputs)
            if loss_dict:
                self.update_history_loss_dict(outputs)
                self.log(loss_dict)
                if self.state.global_step > 0 and (self.state.global_step + 1) % max(1, self.args.logging_steps) == 0:
                    self._log_losses_to_output(loss_dict)

        return (loss, outputs) if return_outputs else loss

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        # Keep train drop_last behavior unchanged, but force eval to keep tail batches.
        original_drop_last = self.args.dataloader_drop_last
        self.args.dataloader_drop_last = False
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.args.dataloader_drop_last = original_drop_last

    @staticmethod
    def _bbox_from_mask(mask_np):
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    @staticmethod
    def _bbox_area(box):
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1 + 1) * max(0.0, y2 - y1 + 1)

    @classmethod
    def _bbox_iou(cls, box1, box2, eps=1e-7):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = 0.0
        if x2 >= x1 and y2 >= y1:
            inter = (x2 - x1 + 1) * (y2 - y1 + 1)
        union = cls._bbox_area(box1) + cls._bbox_area(box2) - inter
        return float(inter / (union + eps))

    @classmethod
    def _box_giou_ciou(cls, box1, box2, eps=1e-7):
        iou = cls._bbox_iou(box1, box2, eps=eps)

        cx1 = min(box1[0], box2[0])
        cy1 = min(box1[1], box2[1])
        cx2 = max(box1[2], box2[2])
        cy2 = max(box1[3], box2[3])
        c_area = cls._bbox_area([cx1, cy1, cx2, cy2])

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = 0.0
        if x2 >= x1 and y2 >= y1:
            inter = (x2 - x1 + 1) * (y2 - y1 + 1)
        union = cls._bbox_area(box1) + cls._bbox_area(box2) - inter
        giou = float(iou - (c_area - union) / (c_area + eps))

        b1x = (box1[0] + box1[2]) / 2.0
        b1y = (box1[1] + box1[3]) / 2.0
        b2x = (box2[0] + box2[2]) / 2.0
        b2y = (box2[1] + box2[3]) / 2.0
        center_dist_sq = (b1x - b2x) ** 2 + (b1y - b2y) ** 2

        cw = cx2 - cx1 + 1.0
        ch = cy2 - cy1 + 1.0
        c_diag_sq = cw ** 2 + ch ** 2 + eps

        w1 = box1[2] - box1[0] + 1.0
        h1 = box1[3] - box1[1] + 1.0
        w2 = box2[2] - box2[0] + 1.0
        h2 = box2[3] - box2[1] + 1.0
        v = (4.0 / (math.pi ** 2)) * (math.atan(w1 / h1) - math.atan(w2 / h2)) ** 2
        alpha = v / (1.0 - iou + v + eps)
        ciou = float(iou - (center_dist_sq / c_diag_sq) - alpha * v)
        return giou, ciou

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()

        model = self._wrap_model(self.model, training=False, dataloader=None)
        model.eval()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model_dtype = next(model.parameters()).dtype

        iou_sum = 0.0
        total_inter = 0.0
        total_union = 0.0
        valid_count = 0
        eps = 1e-7

        # Diagnostic accumulators
        union_iou_sum = 0.0
        union_inter_sum = 0.0
        union_union_sum = 0.0
        union_valid = 0
        instance_iou_k1_sum = 0.0
        instance_iou_k1_count = 0
        instance_iou_kgt1_sum = 0.0
        instance_iou_kgt1_count = 0

        for inputs in eval_dataloader:
            with torch.no_grad():
                inputs = self._prepare_inputs(inputs)
                token_refer_id = [ids.to(self.args.device) for ids in inputs["token_refer_id"]]
                outputs, union_mask_info = model.eval_seg(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    images=inputs["images"].to(dtype=model_dtype),
                    images_clip=inputs["images_clip"].to(dtype=model_dtype),
                    seg_info=inputs["seg_info"],
                    token_refer_id=token_refer_id,
                    SEG_token_embedding_indices=inputs["SEG_token_embedding_indices"],
                    SET_token_embedding_indices=inputs.get("SET_token_embedding_indices", None),
                    labels=inputs["labels"],
                    mask_num=inputs["mask_num"],
                )

            gt_by_key = {}
            for gt_item in inputs["seg_info"]:
                gt_key = (
                    str(gt_item.get("image_id")),
                    str(gt_item.get("data_id")),
                    str(gt_item.get("mask_id")),
                )
                gt_by_key[gt_key] = gt_item

            # Build per-image K mapping from mask_num for instance stratification
            mask_num_list = inputs["mask_num"]
            k_by_image = {}
            offset = 0
            for b_idx, k_val in enumerate(mask_num_list):
                if k_val > 0:
                    first_item = inputs["seg_info"][offset]
                    img_key = (str(first_item.get("image_id")), str(first_item.get("data_id")))
                    k_by_image[img_key] = k_val
                offset += k_val

            # --- Diagnostic: union mask IoU ---
            union_preds = union_mask_info.get('union_preds', [])
            union_image_ids = union_mask_info.get('image_ids', [])
            union_data_ids = union_mask_info.get('data_ids', [])
            # Build per-image GT union (OR of all GT masks for that image)
            for b_idx in range(len(union_preds)):
                img_id = union_image_ids[b_idx]
                data_id = union_data_ids[b_idx]
                # Collect all GT masks for this (image_id, data_id) pair
                gt_masks_for_image = []
                for key, gt_item in gt_by_key.items():
                    if str(gt_item.get('image_id')) == str(img_id) and str(gt_item.get('data_id')) == str(data_id):
                        gt_mask = gt_item.get('mask')
                        if gt_mask is not None:
                            if torch.is_tensor(gt_mask):
                                gt_mask_np = gt_mask.detach().cpu().numpy()
                            else:
                                gt_mask_np = np.asarray(gt_mask)
                            if gt_mask_np.ndim > 2:
                                gt_mask_np = np.squeeze(gt_mask_np)
                            gt_mask_np = (gt_mask_np > 0).astype(np.uint8)
                            gt_masks_for_image.append(gt_mask_np)
                if len(gt_masks_for_image) > 0:
                    # OR all instance masks to get union GT
                    union_gt = np.zeros_like(gt_masks_for_image[0], dtype=np.uint8)
                    for m in gt_masks_for_image:
                        union_gt = np.logical_or(union_gt, m > 0).astype(np.uint8)
                    union_pred = union_preds[b_idx]
                    if union_pred.shape != union_gt.shape:
                        union_pred = cv2.resize(union_pred, (union_gt.shape[1], union_gt.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                        union_pred = (union_pred > 0).astype(np.uint8)
                    inter = float(np.logical_and(union_pred > 0, union_gt > 0).sum())
                    uni = float(np.logical_or(union_pred > 0, union_gt > 0).sum())
                    union_iou_sum += inter / (uni + eps)
                    union_inter_sum += inter
                    union_union_sum += uni
                    union_valid += 1

            for pred_item in outputs:
                pred_key = (
                    str(pred_item.get("image_name")),
                    str(pred_item.get("id")),
                    str(pred_item.get("mask_id")),
                )
                gt_item = gt_by_key.get(pred_key)
                if gt_item is None:
                    continue

                gt_mask = gt_item.get("mask")
                if gt_mask is None:
                    continue
                if torch.is_tensor(gt_mask):
                    gt_mask_np = gt_mask.detach().cpu().numpy()
                else:
                    gt_mask_np = np.asarray(gt_mask)
                if gt_mask_np.ndim > 2:
                    gt_mask_np = np.squeeze(gt_mask_np)
                gt_mask_np = (gt_mask_np > 0).astype(np.uint8)

                pred_mask_np = np.asarray(pred_item.get("pred"))
                if pred_mask_np.ndim > 2:
                    pred_mask_np = np.squeeze(pred_mask_np)
                pred_mask_np = (pred_mask_np > 0).astype(np.uint8)

                if pred_mask_np.shape != gt_mask_np.shape:
                    pred_mask_np = cv2.resize(
                        pred_mask_np,
                        (gt_mask_np.shape[1], gt_mask_np.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    pred_mask_np = (pred_mask_np > 0).astype(np.uint8)

                inter = float(np.logical_and(pred_mask_np > 0, gt_mask_np > 0).sum())
                union = float(np.logical_or(pred_mask_np > 0, gt_mask_np > 0).sum())
                iou = inter / (union + eps)

                iou_sum += iou
                total_inter += inter
                total_union += union
                valid_count += 1

                # Stratify by K
                img_key = (str(gt_item.get('image_id')), str(gt_item.get('data_id')))
                k = k_by_image.get(img_key, 1)
                if k == 1:
                    instance_iou_k1_sum += iou
                    instance_iou_k1_count += 1
                else:
                    instance_iou_kgt1_sum += iou
                    instance_iou_kgt1_count += 1

        if dist.is_available() and dist.is_initialized():
            stats = torch.tensor([iou_sum, total_inter, total_union, float(valid_count),
                                   union_iou_sum, union_inter_sum, union_union_sum, float(union_valid),
                                   instance_iou_k1_sum, float(instance_iou_k1_count),
                                   instance_iou_kgt1_sum, float(instance_iou_kgt1_count)],
                                  device=self.args.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            iou_sum = float(stats[0].item())
            total_inter = float(stats[1].item())
            total_union = float(stats[2].item())
            valid_count = int(stats[3].item())
            union_iou_sum = float(stats[4].item())
            union_inter_sum = float(stats[5].item())
            union_union_sum = float(stats[6].item())
            union_valid = int(stats[7].item())
            instance_iou_k1_sum = float(stats[8].item())
            instance_iou_k1_count = int(stats[9].item())
            instance_iou_kgt1_sum = float(stats[10].item())
            instance_iou_kgt1_count = int(stats[11].item())

        if valid_count > 0:
            eval_giou = iou_sum / valid_count
            eval_ciou = total_inter / (total_union + eps)
        else:
            eval_giou = 0.0
            eval_ciou = 0.0
        eval_score = 0.5 * eval_giou + 0.5 * eval_ciou

        metrics = {
            f"{metric_key_prefix}_giou": float(eval_giou),
            f"{metric_key_prefix}_ciou": float(eval_ciou),
            f"{metric_key_prefix}_score": float(eval_score),
        }
        # Diagnostic metrics
        if union_valid > 0:
            metrics[f"{metric_key_prefix}_union_giou"] = float(union_iou_sum / union_valid)
            metrics[f"{metric_key_prefix}_union_ciou"] = float(union_inter_sum / (union_union_sum + eps))
        if instance_iou_k1_count > 0:
            metrics[f"{metric_key_prefix}_instance_giou_K1"] = float(instance_iou_k1_sum / instance_iou_k1_count)
        if instance_iou_kgt1_count > 0:
            metrics[f"{metric_key_prefix}_instance_giou_Kgt1"] = float(instance_iou_kgt1_sum / instance_iou_kgt1_count)
        metrics[f"{metric_key_prefix}_union_valid"] = float(union_valid)
        metrics[f"{metric_key_prefix}_instance_K1_count"] = float(instance_iou_k1_count)
        metrics[f"{metric_key_prefix}_instance_Kgt1_count"] = float(instance_iou_kgt1_count)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics