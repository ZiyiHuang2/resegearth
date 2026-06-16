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
import json
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

    def update_history_loss_dict(self,outputs):
        if not hasattr(self,'history_loss_dict'):
            self.history_loss_dict = {}
        for name, value in outputs.items():
            if 'loss' in name and name != 'loss':
                if name not in self.history_loss_dict:
                    self.history_loss_dict[name] = value.item()
                else:
                    if value != 0:
                        self.history_loss_dict[name] = value.item()

    def _maybe_export_seg_loc_prior(self, outputs):
        """Append prior batch metrics and/or per-sample rows to jsonl under output_dir (rank 0)."""
        if not self.is_world_process_zero():
            return
        if not (
            getattr(self.args, "export_seg_loc_prior_metrics", False)
            or getattr(self.args, "export_seg_loc_prior_samples", False)
        ):
            return
        out_dir = getattr(self.args, "output_dir", None) or None
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        step = int(self.state.global_step)

        if getattr(self.args, "export_seg_loc_prior_metrics", False):
            ext = getattr(outputs, "seg_loc_prior_extended_metrics", None) or {}
            row = {"global_step": step}
            for k, v in ext.items():
                fv = self._to_float_metric(v)
                if fv is not None:
                    row[k] = fv
            for k in (
                "loss_seg_loc_prior",
                "seg_loc_prior_mean",
                "seg_loc_prior_entropy",
                "seg_loc_prior_fg_mass",
                "seg_loc_prior_bg_mass",
            ):
                fv = self._to_float_metric(getattr(outputs, k, None))
                if fv is not None:
                    row[k] = fv
            path = os.path.join(out_dir, "seg_loc_prior_metrics.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if getattr(self.args, "export_seg_loc_prior_samples", False):
            rows = getattr(outputs, "seg_loc_prior_sample_rows", None)
            if rows:
                path = os.path.join(out_dir, "seg_loc_prior_samples.jsonl")
                with open(path, "a", encoding="utf-8") as f:
                    for r in rows:
                        rr = dict(r)
                        rr["global_step"] = step
                        f.write(json.dumps(rr, ensure_ascii=False) + "\n")

    @staticmethod
    def _to_float_metric(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().item())
        if isinstance(value, (float, int)):
            return float(value)
        return None

    @staticmethod
    def _resolve_subset_name(gt_item):
        subset = gt_item.get("subset", None)
        if subset is None:
            subset = gt_item.get("subset_name", None)
        if subset is None:
            subset = gt_item.get("eval_subset", None)
        if subset is None:
            return None
        subset = str(subset).strip().upper()
        if subset.startswith("B"):
            return "B"
        if subset.startswith("R"):
            return "R"
        return None

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
        iter_start = time.perf_counter()
        data_time = 0.0
        if hasattr(self, "_last_iter_end_time"):
            data_time = max(0.0, iter_start - self._last_iter_end_time)
        forward_start = time.perf_counter()
        outputs = model(**inputs)
        forward_time = max(0.0, time.perf_counter() - forward_start)
        iter_time = max(0.0, time.perf_counter() - iter_start)
        batch_size = None
        if isinstance(inputs, dict) and "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
            batch_size = int(inputs["input_ids"].shape[0])
        throughput = float(batch_size / iter_time) if batch_size is not None and iter_time > 0 else 0.0
        self._last_iter_end_time = time.perf_counter()

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, Mapping) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, Mapping) else outputs[0]
            if isinstance(outputs, Mapping):
                metric_dict = {
                    "loss": self._to_float_metric(loss),
                    "data_time": data_time,
                    "iter_time": iter_time,
                    "forward_time": forward_time,
                    "throughput": throughput,
                }
                for name, value in outputs.items():
                    if 'loss' in name and name != 'loss':
                        loss_value = self._to_float_metric(value)
                        if loss_value is None:
                            continue
                        if loss_value == 0 and hasattr(self,'history_loss_dict') and name in self.history_loss_dict:
                            loss_value = self.history_loss_dict[name]
                        metric_dict[name] = loss_value
                for key in [
                    "precomputed_structured_count",
                    "fallback_structured_count",
                    "missing_precomputed_count",
                    "selected_attn_layers",
                    "selected_attn_heads_total",
                    "selected_attn_heads_per_layer",
                    "structured_effective_weight",
                    "structured_schedule_phase_id",
                    "structured_small_count",
                    "structured_non_small_count",
                    "structured_skipped_batches",
                    "structured_skip_ratio",
                    "structured_skip_reason_code",
                    "structured_loss_small",
                    "structured_loss_non_small",
                    "loss_attn_fg_bg_small",
                    "loss_attn_fg_bg_non_small",
                    "loss_attn_boundary_outer_small",
                    "loss_attn_boundary_outer_non_small",
                    "structured_output_effect_mean",
                    "structured_output_effect_nonzero_ratio",
                    "structured_output_effect_gt_threshold_ratio",
                    "structured_output_effect_mean_small",
                    "structured_output_effect_mean_non_small",
                    "structured_output_effect_nonzero_ratio_small",
                    "structured_output_effect_nonzero_ratio_non_small",
                    "diag_mask_logits_mean_abs_diff",
                    "diag_binary_mask_disagree_ratio",
                    "diag_pred_mask_iou_between_runs",
                    "diag_pred_area_change_ratio",
                    "diag_iou_with_gt_run_a",
                    "diag_iou_with_gt_run_b",
                    "diag_delta_iou_vs_gt",
                    "diag_attention_sup_default_minus_unstructured",
                    "diag_structured_supervision_alters_mask_forward_path",
                    "seg_query_delta_norm",
                    "seg_query_original_norm",
                    "seg_query_refined_norm",
                    "seg_query_cosine_q_delta",
                    "seg_query_cosine_q_qnew",
                    "loss_seg_loc_prior",
                    "seg_loc_prior_mean",
                    "seg_loc_prior_entropy",
                    "seg_loc_prior_fg_mass",
                    "seg_loc_prior_bg_mass",
                    "seg_loc_code_norm",
                    "seg_loc_sem_norm",
                    "seg_loc_qfinal_norm",
                    "seg_loc_code_to_sem_norm_ratio",
                    "seg_loc_cosine_sem_qfinal",
                    "seg_loc_cosine_sem_loccode",
                ]:
                    if key in outputs:
                        scalar = self._to_float_metric(outputs[key])
                        if scalar is not None:
                            metric_dict[key] = scalar
                ext_map = getattr(outputs, "seg_loc_prior_extended_metrics", None)
                if ext_map:
                    for kn, val in ext_map.items():
                        s = self._to_float_metric(val)
                        if s is not None:
                            metric_dict[kn] = s
                self.update_history_loss_dict(outputs)
                log_payload = {k: v for k, v in metric_dict.items() if v is not None}
                if getattr(self.args, "diagnose_seg_loc_prior_only", False):
                    keep = {
                        "loss",
                        "data_time",
                        "iter_time",
                        "forward_time",
                        "throughput",
                        "loss_seg_loc_prior",
                    }
                    log_payload = {
                        k: v
                        for k, v in log_payload.items()
                        if k in keep or str(k).startswith("seg_loc_prior")
                    }
                self.log(log_payload)

        self._maybe_export_seg_loc_prior(outputs)
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

    _PR_IOU_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)

    @staticmethod
    def _mask_sample_metrics(pred_mask_np, gt_mask_np, eps=1e-7):
        """Per-sample mask metrics (aligned with eval_val_metrics.py)."""
        inter = float(np.logical_and(pred_mask_np > 0, gt_mask_np > 0).sum())
        union = float(np.logical_or(pred_mask_np > 0, gt_mask_np > 0).sum())
        pred_area = float(pred_mask_np.sum())
        gt_area = float(gt_mask_np.sum())
        iou = inter / (union + eps)
        dice = (2.0 * inter) / (pred_area + gt_area + eps)
        recall = inter / (gt_area + eps)
        precision = inter / (pred_area + eps)
        return iou, dice, recall, precision

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
        dice_sum = 0.0
        recall_sum = 0.0
        precision_sum = 0.0
        pr_counts = {thr: 0 for thr in self._PR_IOU_THRESHOLDS}
        valid_count = 0
        eps = 1e-7

        for inputs in eval_dataloader:
            with torch.no_grad():
                inputs = self._prepare_inputs(inputs)
                token_refer_id = [ids.to(self.args.device) for ids in inputs["token_refer_id"]]
                outputs = model.eval_seg(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    images=inputs["images"].to(dtype=model_dtype),
                    images_clip=inputs["images_clip"].to(dtype=model_dtype),
                    seg_info=inputs["seg_info"],
                    token_refer_id=token_refer_id,
                    SEG_token_embedding_indices=inputs["SEG_token_embedding_indices"],
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
                if gt_mask_np.sum() == 0:
                    continue

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

                iou, dice, recall, precision = self._mask_sample_metrics(
                    pred_mask_np, gt_mask_np, eps=eps
                )
                inter = float(np.logical_and(pred_mask_np > 0, gt_mask_np > 0).sum())
                union = float(np.logical_or(pred_mask_np > 0, gt_mask_np > 0).sum())

                iou_sum += iou
                dice_sum += dice
                recall_sum += recall
                precision_sum += precision
                total_inter += inter
                total_union += union
                valid_count += 1
                for thr in self._PR_IOU_THRESHOLDS:
                    if iou >= thr:
                        pr_counts[thr] += 1

        stats_list = [
            iou_sum,
            total_inter,
            total_union,
            float(valid_count),
            dice_sum,
            recall_sum,
            precision_sum,
        ] + [float(pr_counts[thr]) for thr in self._PR_IOU_THRESHOLDS]

        if dist.is_available() and dist.is_initialized():
            stats = torch.tensor(stats_list, device=self.args.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            stats_list = stats.tolist()

        (
            iou_sum,
            total_inter,
            total_union,
            valid_count,
            dice_sum,
            recall_sum,
            precision_sum,
        ) = stats_list[:7]
        pr_count_values = stats_list[7:]
        valid_count = int(valid_count)

        if valid_count > 0:
            eval_giou = iou_sum / valid_count
            eval_ciou = total_inter / (total_union + eps)
            eval_mdice = dice_sum / valid_count
            eval_mrecall = recall_sum / valid_count
            eval_mprecision = precision_sum / valid_count
        else:
            eval_giou = 0.0
            eval_ciou = 0.0
            eval_mdice = 0.0
            eval_mrecall = 0.0
            eval_mprecision = 0.0
        eval_pr_at_0_9 = float(pr_count_values[-1] / valid_count) if valid_count > 0 else 0.0
        eval_score = 0.45 * eval_giou + 0.35 * eval_ciou + 0.20 * eval_pr_at_0_9

        metrics = {
            f"{metric_key_prefix}_giou": float(eval_giou),
            f"{metric_key_prefix}_ciou": float(eval_ciou),
            f"{metric_key_prefix}_score": float(eval_score),
            f"{metric_key_prefix}_mdice": float(eval_mdice),
            f"{metric_key_prefix}_mrecall": float(eval_mrecall),
            f"{metric_key_prefix}_mprecision": float(eval_mprecision),
        }
        for thr, pr_count in zip(self._PR_IOU_THRESHOLDS, pr_count_values):
            thr_key = str(thr).replace(".", "_")
            pr_at = float(pr_count / valid_count) if valid_count > 0 else 0.0
            metrics[f"{metric_key_prefix}_pr_at_{thr_key}"] = pr_at
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics
