"""Lightweight W&B metrics for DGP-QDTI training verification."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

DGP_KEYWORDS = (
    "prompt_adapter",
    "query_refiner",
    "query_specific_text_memory_bias",
    "qdti",
    "gate",
    "qdti_scale",
)


def _is_rank0(trainer) -> bool:
    return getattr(trainer.args, "local_rank", 0) in (-1, 0)


def _wandb_active(trainer) -> bool:
    report_to = trainer.args.report_to
    if isinstance(report_to, str):
        report_to = [report_to]
    if "wandb" not in (report_to or []):
        return False
    try:
        import wandb
    except ImportError:
        return False
    return wandb.run is not None


def _scalar_param(param: Optional[torch.nn.Parameter]) -> Optional[float]:
    if param is None:
        return None
    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        if hasattr(param, "ds_id"):
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                return None
            with zero.GatheredParameters([param], modifier_rank=0):
                return float(param.detach().float().cpu().reshape(-1)[0].item())
    except Exception:
        pass
    return float(param.detach().float().cpu().reshape(-1)[0].item())


def _grad_norm_metric(param: Optional[torch.nn.Parameter]) -> float:
    if param is None:
        return -1.0
    grad = param.grad
    if grad is None:
        return -1.0
    try:
        return float(grad.detach().float().norm().item())
    except Exception:
        return -1.0


def _group_grad_norm(model, keyword: str) -> float:
    norms = []
    for name, param in model.named_parameters():
        if keyword in name and param.requires_grad:
            g = _grad_norm_metric(param)
            if g >= 0.0:
                norms.append(g)
    if not norms:
        return -1.0
    return float(max(norms))


def _count_params(model) -> Tuple[int, int, int, int]:
    total = 0
    trainable = 0
    dgp_trainable = 0
    lora_trainable = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if not param.requires_grad:
            continue
        trainable += n
        if "lora_" in name.lower():
            lora_trainable += n
        if any(k in name for k in DGP_KEYWORDS):
            dgp_trainable += n
    return total, trainable, dgp_trainable, lora_trainable


def _find_param(model, needle: str) -> Optional[torch.nn.Parameter]:
    for name, param in model.named_parameters():
        if needle in name:
            return param
    for name, param in model.named_parameters():
        if name.endswith(needle):
            return param
    return None


class DGPWandbMonitor:
    def __init__(self, trainer):
        self.trainer = trainer
        self.interval = max(int(getattr(trainer.args, "dgp_monitor_steps", 10)), 1)
        self.enabled = bool(getattr(trainer.args, "dgp_monitor_wandb", False))
        self.init_gate: Optional[float] = None
        self.init_qdti_scale: Optional[float] = None
        self._pending: Dict[str, Any] = {}
        self._init_logged = False

    def maybe_log_init(self, model) -> None:
        if not self.enabled or not _is_rank0(self.trainer) or self._init_logged:
            return
        if not _wandb_active(self.trainer):
            return

        gate_p = _find_param(model, "query_refiner.gate")
        scale_p = _find_param(model, "qdti_scale")
        self.init_gate = _scalar_param(gate_p)
        self.init_qdti_scale = _scalar_param(scale_p)

        total, trainable, dgp_trainable, lora_trainable = _count_params(model)
        payload = {
            "params/total": total,
            "params/trainable": trainable,
            "params/dgp_qdti_trainable": dgp_trainable,
            "params/lora_trainable": lora_trainable,
            "init/query_refiner_gate": self.init_gate if self.init_gate is not None else -1.0,
            "init/qdti_scale": self.init_qdti_scale if self.init_qdti_scale is not None else -1.0,
        }
        self._log_metrics(payload)
        self._init_logged = True

    def on_training_step_end(self, model, step_time_sec: float) -> None:
        if not self.enabled or not _is_rank0(self.trainer):
            return
        if not _wandb_active(self.trainer):
            return
        if not getattr(self.trainer.accelerator, "sync_gradients", False):
            return

        self.maybe_log_init(model)

        gate_p = _find_param(model, "query_refiner.gate")
        scale_p = _find_param(model, "qdti_scale")

        gate_val = _scalar_param(gate_p)
        scale_val = _scalar_param(scale_p)
        scale_grad = scale_p.grad if scale_p is not None else None

        self._pending = {
            "dgp/query_refiner_gate": gate_val if gate_val is not None else -1.0,
            "dgp/qdti_scale": scale_val if scale_val is not None else -1.0,
            "grad/query_refiner_gate": _grad_norm_metric(gate_p),
            "grad/qdti_scale": _grad_norm_metric(scale_p),
            "grad/prompt_adapter": _group_grad_norm(model, "prompt_adapter"),
            "grad/query_refiner_ffn": _group_grad_norm(model, "query_refiner.ffn"),
            "grad/qdti_bias_mlp": _group_grad_norm(model, "query_specific_text_memory_bias.bias_mlp"),
            "perf/step_time_sec": float(step_time_sec),
            "perf/max_cuda_memory_allocated_gb": (
                float(torch.cuda.max_memory_allocated() / (1024 ** 3))
                if torch.cuda.is_available()
                else -1.0
            ),
            "status/gate_started": int(
                self.init_gate is not None
                and gate_val is not None
                and gate_val != self.init_gate
            ),
            "status/qdti_scale_started": int(
                self.init_qdti_scale is not None
                and scale_val is not None
                and scale_val != self.init_qdti_scale
            ),
            "status/qdti_scale_grad_available": int(scale_grad is not None),
        }

        next_step = int(self.trainer.state.global_step) + 1
        if next_step == 1 or next_step % self.interval == 0:
            self._log_metrics(dict(self._pending))

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        payload = dict(metrics)
        payload["dgp/global_step"] = int(self.trainer.state.global_step)
        self.trainer.log(payload)


def attach_dgp_wandb_monitor(trainer, model) -> None:
    monitor = DGPWandbMonitor(trainer)
    trainer._dgp_monitor = monitor
    monitor.maybe_log_init(model)
