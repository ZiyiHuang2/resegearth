"""Lightweight W&B metrics for DGP-QDTI training verification (v6.1 guardrails)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

DGP_KEYWORDS = (
    "prompt_adapter",
    "query_refiner",
    "query_specific_text_memory_bias",
    "qdti",
    "gate",
    "qdti_scale",
)

PROBLEM_CLASSES = ("bridge", "vehicle", "ship", "tennis")

HEALTH_SCALAR_KEYS = (
    "cos_pg_pl",
    "cos_pg_qseg",
    "cos_pl_qseg",
    "cos_qdetail_qseg",
    "cos_qref_qseg",
    "entropy_pg_attention",
    "entropy_pl_attention",
    "entropy_pg_attention_norm",
    "entropy_pl_attention_norm",
    "detail_prompt_source",
    "target_flip_rate",
    "pg_top_token_idx_mean",
    "pl_top_token_idx_mean",
    "delta_g_norm",
    "delta_l_norm",
    "refiner_delta_norm",
    "seg_query_norm",
    "refiner_delta_over_seg",
    "query_refiner_gate_g",
    "query_refiner_gate_l",
    "query_refiner_gate",
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


def _unwrap_model(model):
    if hasattr(model, "base_model"):
        return model.base_model
    return model


def _read_dgp_health(model) -> Dict[str, Any]:
    base = _unwrap_model(model)
    health = getattr(base, "_dgp_last_health", None)
    if not isinstance(health, dict):
        return {}
    return dict(health)


def _gate_values(model) -> Tuple[Optional[float], Optional[float]]:
    base = _unwrap_model(model)
    refiner = getattr(base, "query_refiner", None)
    if refiner is None:
        return None, None
    if getattr(refiner, "use_sigmoid_gate", False):
        gate_g = float(torch.sigmoid(refiner.gate_g_logit).detach().float().cpu().reshape(-1)[0].item())
        gate_l = float(torch.sigmoid(refiner.gate_l_logit).detach().float().cpu().reshape(-1)[0].item())
        return gate_g, gate_l
    gate_g_p = _find_param(model, "query_refiner.gate_g")
    gate_l_p = _find_param(model, "query_refiner.gate_l")
    return _scalar_param(gate_g_p), _scalar_param(gate_l_p)


def check_stage_a_stop_conditions(
    health: Dict[str, Any],
    eval_metrics: Optional[Dict[str, float]] = None,
    prev_eval_score: Optional[float] = None,
    gate_g_init: float = 0.01,
    gate_l_init: float = 0.02,
) -> Dict[str, Any]:
    """Evaluate v6.1 Stage A stop criteria; returns flags + human-readable triggers."""
    triggers: List[str] = []
    flags: Dict[str, int] = {}

    cos_pg_pl = float(health.get("cos_pg_pl", 0.0))
    entropy_pl_norm = float(health.get("entropy_pl_attention_norm", 0.0))
    gate_g = float(health.get("query_refiner_gate_g", gate_g_init))
    gate_l = float(health.get("query_refiner_gate_l", gate_l_init))
    eval_score = None
    if eval_metrics:
        eval_score = float(eval_metrics.get("eval_score", eval_metrics.get("eval_eval_score", 0.0)))

    if cos_pg_pl > 0.9:
        if prev_eval_score is not None and eval_score is not None and eval_score <= prev_eval_score + 1e-6:
            triggers.append("cos(P_g,P_l)>0.9 and no eval_score gain")
            flags["stop_cos_no_gain"] = 1
        else:
            flags["stop_cos_no_gain"] = 0
    else:
        flags["stop_cos_no_gain"] = 0

    if entropy_pl_norm < 0.05:
        triggers.append("P_l normalized attention entropy collapsed (<0.05)")
        flags["stop_pl_attn_collapse"] = 1
    else:
        flags["stop_pl_attn_collapse"] = 0

    if entropy_pl_norm > 0.95:
        triggers.append("P_l normalized attention entropy near uniform (>0.95)")
        flags["stop_pl_attn_uniform"] = 1
    else:
        flags["stop_pl_attn_uniform"] = 0

    # legacy alias
    flags["stop_pl_attn_degenerate"] = flags["stop_pl_attn_collapse"]

    if gate_g <= 1e-4 or gate_l <= 1e-4:
        triggers.append("query_refiner gate_g or gate_l near 1e-4")
        flags["stop_gate_near_zero"] = 1
    else:
        flags["stop_gate_near_zero"] = 0

    base_good_model_zero_count = None
    base_good_model_zero_rate = None
    if eval_metrics:
        base_good_model_zero_count = float(
            eval_metrics.get(
                "base_good_to_model_zero_count",
                eval_metrics.get("eval_base_good_to_model_zero_count", 0.0),
            )
        )
        base_good_model_zero_rate = float(
            eval_metrics.get(
                "base_good_to_model_zero_rate",
                eval_metrics.get("eval_base_good_to_model_zero_rate", 0.0),
            )
        )

    if base_good_model_zero_count is not None and base_good_model_zero_count > 35:
        triggers.append(f"base>0.5->model=0 count={int(base_good_model_zero_count)} > 35")
        flags["stop_target_flip_base_good_model_zero"] = 1
    else:
        flags["stop_target_flip_base_good_model_zero"] = 0

    flags["stop_high_pr_at_0_9"] = 0
    flags["stop_problem_class_decline"] = 0

    should_stop = int(len(triggers) > 0)
    return {
        "should_stop": should_stop,
        "triggers": triggers,
        "base_good_to_model_zero_count": base_good_model_zero_count,
        "base_good_to_model_zero_rate": base_good_model_zero_rate,
        **flags,
    }


class DGPWandbMonitor:
    def __init__(self, trainer):
        self.trainer = trainer
        self.interval = max(int(getattr(trainer.args, "dgp_monitor_steps", 10)), 1)
        self.enabled = bool(getattr(trainer.args, "dgp_monitor_wandb", False))
        self.init_gate_g: Optional[float] = None
        self.init_gate_l: Optional[float] = None
        self.init_qdti_scale: Optional[float] = None
        self._pending: Dict[str, Any] = {}
        self._init_logged = False
        self._prev_eval_score: Optional[float] = None
        self._class_eval_prev: Dict[str, float] = {}

    def maybe_log_init(self, model) -> None:
        if not self.enabled or not _is_rank0(self.trainer) or self._init_logged:
            return
        if not _wandb_active(self.trainer):
            return

        self.init_gate_g, self.init_gate_l = _gate_values(model)
        scale_p = _find_param(model, "qdti_scale")
        self.init_qdti_scale = _scalar_param(scale_p)

        total, trainable, dgp_trainable, lora_trainable = _count_params(model)
        payload = {
            "params/total": total,
            "params/trainable": trainable,
            "params/dgp_qdti_trainable": dgp_trainable,
            "params/lora_trainable": lora_trainable,
            "init/query_refiner_gate_g": self.init_gate_g if self.init_gate_g is not None else -1.0,
            "init/query_refiner_gate_l": self.init_gate_l if self.init_gate_l is not None else -1.0,
            "init/qdti_scale": self.init_qdti_scale if self.init_qdti_scale is not None else -1.0,
        }
        self._log_metrics(payload)
        self._init_logged = True

    def _health_payload(self, model) -> Dict[str, Any]:
        health = _read_dgp_health(model)
        payload: Dict[str, Any] = {}
        for key in HEALTH_SCALAR_KEYS:
            if key in health:
                payload[f"health/{key}"] = float(health[key])
        source_name = health.get("detail_prompt_source_name")
        if source_name:
            payload["health/detail_prompt_source_name"] = source_name
        class_hints = health.get("class_hints", [])
        for cls in PROBLEM_CLASSES:
            count = sum(1 for hints in class_hints if cls in hints)
            payload[f"health/class_{cls}_batch_count"] = count
            key = f"class_{cls}_cos_pl_qseg"
            if key in health:
                payload[f"health/class_{cls}_cos_pl_qseg"] = float(health[key])
        return payload

    def on_training_step_end(self, model, step_time_sec: float) -> None:
        if not self.enabled or not _is_rank0(self.trainer):
            return
        if not _wandb_active(self.trainer):
            return
        if not getattr(self.trainer.accelerator, "sync_gradients", False):
            return

        self.maybe_log_init(model)

        gate_g, gate_l = _gate_values(model)
        scale_p = _find_param(model, "qdti_scale")
        scale_val = _scalar_param(scale_p)
        scale_grad = scale_p.grad if scale_p is not None else None
        health = _read_dgp_health(model)

        gate_g_p = _find_param(model, "query_refiner.gate_g")
        gate_l_p = _find_param(model, "query_refiner.gate_l")
        if gate_g_p is None:
            gate_g_p = _find_param(model, "query_refiner.gate_g_logit")
        if gate_l_p is None:
            gate_l_p = _find_param(model, "query_refiner.gate_l_logit")

        self._pending = {
            "dgp/query_refiner_gate_g": gate_g if gate_g is not None else -1.0,
            "dgp/query_refiner_gate_l": gate_l if gate_l is not None else -1.0,
            "dgp/qdti_scale": scale_val if scale_val is not None else -1.0,
            "grad/query_refiner_gate_g": _grad_norm_metric(gate_g_p),
            "grad/query_refiner_gate_l": _grad_norm_metric(gate_l_p),
            "grad/qdti_scale": _grad_norm_metric(scale_p),
            "grad/prompt_adapter": _group_grad_norm(model, "prompt_adapter"),
            "grad/query_refiner": _group_grad_norm(model, "query_refiner"),
            "grad/qdti_bias_mlp": _group_grad_norm(model, "query_specific_text_memory_bias.bias_mlp"),
            "perf/step_time_sec": float(step_time_sec),
            "perf/max_cuda_memory_allocated_gb": (
                float(torch.cuda.max_memory_allocated() / (1024 ** 3))
                if torch.cuda.is_available()
                else -1.0
            ),
            "status/gate_g_started": int(
                self.init_gate_g is not None
                and gate_g is not None
                and gate_g != self.init_gate_g
            ),
            "status/gate_l_started": int(
                self.init_gate_l is not None
                and gate_l is not None
                and gate_l != self.init_gate_l
            ),
            "status/qdti_scale_started": int(
                self.init_qdti_scale is not None
                and scale_val is not None
                and scale_val != self.init_qdti_scale
            ),
            "status/qdti_scale_grad_available": int(scale_grad is not None),
        }
        self._pending.update(self._health_payload(model))

        stop_report = check_stage_a_stop_conditions(
            health,
            gate_g_init=self.init_gate_g if self.init_gate_g is not None else 0.01,
            gate_l_init=self.init_gate_l if self.init_gate_l is not None else 0.02,
        )
        self._pending["stage_a/should_stop"] = stop_report["should_stop"]
        for key, value in stop_report.items():
            if key.startswith("stop_"):
                self._pending[f"stage_a/{key}"] = value

        next_step = int(self.trainer.state.global_step) + 1
        if next_step == 1 or next_step % self.interval == 0:
            self._log_metrics(dict(self._pending))

    def on_evaluate_end(self, eval_metrics: Dict[str, float]) -> None:
        if not self.enabled or not _is_rank0(self.trainer):
            return
        if not _wandb_active(self.trainer):
            return

        eval_score = float(eval_metrics.get("eval_score", eval_metrics.get("eval_eval_score", 0.0)))
        payload = {"eval/score": eval_score}
        prev_score = self._prev_eval_score
        if prev_score is not None:
            payload["eval/score_delta"] = eval_score - prev_score
        self._prev_eval_score = eval_score

        flip_keys = (
            ("base_good_to_model_zero_count", "stage_a/base_good_to_model_zero_count"),
            ("base_good_to_model_zero_rate", "stage_a/base_good_to_model_zero_rate"),
            ("base_zero_to_model_good_count", "stage_a/base_zero_to_model_good_count"),
            ("base_zero_to_model_good_rate", "stage_a/base_zero_to_model_good_rate"),
        )
        for src, dst in flip_keys:
            val = eval_metrics.get(src, eval_metrics.get(f"eval_{src}"))
            if val is not None:
                payload[dst] = float(val)

        model = _unwrap_model(self.trainer.model)
        health = _read_dgp_health(model)
        stop_report = check_stage_a_stop_conditions(
            health,
            eval_metrics=eval_metrics,
            prev_eval_score=prev_score,
            gate_g_init=self.init_gate_g if self.init_gate_g is not None else 0.01,
            gate_l_init=self.init_gate_l if self.init_gate_l is not None else 0.02,
        )
        payload["stage_a/should_stop"] = stop_report["should_stop"]
        for key, value in stop_report.items():
            if key.startswith("stop_"):
                payload[f"stage_a/{key}"] = value
        if stop_report["triggers"]:
            payload["stage_a/stop_triggers"] = "|".join(stop_report["triggers"])

        self._log_metrics(payload)

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        payload = dict(metrics)
        payload["dgp/global_step"] = int(self.trainer.state.global_step)
        self.trainer.log(payload)


def attach_dgp_wandb_monitor(trainer, model) -> None:
    monitor = DGPWandbMonitor(trainer)
    trainer._dgp_monitor = monitor
    monitor.maybe_log_init(model)

    original_evaluate = trainer.evaluate

    def evaluate_with_dgp_monitor(*args, **kwargs):
        metrics = original_evaluate(*args, **kwargs)
        monitor.on_evaluate_end(metrics)
        return metrics

    trainer.evaluate = evaluate_with_dgp_monitor
