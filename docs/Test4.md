# Test 4: Query-Side Only Training (`segearth+pd`)

## Purpose

LaSeRS `test_multi_cate` gIoU is stuck around **~42.5%** on the 8w baseline. Diagnostics show:

1. Swin res4/5 features are linearly separable, but the text-blind `pixel_decoder` loses separability on hard samples (Probe A).
2. Frozen-LLM SEG/phrase hidden states are not zero-shot aligned with Swin (Probe E FAIL).
3. Phase 1a (PD-front FiLM + frozen SEG) also FAIL.

**Test 4** asks: can **mask loss** train spatial signal into language-side representations if we:

- unfreeze **LLM last 2 layers** (full params on **32-layer Mipha**: indices **30–31**),
- train **`SEG_token_projector` + `predictor`**,
- **freeze `pixel_decoder` and `lm_head`** (default),

without changing the forward topology?

**Test 5 / PPGB / PD-front fusion is out of scope for this fork.**

## vs Test 5

| | Test 4 | Test 5 (later) |
|---|---|---|
| Forward | baseline: Swin → PD → predictor ← SEG | PD-front language fusion |
| LLM | last N layers unfrozen (30–31) | TBD |
| pixel_decoder | frozen | may train / fuse |
| lm_head | **frozen by default** | TBD |
| Goal | query-side representation | cross-modal bridge |

## Fork

```
/root/rivermind-data/huangziyi/reseg/segearth+pd
```

Changes are **training-only** (`train.py`, `llava_trainer.py`, scripts). **`llava_phi.py` is unchanged.**

Scripts under `scripts/test4_*.sh` use **Unix LF** line endings (required on Linux).

**Output root:** all Test 4 artifacts default to `huangziyi/reseg/output/pd/` (override with `TEST4_OUTPUT_ROOT` or `RUN_ROOT`). Base checkpoints (`8w merged`) remain under `output/base/`.

## Full pipeline (recommended)

End-to-end: **baseline from base `lasers_test_table2_metrics.json` → train 5k (2k fail-fast stop) → test eval at 2k + 5k only**.

```bash
cd /root/rivermind-data/huangziyi/reseg/segearth+pd

# Link check + 1-step train + export (no full test eval, ~2 min)
bash scripts/test4_pipeline_verify.sh

# Full run (~2× test eval + train; baseline from output/base/, no identity re-eval)
bash scripts/test4_run_pipeline.sh
```

### Pipeline scripts

| Script | Purpose |
|--------|---------|
| `test4_common.sh` | Shared paths + parse gIoU + gate (source only) |
| `test4_identity_eval.sh` | Optional: re-measure baseline on merged model |
| `test4_smoke_train.sh` | Train (default **5k**, save every **1k**) |
| `test4_merge_checkpoint.sh` | ZeRO ckpt → merged HF (HF Trainer saves skip this) |
| `test4_eval_model.sh` | LaSeRS **test** eval + metrics + gate |
| `test4_smoke_eval.sh` | export checkpoint + test eval |
| `test4_eval_checkpoints.sh` | Eval selected steps (default **2k, 5k**) → CSV |
| `test4_run_pipeline.sh` | Baseline + train + eval orchestrator |
| `test4_probe_e.sh` | Optional Probe E after best ckpt |

### Environment knobs

```bash
MAX_STEPS=5000              # default
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=3          # keep latest checkpoints on disk
SKIP_IDENTITY=1             # default: read baseline from base table2 JSON
FAILFAST_STOP_TRAIN=1       # train 2k → eval → stop if Δ<+0.5pt, else resume to 5k
EVAL_CHECKPOINT_STEPS=2000,5000
RUN_PROBE_E=1               # optional Probe E
SKIP_TRAIN=1                # re-eval existing run only
TRAIN_OUTPUT_DIR=...        # with SKIP_TRAIN
MAX_EVAL_SAMPLES=5          # tiny eval for debugging only
SKIP_IDENTITY=0             # force identity re-eval before train
```

Output layout (`RUN_ROOT=output/pd/test4-YYYYMMDD_HHMMSS/`):

```
test4_baseline_multi_cate_giou.txt          # parsed from base table2 (42.45%)
baseline_lasers_test_table2_metrics.json  # copy of base metrics reference
train/                                      # checkpoints + test4_train_config.txt
train/test4_checkpoint_eval_summary.csv     # rows for 2k and 5k only
train/eval_ckpt-2000/ eval_ckpt-5000/
pipeline.log
```

**Gate rule:** compare checkpoint evals to **`test4_baseline_multi_cate_giou.txt`** (seeded from `output/base/.../lasers_test_table2_metrics.json` by default).

## One-Command Smoke Train

```bash
cd /root/rivermind-data/huangziyi/reseg/segearth+pd
bash scripts/test4_smoke_train.sh
```

Quick 1-step sanity check (outputs on data volume, not `/tmp`):

```bash
MAX_STEPS=1 WANDB_MODE=disabled \
  OUTPUT_DIR=/root/rivermind-data/huangziyi/reseg/output/pd/test4-smoke-1step \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  bash scripts/test4_smoke_train.sh
```

### Model base selection

`test4_smoke_train.sh` picks the first existing path:

1. `.../standard-base-lasers-siglip1-28w-gd4/merged_model`
2. else `.../standard-base-lasers-siglip1-8w-gd4/merged_model`
3. else `pretrained_model/mllm/Mipha-3B`

Override with `MODEL_PATH=...`. Log line `MODEL_BASE_SOURCE` shows which was used.

**Gate comparisons must use the same base checkpoint identity** (do not mix 8w vs 28w conclusions).

### What `--test4_mode` does

Automatically sets:

- `unfreeze_llm_last_n=2` → layers **30–31** on 32-layer Mipha-3B
- `freeze_pixel_decoder=True`
- `freeze_lm_head=True` (override with `--no_freeze_lm_head` for ablation)
- `lora_enable=False` (avoids LoRA + full-unfreeze double update)

Use **`--no_lora_enable`** (underscore) if disabling LoRA manually.

Differential learning rates (optimizer param groups):

- LLM last-N layers: `--llm_lr 1e-5`
- `SEG_token_projector` / `predictor` (and `lm_head` if unfrozen): `--mask_lr 1e-4`

`test4_smoke_train.sh` sets `WANDB_MODE=disabled` and `--report_to none` by default.

Logs are written to:

```
<output_dir>/test4_train_config.txt
```

## Eval + Gate

```bash
TRAIN_OUTPUT_DIR=/path/to/test4/run bash scripts/test4_smoke_eval.sh
```

Merge spot-check defaults to `model.model.layers.30.self_attn.q_proj.weight` (aligns with unfrozen layer 30). Override:

```bash
SPOT_CHECK_BASE=/path/to/base/merged_model bash scripts/test4_smoke_eval.sh
```

Optional Probe E:

```bash
MODEL_PATH=/path/to/merged_model bash scripts/test4_probe_e.sh
```

Requires `sample_list.json` (default tries `feature_probe_stage3`, falls back to `feature_probe_v1` under 8w output). Override:

```bash
SAMPLE_LIST=/path/to/sample_list.json bash scripts/test4_probe_e.sh
```

### Gates

| Gate | Criterion |
|---|---|
| **Primary** | `test_multi_cate` gIoU ≥ **same base checkpoint** + **2.0 pt** (8w baseline ≈ 42.45% → **≥ 44.45%**) |
| **Fail-fast** | after 2k steps, delta < **+0.5 pt** → stop before 5k |
| **Safety** | overall avg gIoU should not drop noticeably |
| **Probe E (aux)** | bad bucket SEG top100 > random (~0.011) |

Training-time `eval_dataset` is **not** the gate: LaSeRS has no `val/` split here, so train falls back to `train_data.json` with a `[WARN]`. **Gate uses `test4_smoke_eval.sh` → LaSeRS test split.**

## Why freeze `lm_head` by default?

`lm_head` is ~131M parameters. Training it dominates the mask-side optimizer group and can drift token logits away from the SEG-grounding objective. Test 4 focuses on **LLM last layers + SEG projector + predictor**. Use `--no_freeze_lm_head` only for ablation.

## Reference Numbers (same base only)

| Experiment | test_multi_cate gIoU |
|---|---|
| 8w baseline | 42.45% |
| Phase 1a FiLM | 41.63% (FAIL) |
| Probe E | Phrase−SEG Δ ≈ −0.003, win ~10% (FAIL) |

## Known Limits

- LaSeRS checkout here has **no `val/` split**; training eval falls back to `train_data.json` (`[WARN]`). **Gate = test eval only.**
- `--test4_mode` disables LoRA (scheme A).
- **HF Trainer checkpoints** (`model-*.safetensors` under `checkpoint-N/`) are used **directly for eval**; `test4_export_checkpoint` skips ZeRO merge. Use `test4_merge_checkpoint.sh` only for DeepSpeed ZeRO dirs (`zero_to_fp32.py` / `global_step*`).
- Root overlay `/` is often full; scripts set `TMPDIR` / Triton cache under `${RESEG_ROOT}/.tmp`. All Test 4 artifacts go under **`output/pd/`** (~50GB free for 5k + 5 evals).
- Legacy scripts `smoke_train_only.sh` etc. still point at `segearth+base`; use `scripts/test4_*.sh` for Test 4.
- Each DeepSpeed checkpoint is **~6GB+**; keep **~50GB free** for 5k run with 5 saves + merges under `output/base/`.

## Files

| File | Role |
|---|---|
| `segearth_r2/train/train.py` | Test 4 flags, freeze/unfreeze order, `[Test4]` log |
| `segearth_r2/train/llava_trainer.py` | differential LR param groups |
| `segearth_r2/train/merge_lora_weights_and_save_hf_model.py` | ZeRO export; spot-check layer 30 |
| `scripts/test4_smoke_train.sh` | 2k LaSeRS smoke train (LF) |
| `scripts/test4_smoke_eval.sh` | export + test eval + gate print (LF) |
| `scripts/test4_probe_e.sh` | Probe E wrapper with `--dump_features` (LF) |
