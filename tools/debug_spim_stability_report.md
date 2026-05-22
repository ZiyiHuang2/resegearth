# SPIM-v1 small-scale stability test

## Environment

| 项 | 值 |
|---|---|
| branch | `exp/spim-v1-cross-attn-bias` |
| commit | `5ac1bbc`（以 `git log -1` 为准） |
| GPU | DeepSpeed `--include=localhost:1` → 物理 GPU **1**（RTX 4090） |
| torch / cuda | torch **2.1.0**, cuda **12.1**（`conda env: reseg`） |
| max_steps | **100**（脚本默认 `STABILITY_MAX_STEPS=500` 可改；本次为加快反馈采用 100） |

## Commands

**共同**：`seed=42`, `data_seed=42`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=1`, `bf16=True`, `gradient_checkpointing=False`, `report_to none`, `dataloader_num_workers=2`, `logging_steps=1`, `save_steps=999999`, `spim_debug=False`，`--use_spim True` + 下表 alpha。

**Test S0（完整一行，与日志中 deepspeed 命令一致）**

见 `tools/spim_stability_logs/s0_alpha0_steps100.log` **第 2 行**；等价参数核心：`--spim_alpha 0.0`，`--output_dir .../s0_alpha0`。

**Test S1**

见 `tools/spim_stability_logs/s1_alpha005_steps100.log` **第 2 行**；等价参数核心：`--spim_alpha 0.05`，`--output_dir .../s1_alpha005`。

**一键复现（100 step）**

```bash
cd /home/wangchengjun/huangziyi/reseg/segearth+att
source ~/miniconda3/etc/profile.d/conda.sh && conda activate reseg
STABILITY_MAX_STEPS=100 GPU_SLOT=localhost:1 bash tools/debug_spim_stability.sh
```

**500 step**

```bash
STABILITY_MAX_STEPS=500 GPU_SLOT=localhost:1 bash tools/debug_spim_stability.sh
```

## Results

| Test | alpha | max_steps | final / 指标 | NaN | OOM | peak memory (MiB) | avg step time | status |
|------|------:|----------:|---------------|:---:|:---:|------------------:|---------------:|--------|
| S0 | 0.0 | 100 | `train_loss`（100 步均值）**69.4427** | 否 | 否 | **~19682** | **~0.43 s/step**（43.4s/100） | OK |
| S1 | 0.05 | 100 | `train_loss`（100 步均值）**69.2976** | 否 | 否 | **~19682** | **~0.44 s/step**（44.2s/100） | OK |

说明：`train_loss` 为 Trainer 在 **100 步上的平均 total loss**（日志末尾 `train_loss` 字段）。

## Loss trend

**S0 前 10 个 `'loss':`（step 1–10 聚合日志）**

`68.8836, 103.0776, 115.0835, 60.9082, 95.89, 85.5536, 78.7254, 76.1675, 83.1883, 56.4191`

**S0 后 10 个 `'loss':`（step 91–100）**

`105.9768, 30.4812, 57.6457, 52.8389, 36.9215, 52.4349, 56.514, 52.2666, 56.4626, 61.1565`

**S1 前 10**

`68.8274, 103.1309, 113.6324, 60.835, 92.1279, 83.5465, 77.3197, 75.4352, 79.8216, 39.6213`

**S1 后 10**

`108.3728, 32.9251, 58.2171, 55.1999, 40.803, 52.0454, 56.6912, 54.4581, 59.2713, 60.1558`

**子项**：`loss_mask` / `loss_dice` / `loss_attention` 随 batch 波动（偶发 `loss_mask` 上百），**两组均类似**，未见仅 S1 发散；**全程无 NaN**。

## Comparison

- **alpha=0.05**：100 step 内 **无 NaN、无 OOM**，速度与 S0 几乎相同（~2.26 vs ~2.30 steps/s），显存峰值与 S0 一致 → **稳定性可接受**。
- **相对 alpha=0.0**：**首步 total loss** 与 1-step sanity 一致（S1 首步 **68.8274** vs S0 **68.8836**）；**100 步平均 total loss** S1 略低于 S0（**69.30 vs 69.44**，差约 **0.15**），属 batch 噪声量级，**不能单独证明「全程更强正则」**，但结合首步差异可认为 **SPIM bias 在前向中保持活跃且未破坏训练**。
- **速度**：相对 1-step（~2.2s/it）多步略快（~0.44s/it），符合 **warmup/cache** 后正常行为；**无明显因 SPIM 导致的减速**。

## Conclusion

**A. PASS** — 在 **100 steps、batch=1、GPU1** 条件下可进入更长实验/正式训练；若需更强证据可再跑 **`STABILITY_MAX_STEPS=500`** 同脚本对照。

## Next action

- 建议再跑 **`STABILITY_MAX_STEPS=500`** 确认长程均值与波动。
- 正式训练时保持 **`spim_debug False`**；显存紧张时优先 **换卡 / ZeRO**，勿与 **`gradient_checkpointing=True` + 当前 attention loss** 混用（见 sanity 报告）。
