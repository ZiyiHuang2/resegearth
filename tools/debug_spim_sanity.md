# SPIM-v1 1-step sanity check report

（由 `tools/debug_spim_sanity.sh` 于本机执行生成；未改 `llava_phi.py` / `mask2former_transformer_decoder.py` / `train.py`。）

## Environment

| 项 | 值 |
|---|---|
| branch | `exp/spim-v1-cross-attn-bias` |
| commit | `5ac1bbc`（当时 HEAD；工作区另有未提交 `run_train_merge_test.sh` 时请以 `git log -1` 为准） |
| conda/venv | `conda activate reseg` |
| GPU | DeepSpeed `--include=localhost:1` → `CUDA_VISIBLE_DEVICES=1`（RTX 4090） |
| torch / cuda | torch 2.1.0, cuda 12.1 |

## 前置检查：`--help` 中的 SPIM 参数

```bash
cd /home/wangchengjun/huangziyi/reseg/segearth+att
source ~/miniconda3/etc/profile.d/conda.sh && conda activate reseg
python segearth_r2/train/train.py --help | grep -E 'spim|use_spim'
```

已通过，可见 `--use_spim`、`--spim_alpha`、`--spim_layer_idx`、`--spim_detach`、`--spim_norm`、`--spim_seg_agg`、`--spim_near_zero_eps`、`--spim_debug`。

## Commands（与脚本等价）

统一：`max_steps=1`，`save_steps=999999`，`seed=42`，`data_seed=42`，`report_to none`，`dataloader_num_workers=2`（因 `train.py` 默认 `dataloader_prefetch_factor=2`，**不能**与 `num_workers=0` 同用，否则会触发 Transformers 校验错误），`gradient_checkpointing=False`（见下「踩坑」），`GPU_SLOT=localhost:1`。

**Test 1（alpha=0.0）**

```bash
deepspeed --master_port=29610 --include=localhost:1 segearth_r2/train/train.py \
  ...（同 tools/debug_spim_sanity.sh 内 run_one 的完整参数）... \
  --use_spim True --spim_alpha 0.0 --spim_layer_idx -1 --spim_detach True --spim_norm True \
  --spim_seg_agg mean --spim_near_zero_eps 1e-8 --spim_debug True
```

**Test 2（alpha=0.001）**：同上，`--spim_alpha 0.001`，`--output_dir .../t2_alpha0001`。

**Test 3（alpha=0.05）**：同上，`--spim_alpha 0.05`，`--output_dir .../t3_alpha005`。

一键复现：

```bash
bash tools/debug_spim_sanity.sh
```

日志：`tools/spim_sanity_logs/t1_alpha0.log`、`t2_alpha0001.log`、`t3_alpha005.log`。

## Results

| Test | alpha | prior shape | near_zero_count | spatial_bias shape（示例：最后一层 scale） | first-step `train_loss` | status |
|------|------:|-------------|----------------:|-----------------------------------------------|--------------------------:|--------|
| Test 1 | 0.0 | (1,1,27,27) | 0 | **N/A**（无 `spatial_bias` 日志，符合预期） | **68.8836** | OK |
| Test 2 | 0.001 | (1,1,27,27) | 0 | 每层打印，例如 `(8, 1, 1024)` / `(8, 1, 4096)` / `(8, 1, 16384)`（num_heads=8, Q=1） | **68.8843** | OK |
| Test 3 | 0.05 | (1,1,27,27) | 0 | 同 Test 2 形式 | **68.8274** | OK |

**Prior 统计（三组相同，来自最后一层 attention 与 z-score）**

- min ≈ -0.060684，max ≈ 19.821026，mean ≈ 0.0，std ≈ 0.999921  
- `target_size`：按层循环为 `(32,32)`、`(64,64)`、`(128,128)`（与 Mask2Former 多尺度一致）

**NaN / shape assert / OOM**

- 三步均未出现 `NaN`、`spatial_bias` shape 报错或 CUDA OOM（在 **GPU1** 上）。

## Loss comparison

- **L_alpha0** = 68.88361358642578  
- **L_alpha0001** = 68.88426208496094  
- **L_alpha005** = 68.82740783691406  

- **L_alpha0001 vs L_alpha0**：差约 **6.5e-4**，与「极小 alpha、bf16」预期一致。  
- **L_alpha005 vs L_alpha0**：差约 **0.056**，`loss_mask` 等分量亦有小幅变化，**可视为 SPIM bias 已参与前向并影响标量 loss**。

## 执行过程中的踩坑（仅脚本/环境，未改模型）

1. **`--dataloader_num_workers 0`**：与 `TrainingArguments` 默认 `dataloader_prefetch_factor=2` 冲突 → 改为 **`workers=2`**。  
2. **GPU0 OOM**：本机 GPU0 已被占满 → 默认改用 **`localhost:1`**。  
3. **`gradient_checkpointing=True`**：`outputs.attentions` 为 `None`，现有 `loss_attention` 仍 `sum` attention → **AttributeError**；sanity 脚本保持 **`gradient_checkpointing False`**。若日后既要 GC 又要 attention loss，需另开 issue 改模型侧（本次未做）。

## Conclusion

**A. PASS（在「单卡、同 seed、1 step、当前数据 batch」意义下）**

- `[SPIM config]` 与 CLI 一致，说明 **train.py 已将 SPIM 写入 `model.config`**。  
- `prior` 为 **(1,1,27,27)**，`near_zero_count=0`，prior 路径与 **mask_num 对齐**正常。  
- **alpha=0.0**：无 decoder 侧 `spatial_bias` 日志。  
- **alpha>0**：出现 `build_spim_bias` 相关 **`target_size` + `spatial_bias.shape`**，且 **无 shape assert 异常**。  
- **loss 非 NaN**；**alpha=0.05 相对 0.0 的 total loss 有明显差异**。

## Next action

- 可在当前配置下做 **小规模多 step 训练**（正式跑时 `--spim_debug False`）。  
- 若需在 **显存紧张 GPU** 上跑：优先 **换空闲卡 / ZeRO offload**，而不是打开 **gradient_checkpointing**（除非同时修 attention 为 None 时的 `loss_attention` 分支）。  
- 若某环境 **L_alpha005 与 L_alpha0 仍完全一致**：对照本报告查 `spim_near_zero_count`、`model.config.spim_alpha` 与日志是否进入 `spatial_bias` 分支。
