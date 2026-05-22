# public_semantic_v2 相对 baseline_raw 的归因分析（RRSISD test）

本文档为只读分析结论；**未进行训练、未改模型与语义库、未调用 API**。  
分析脚本：`tools/analyze_public_semantic_v2_per_sample.py`  
数据产物目录：`outputs/source/public_semantic_v2_analysis/`

---

## 1. 当前整体结果复述

| 指标 | baseline_raw | public_semantic_v2 | 差值 |
|------|----------------|-------------------|------|
| gIoU (%) | 66.4690 | 66.9801 | +0.5111 |
| cIoU (%) | 78.2134 | 78.3064 | +0.0930 |
| mDice (%) | 74.2122 | 74.7022 | +0.4900 |
| mPrecision (%) | 75.4938 | 76.1665 | +0.6727 |
| mRecall (%) | 75.9295 | 76.1746 | +0.2451 |

整体上 **mDice / gIoU 提升与 mPrecision 提升幅度大于 mRecall**，与下文「预测面积略减 + 精度主导」一致。

---

## 2. 任务一：预测文件与标注对齐（只读结论）

### 2.1 `test_results` 目录与命名

- 两组实验输出目录下均存在 `test_results/`，预测数量约为 **3481** 个 `*.tif`（与 `refs(unc).p` 中 `split=="test"` 条数一致）。
- 命名规则来自 `segearth_r2/eval/eval.py`：`{image_stem}_{ref_id}_test_{mask_id}.tif`，其中 `image_stem` 为 JPEG 文件名去掉扩展名（保留前导零，如 `03600`），`ref_id` 为 `refs(unc).p` 中的 `ref_id`，单目标时 `mask_id` 恒为 `0`。

### 2.2 预测格式

- 使用 `tifffile.imwrite` 写出；用 OpenCV读取验证为 **uint8、取值 {0,255}** 的二值 mask，空间尺寸与评估时图像一致（示例 1024×1024）。

### 2.3 与 `refs(unc).p`、`instances.json` 的对齐

- `RRSISDDataset`（`segearth_r2/datasets/dataset.py`）对 `split=="test"` 按 **refs 列表顺序** 构造样本；`__getitem__(idx)` 使用 `ref = self.reason_file[idx]`，`data_id = ref["ref_id"]`，与 `eval_seg` 写回 `output["id"]` 一致。
- `instances.json` 提供 `annotations` 与 `categories`；通过 `ref["ann_id"]` 取标注，`ref["category_id"]` 与 `categories` 表得到 **category_name**。
- `rrsisd_test_metrics.json` 的 `details[].idx` **等于上述数据集索引**；当前 test 共 3481 条 ref，其中 **1 条因空 GT 在指标计算中被跳过**（本数据中缺失索引为 **83**，对应 `ref_id=413`），故 `details` 共 **3480** 条。预测文件仍可能为跳过样本生成 mask（例如存在 `00413_413_test_0.tif`），但 **本分析仅使用 `details` 中有 per-pixel 指标的 3480 条**，与正式 test 汇总一致。

### 2.4 每条样本可恢复的字段

由 `details` + `refs(unc).p` + `instances.json` 可恢复：

| 字段 | 来源 |
|------|------|
| `sample_index` | `details[].idx`（= `RRSISDDataset` 在 test split 的下标） |
| `ref_id` | `details[].id` = `ref["ref_id"]` |
| `ann_id` | `ref["ann_id"]` |
| `image_id` | `ref["image_id"]`（COCO 图像 id，整数） |
| `category_id` | `ref["category_id"]` |
| `category_name` | `instances.json` → `categories` |
| `expression` | `details[].description`（与评估所用句子一致） |

`details` 中另含 `pred_path`、`iou`、`dice`、`precision`、`recall`、`inter`、`union`，无需再读预测图即可做 IoU / 面积归因（面积由 `inter/precision`、`inter/recall` 恢复，见脚本说明）。

---

## 3. 任务二：per-sample IoU / Dice 差异摘要

输出文件：

- `outputs/source/public_semantic_v2_analysis/per_sample_iou_diff.jsonl`
- `outputs/source/public_semantic_v2_analysis/top50_improved.jsonl`
- `outputs/source/public_semantic_v2_analysis/top50_degraded.jsonl`

在 **3480** 条有效样本上：

- **ΔIoU > 0**：1647；**< 0**：1566；**= 0**：267  
- **ΔIoU 中位数**：0（大量完全重合或相同预测导致 ties）  
- **ΔIoU 分位数（近似）**：P1 ≈ -0.451，P99 ≈ +0.629  
- **范围**：约 [-0.940, +0.939]  
- **ΣΔIoU**（全测试子集 macro 改进之和）≈ **17.79**

---

## 4. top improved / top degraded 的类别分布

来自 `analysis_summary.json` 的计数（各取 ΔIoU 绝对值最大的 50 条中的类别频次）。

**Top50 improved** 中较多：**vehicle**（11）、**groundtrackfield**（5）、**bridge / ship**（各 4）、高速收费站/篮球场/overpass/储罐/网球场等。

**Top50 degraded** 中较多：**bridge**（8）、**vehicle**（7）、**groundtrackfield / overpass**（各 6）、**chimney**（4）等。

解读：少数结构类（桥、立交）同时出现在改善与恶化尾部，说明 **同类内方差大、存在难例与强正例**；**vehicle** 在改善榜占比高，与下文「vehicle 总贡献最大」一致。

---

## 5. 任务三：per-category 结果说明

完整表见：

- `outputs/source/public_semantic_v2_analysis/per_category_metrics.csv`
- `outputs/source/public_semantic_v2_analysis/per_category_metrics.json`

**指标定义（脚本实现）：**

- **gIoU / macro mean IoU**：该类别内 per-sample `iou` 的算术平均（与全局 `mIoU` 在 RRSISD 汇总脚本中与 gIoU 数值对齐的口径一致）。  
- **cIoU / pooled**：该类别内 `sum(inter) / sum(union)`，类比整体 cIoU 的「像素池化」口径。  
- **pred_area / gt_area**：由每条 `inter、precision、recall` 反推（`pred_area = inter/precision`，`gt_area = inter/recall`，并对零分母保护）。

**RRSISD 类别集合说明：** 官方 **20 类**（如 `airport`、`bridge`、`vehicle`、`harbor`、`ship`、`golffield` 等）。你列出的 water / road / vegetation 等 **不在该数据集的 category 表中**；脚本中 `user_focus_overlap` 列仅标记与给定英文焦点词 **精确同名** 的类（本数据中命中：`harbor`、`ship`、`bridge`、`vehicle`、`airport`）。

### 5.1 按 macro-ΔIoU 排序的前若干类（节选）

| category_name | n | macro ΔIoU | pooled ΔcIoU | Δ mean pred area |
|---------------|---:|------------|--------------|------------------|
| harbor | 27 | +0.0217 | +0.0353 | +832 |
| ship | 182 | +0.0207 | +0.0175 | -507 |
| golffield | 112 | +0.0203 | +0.0055 | +1517 |
| basketballcourt | 121 | +0.0187 | +0.0104 | -99 |
| storagetank | 110 | +0.0114 | -0.0069 | -65 |
| vehicle | 558 | +0.0114 | +0.0198 | -15 |
| Expressway-Service-area | 123 | +0.0081 | -0.0015 | -2515 |
| bridge | 269 | +0.0047 | +0.0116 | +384 |
| airport | 196 | ~0 | -0.0048 | -1402 |
| groundtrackfield | 254 | -0.0017 | +0.0021 | +168 |
| overpass | 226 | -0.0085 | -0.0165 | -489 |
| chimney | 107 | -0.0183 | -0.0208 | +637 |

（完整 20 行见 CSV。）

### 5.2 按「总贡献」`sum(ΔIoU)` 排序的头部类别

1. **vehicle**（558 样本）总增益最大  
2. **ship**  
3. **golffield**  
4. **basketballcourt**  
5. **bridge**  
6. **storagetank**  
7. **Expressway-Service-area**  
8. **windmill**

---

## 6. 哪些类别贡献最大（结论）

- **样本量 × 单样本改进叠加**：**vehicle** 为首要正贡献来源；**ship、golffield、basketballcourt** 次之。  
- **macro 平均改进最高的类**（小样本需方差警惕）：**harbor、ship、golffield、basketballcourt** 等。  
- **明确拖累或接近零**：**chimney** 明显下降；**overpass、trainstation** macro 下降；**airport** 几乎持平略负。  
→ 与「仅靠 scope/exclusion 修复空间溢出」的叙事 **部分一致**（vehicle、ship、harbor、bridge 有正信号），但 **airport 未体现收益、overpass 反降**，不能将 v2 简单归因于「所有易溢出基础设施类全面变好」。

---

## 7. public_v2 更像提升 precision、recall，还是减少 mask 外溢？

全量 3480 条上，脚本统计的平均变化为：

- **mean(Δprecision) ≈ +0.00673**  
- **mean(Δrecall) ≈ +0.00245**  
- **mean(Δpred_area) ≈ -172 像素/样本**（在 1024 级别分辨率上为温和收缩）

结论：**整体更接近「精度主导、略抬召回」，并伴随平均预测面积略减**——与「减少无关区域响应（外溢）从而抬高 precision」的图像一致，但 **不是**「只靠大幅收缩 mask」那种极端模式（recall 仍有小正增益）。

---

## 8. 是否支持「scope + exclusion 是主要收益来源」？

- **支持的部分证据**：**vehicle、ship、harbor、bridge（pooled）** 等空间语义强的类别有 **正向 macro / 或 pooled IoU**；全样本 **precision 提升大于 recall**、平均预测面积略降，符合「抑制无关区域」机制的部分表型。  
- **不支持或过弱的证据**：**airport** 基本持平略差；**overpass** 明显下降；**top50 恶化**里 **bridge/overpass** 占比较高。  
→ 综合判断：**弱到中等支持**该假设；更准确的表述是「**对若干交通/舰船/港口类有帮助，但基础设施子类之间不一致**」，不宜仅凭先验断定 v2.1 的 scope/exclusion 字段会系统性再涨一截。

---

## 9. 任务四：是否建议跑 v2.1（自动规则 + 本数据解读）

脚本启发式结论（见 `per_category_metrics.json` → `verdict`）：

- **规则码 D**：**top50 正增益样本占全部正增益之和约 36.3%**（略高于 35% 阈值），且 **前三类别对 |ΣΔIoU| 占比约 44.4%** → 增益在样本与类别维度上 **相对集中**。  
- 同时，**空间代理组 A**（bridge、airport、vehicle、harbor、ship、overpass、高速服务区/收费站）的 **平均 ΔIoU（≈0.00695）仅略高于** **场地近似组 B**（golffield、groundtrackfield、stadium、球场类、dam 等，≈0.00596），**未达到**规则 A 所要求的「明显强于 B + 余类」的严格阈值。

**对应用户给定决策树：**

- **D（优先）**：提升集中在少数类别/少数高 Δ 样本上，**应先做 top improved / degraded 可视化核对**，不宜贸然开 v2.1 长训。  
- **B 信号（次要）**：golffield、basketballcourt 等 **场地/纹理类** 的总贡献与 macro 提升 **不可忽视** → 若 v2.1 强化与视觉证据相关描述，**存在「对部分类过拟合先验」的风险**，与「不建议优先跑 v2.1」方向一致，但原因不是 RRSISD 里的「森林/农田」（数据集中不存在），而是 **球场/绿地类已分得一杯羹**。  
- **A 信号（弱）**：vehicle/ship/harbor/bridge 有正贡献，但 **airport/overpass 不支撑「溢出类全面上涨」**。

**是否建议跑 v2.1：** 基于当前归因，**不建议立刻以「确定大涨」的预期开 v2.1 长训**；更稳妥的是 **先抽样可视化 D 类尾部样本**，若确认 vehicle/ship 等的收益来自「定位更准而非偶然」，再小规模消融或短跑验证 v2.1 字段。

---

## 10. 后续是否更应该多 seed？

在 **规则 C（各类均匀小幅）不成立**、且 **D（集中性）成立** 的前提下：整体 gIoU 提升约 **0.51pt**，其中相当比例可由 **少数样本/少数类别**解释 → **优先多 seed / 重复实验估计方差** 与「先可视化再改库」同样重要；否则难以区分 **语义库效应** 与 **优化噪声/随机性**。

---

## 11. 复现命令

在仓库根目录 `resegearth+source` 下：

```bash
python3 tools/analyze_public_semantic_v2_per_sample.py \
  --baseline-metrics /home/wangchengjun/huangziyi/reseg/output/source/rrsisd_baseline_raw_7w/rrsisd_test_metrics.json \
  --public-v2-metrics /home/wangchengjun/huangziyi/reseg/output/source/rrsisd_public_semantic_v2_7w/rrsisd_test_metrics.json \
  --rrsisd-data-root /home/wangchengjun/huangziyi/data/RRSISD \
  --out-dir outputs/source/public_semantic_v2_analysis
```

---

*生成说明：表格与数值来自上述脚本一次性跑出的 JSON/CSV；若你更新了 `rrsisd_test_metrics.json`，请重新运行脚本并视需要同步修订本节数字。*
