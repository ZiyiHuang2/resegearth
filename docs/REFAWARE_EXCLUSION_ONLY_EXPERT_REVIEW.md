# REFAWARE_EXCLUSION_ONLY 专家审核报告

本文档基于仓库 **`resegearth+source`** 内真实代码、shell、以及 **`/home/wangchengjun/huangziyi/reseg/output/source/`** 下已落盘的 `trainer_state.json`、`rrsisd_test_metrics.json` 等只读材料整理。**未修改**任何 Python/JSON/sh/ckpt/eval 产物；未执行训练、merge、eval。

---

## 0. 一页结论

| 维度 | 结论 |
|------|------|
| **当前指标结论** | 三组在 RRSIS-D **test** 上的整体指标（`mIoU`/`oIoU`/`mDice`/`mPrecision`/`mRecall`）差异很小；**不能**声称 `refaware_exclusion_only_28w` 相对 `baseline_raw_7w` 或 `public_semantic_v2_7w` 有显著整体提升。精确数值见 **§7** 与 `outputs/source/refaware_exclusion_only_expert_review/experiment_metric_summary.csv`。 |
| **代码链路结论** | `concept_refaware_prior` 在 `DataArguments` 中默认 `False`，与 `concept_match_strict` 互斥；`RRSISDDataset` 在「有 lib 且 `concept_refaware_prior`」分支调用 `build_rrsisd_refaware_exclusion_only_human_value`；`token_refer_id` 仍来自原始 `instruction`；`labels` 侧人类轮次仍经 `preprocess_llama2` 将 instruction（含 prior）mask 为 `IGNORE_INDEX=-100`。Eval 无 semantic 参数，符合「训练注入 prior、推理不注入」设计。 |
| **最大风险** | **训练步数不对照**：`baseline_raw_7w` / `public_semantic_v2_7w` 实际 **`max_steps=70000`**，而 `refaware_exclusion_only_28w` 为 **`280000`**。整体 test 指标不可严格视为「同计算预算下单变量 refaware」的结论。另：工作区对 `run_refaware_exclusion_only_28w_full.sh` 有未提交修改（默认 `MODEL_NAME_OR_PATH` / `MASTER_PORT`），但**磁盘上已训练 LoRA** 的 `adapter_config.json` 显示基座仍为 `bseg/baseline_standard-base_5w/merged_model`，与 7w 实验一致。 |
| **是否建议继续** | 语义库方向**未**被当前整体指标否定，但需 **步数对齐对照**、**per-category / per-sample**、以及确认后续脚本默认基座不被误改后再做结论。 |

---

## 1. 实验背景与三组对照

| 实验 ID | 语义 prior | 训练步数（以 `trainer_state.json` 为准） | W&B run 名（脚本约定） | test metrics 文件 |
|---------|------------|-------------------------------------------|------------------------|-------------------|
| baseline_raw_7w | 不注入 public v2 库 | 70000 | `rrsisd_baseline_raw_7w` | `.../rrsisd_baseline_raw_7w/rrsisd_test_metrics.json` |
| public_semantic_v2_7w | 命中 concept 全量注入（与 v2 一致） | 70000 | `rrsisd_public_semantic_v2_7w` | `.../rrsisd_public_semantic_v2_7w/rrsisd_test_metrics.json` |
| refaware_exclusion_only_28w | target 全量 JSON；reference 仅 `exclusion_rule` | 280000 | `rrsisd_public_semantic_v2_refaware_exclusion_only_28w` | `.../rrsisd_public_semantic_v2_refaware_exclusion_only_28w_eval_best/rrsisd_test_metrics.json` |

**说明**：`refaware` 的 metrics 位于 `*_eval_best` 子目录，与 baseline/v2 直接在 run 根目录下的命名不同，有利于避免混读（见 §6）。

---

## 2. 代码改动范围

### 2.1 Git 状态（`resegearth+source`）

- **分支**：`source`，**HEAD**：`aab8b92`（message `v2.2`）。
- **未提交修改**：仅 `run_refaware_exclusion_only_28w_full.sh`（2 处：`MASTER_PORT` 默认、`MODEL_NAME_OR_PATH` 默认）。
- 用户点名的核心 Python / JSON：`git diff` **无**工作区改动（字节数 0）。

### 2.2 文件级审计表

| 文件 / 目录 | 是否有 diff（相对 HEAD） | 改动摘要 | 风险判断 |
|-------------|-------------------------|----------|----------|
| `run_refaware_exclusion_only_28w_full.sh` | **是**（未提交） | 默认 `MASTER_PORT` 29500→29522；默认 `MODEL_NAME_OR_PATH` bseg→siglip11 | **中**：不影响已生成 ckpt/metrics；若未设环境变量重训会破坏与 7w 的可比性 |
| `segearth_r2/utils/concept_public_grounding_train.py` | 否 | — | 低 |
| `segearth_r2/datasets/dataset.py` | 否 | — | 低 |
| `segearth_r2/train/train.py` | 否 | — | 低 |
| `segearth_r2/eval/eval.py` | 否 | — | 低 |
| `segearth_r2/train/merge_lora_weights_and_save_hf_model.py` | 否 | — | 低 |
| `configs/concept_public_semantic_library_v2.json` | 否 | — | 低 |
| `segearth_r2/model`、`segearth_r2/models` | 否 | — | 低 |
| `run_train_baseline_raw_vs_public_semantic_v2_7w.sh` | 否（本快照） | — | 低 |

### 2.3 是否只改了「预期」文件

- **已落盘训练代码**（相对 HEAD）：无未提交改动 → 审计的是 **当前提交中的实现**。
- **工作区**：仅有 refaware 一体化脚本的默认路径/端口漂移，应视为 **脚本维护风险**，不是 refaware 算法本身的未提交 patch。

### 2.4 结构性问题速查（是否触碰）

| 问题 | 结论 | 依据 |
|------|------|------|
| 模型结构 | **未**改（本任务为数据侧 prompt + LoRA 训练） | 无 `segearth_r2/model(s)` diff；训练仍为 LoRA |
| mask decoder | **未**改配置路径与 merge 参数仍走原 yaml / tower | 脚本与历史一致 |
| loss | **未**改 refaware 分支 | 仅 human `value` 文本变化 |
| eval | **未**改 `eval.py`；无 concept 相关 CLI | `grep` 无 `concept_` |
| merge | **未**改 merge Python；sh 中 merge 仅 LoRA 与视觉塔 | 见 §5 |
| 语义库 JSON | **未**改 | git diff 为空 |
| train list / split | **未**改（本报告范围） | 未观测到 split 文件 diff |

---

## 3. Refaware 实现逻辑

### 3.1 参数与互斥

1. **`concept_refaware_prior` 是否存在**：是，定义于 `DataArguments`。
2. **默认是否为 False**：是，`default=False`。
3. **与 `concept_match_strict` 是否互斥**：是，二者同时为 True 时 `train.py` 抛出 `ValueError`。
4. **是否仅在 RRSISDDataset 且 lib 存在时生效**：是，`dataset.py` 分支为 `elif lib and self.concept_refaware_prior`，且前述 `concept_match_strict` 分支优先。

代码位置（节选）：

```79:86:huangziyi/reseg/resegearth+source/segearth_r2/train/train.py
    concept_refaware_prior: bool = field(
        default=False,
        metadata={
            "help": "RRSIS-D only: split matched library concepts into target (category match) vs reference; "
            "inject full v2 JSON only for targets and exclusion_rule-only lines for references. "
            "Incompatible with concept_match_strict."
        },
    )
```

```286:288:huangziyi/reseg/resegearth+source/segearth_r2/train/train.py
    if getattr(data_args, "concept_match_strict", False) and getattr(data_args, "concept_refaware_prior", False):
        raise ValueError(
            "concept_match_strict and concept_refaware_prior cannot both be True."
```

```352:456:huangziyi/reseg/resegearth+source/segearth_r2/datasets/dataset.py
        lib = self.concept_public_semantic_library
        if lib and self.concept_match_strict:
            ...
        elif lib and self.concept_refaware_prior:
            matched_all = retrieve_matched_public_grounding(instruction, lib)
            tgt_rows, ref_rows = split_matched_concepts_target_and_reference(
                matched_all, str(category_name_str).strip() if category_name_str else None
            )
            human_value = build_rrsisd_refaware_exclusion_only_human_value(
                instruction,
                lib,
                str(category_name_str).strip() if category_name_str else None,
                matched_precalc=matched_all,
            )
            ...
        else:
            human_value = build_rrsisd_supervised_human_value(instruction, lib)

        token_refer_id = self.preprocess_referring_instruction(instruction)
```

### 3.2 Target / reference 注入规则（与需求对照）

实现位于 `build_rrsisd_refaware_exclusion_only_human_value` 与 `format_reference_exclusion_prior_section`：

```124:182:huangziyi/reseg/resegearth+source/segearth_r2/utils/concept_public_grounding_train.py
def format_reference_exclusion_prior_section(reference_concepts: List[Dict[str, Any]]) -> str:
    """
    Minimal exclusion block for reference concepts only (no visual_evidence / mask_scope).
    """
    ...
def build_rrsisd_refaware_exclusion_only_human_value(
    instruction: str,
    concept_public_library_path: str,
    category_name: Optional[str],
    *,
    matched_precalc: Optional[List[Dict[str, Any]]] = None,
) -> str:
    ...
    target_concepts, reference_concepts = split_matched_concepts_target_and_reference(
        matched, category_name
    )
    blocks: List[str] = []
    if target_concepts:
        tgt = format_grounding_appendix(target_concepts)
        if tgt:
            blocks.append(tgt.rstrip())
    excl = format_reference_exclusion_prior_section(reference_concepts)
    if excl:
        blocks.append(excl)
    appendix = "\n\n".join(blocks) if blocks else ""
    parts: List[str] = [
        "This is an image <|sep|> <|user|>\n",
        "Please do Reasoning Segmentation according to the following expression.\n",
        (instruction or "").strip() + "\n",
    ]
    if appendix:
        parts.append(appendix + "\n")
    parts.append("<|vision_bos|> <image> <|vision_eos|>\n<refer> <|assistant|>")
    return "".join(parts)
```

| 需求项 | 满足情况 |
|--------|----------|
| target：`concept` 与 `category_name` 归一化一致 → 全量 v2 JSON（`format_grounding_appendix`） | **是** |
| reference：仅 `exclusion_rule` 行（无 `visual_evidence` / `mask_scope`） | **是**（exclusion 块为文本行，非整行 JSON） |
| `target_concepts` 为空 → 不追加空 Target 块 | **是**（`if target_concepts`） |
| `reference_concepts` 为空 → 不输出空 Exclusion 段 | **是**（`format_reference_exclusion_prior_section` 在仅 header 时返回 `""`） |
| `matched` 为空 → `appendix` 空 → 等价于无 prior | **是** |
| `token_refer_id` 仍基于原始 `instruction` | **是**（见 `dataset.py` 456 行） |
| prior 在 human 轮 → `preprocess_llama2` mask 为 `IGNORE_INDEX` | **是**（`IGNORE_INDEX = -100`） |

```6:8:huangziyi/reseg/resegearth+source/segearth_r2/utils/constants.py
# Model Constants
IGNORE_INDEX = -100
```

### 3.3 图像 token 与 raw expression 顺序

refaware 与 v2 共用同一布局：**expression →（可选）appendix → `<|vision_bos|> <image>`**。与 `build_rrsisd_supervised_human_value` 中 v2 路径一致（见同文件 `build_rrsisd_supervised_human_value` 212–219 行）。

### 3.4 Refaware 分支伪代码

```
matched = retrieve_matched_public_grounding(instruction, lib)   # 或 matched_precalc
(target_rows, reference_rows) = split_by_normalized_category(matched, category_name)
blocks = []
if target_rows:
    blocks += full_v2_json_block(target_rows)          # format_grounding_appendix
if reference_rows:
    blocks += exclusion_only_lines(reference_rows)     # format_reference_exclusion_prior_section
human_value = template(expression, optional_appendix=join(blocks), then_image_tokens)
token_refer_id = tokenize(instruction + "[SEG]")     # 与 v2 相同，不受 prior 影响
labels = preprocess_llama2.mask_human_round_to_IGNORE_INDEX(...)
```

### 3.5 三个样例（静态推导，未动态验证）

以下假定 **语义库检索** `retrieve_matched_public_grounding` 对英文表达返回给定 concept 行（与真实 JSON 命中一致时成立）。**未**从训练 stderr 截取 debug 行。

**A. `ship near harbor`，`category_name=ship`**

- 若 `matched = [ship_row, harbor_row]`：`split_matched_concepts_target_and_reference` → target `[ship]`，reference `[harbor]`。
- 输出：`format_grounding_appendix([ship])`（全字段 JSON） + `[Exclusion Prior]` 下仅 harbor 的 `exclusion_rule` 行。

**B. `vehicle on road`，`category_name=vehicle`**

- 若 `matched = [vehicle, road]` → target `[vehicle]`，reference `[road]`；vehicle 全量 JSON，road 仅 exclusion。

**C. `overpass near vehicle`，`category_name=overpass`**

- **情形 1**（库中仅命中 `vehicle`，未命中 `overpass`）：`target_concepts=[]`，`reference_concepts=[vehicle]` → 仅有 Exclusion Prior，**不出现**空 Target JSON 块。
- **情形 2**（库中同时命中 `overpass` 与 `vehicle`）：`overpass` 进 target（全量），`vehicle` 进 reference（exclusion only）——与用户文字「无 target」可能不一致，取决于**真实检索是否包含 overpass 行**。

---

## 4. 训练配置与脚本

### 4.1 `refaware_exclusion_only_28w` 一体化脚本（`run_refaware_exclusion_only_28w_full.sh`）

| 检查项 | 结果 |
|--------|------|
| `max_steps` | 默认 `280000`，且 `trainer_state.json` 记录 `max_steps=280000` |
| `--concept_refaware_prior True` | **有**（训练 deepspeed 命令行） |
| 未传 `--concept_match_strict True` | **是**（命令行未出现） |
| 正式训练未开 `debug_concept_refaware_prior` | **是**（未传该 flag） |
| `--seed` / `--data_seed` | 默认均为 **42** |
| 语义库 | `configs/concept_public_semantic_library_v2.json` |
| `dataset_name` | `rrsisd` |
| 独立 `OUTPUT_DIR` | `.../rrsisd_public_semantic_v2_refaware_exclusion_only_28w` |
| 误续训防护 | 若 `OUTPUT_DIR` 已存在 `checkpoint-*` 且 `RESUME_OK!=1`，脚本 **exit 1** |
| `GPU_SLOT` | 默认 `localhost:2`（与 7w baseline 脚本一致） |
| `MASTER_PORT` | 脚本 **HEAD** 为 `29500`；**工作区未提交**改为 `29522`（与 7w 默认一致性问题见 `code_diff_summary.txt`） |

### 4.2 训练参数对照表

**说明**：仓库内 `run_train_baseline_raw_vs_public_semantic_v2_7w.sh` 文件内 `MAX_STEPS` 默认值仍为 `280000`，但 **`trainer_state.json` 显示两组 7w 实际为 70000** → 下表「max_steps」以 **实际训练状态** 为准；其余超参从脚本默认值读取（与 refaware 脚本一致处标「一致」）。

| 参数 | baseline_raw_7w | public_semantic_v2_7w | refaware_exclusion_only_28w | 是否一致 | 备注 |
|------|-----------------|------------------------|-------------------------------|----------|------|
| max_steps（实际） | 70000 | 70000 | 280000 | **否** | 主要对照风险 |
| seed / data_seed | 42 / 42 | 42 / 42 | 42 / 42 | 是 | |
| learning_rate | 1e-4 | 1e-4 | 1e-4 | 是 | |
| per_device_train_batch_size | 1 | 1 | 1 | 是 | |
| gradient_accumulation_steps | 1 | 1 | 1 | 是 | |
| LoRA r / alpha / dropout | 8 / 16 / 0.05 | 同左 | 同左 | 是 | |
| deepspeed | `scripts/zero1.json` | 同左 | 同左 | 是 | |
| bf16 / tf32 | True / False | 同左 | 同左 | 是 | |
| save_steps / save_total_limit | 2000 / 2 | 同左 | 同左 | 是 | |
| logging_steps | 10 | 10 | 10 | 是 | |
| BASE_DATA_PATH（默认） | `.../huangziyi/data/RRSISD` | 同左 | 同左 | 是 | |
| 语义库路径 | **无** | `configs/concept_public_semantic_library_v2.json` | 同左 | v2/refaware 一致 | |
| `concept_refaware_prior` | **False**（未传） | False | **True** | 预期差异 | |
| `concept_match_strict` | False | False | False | 是 | |
| `MODEL_NAME_OR_PATH`（脚本默认） | bseg `baseline_standard-base_5w/merged_model` | 同左 | HEAD 同左；**工作区**默认改为 siglip11 | 见 §2 风险 | **已训练** LoRA 的 `adapter_config.json` 均为 bseg 路径 |

---

## 5. Merge 链路

### 5.1 脚本行为摘要（`run_refaware_exclusion_only_28w_full.sh`）

1. **`merge_lora_weights_and_save_hf_model.py`**：未被修改（git diff 空）；merge 函数仅传 `--model_path`（选中 ckpt）、vision towers、`--save_path`、`--lora_*`，**无** `concept_*` 参数。
2. **输入 checkpoint**：由 `pick_best_checkpoint` 选择；本地 `trainer_state.json` 显示 `best_model_checkpoint` = **`checkpoint-218000`**（在 `OUTPUT_DIR` 根目录）。
3. **选择策略优先级**：用户 `SELECTED_CHECKPOINT` → 根 `trainer_state.json` 的 `best_model_checkpoint` → 子目录 `trainer_state` → 按 checkpoint 内 eval 指标链 `eval_gIoU`… → 最后 **WARN fallback 最大 step**。
4. **输出目录**：`MERGED_DIR` 默认 `..._28w_merged_best`；本地存在 `config.json`（已证实）。
5. **merge 阶段 concept 相关 flag**：**无**（与需求一致）。

### 5.2 结构化结论

| 项目 | 值 |
|------|-----|
| merge 输入目录（训练输出根） | `/home/wangchengjun/huangziyi/reseg/output/source/rrsisd_public_semantic_v2_refaware_exclusion_only_28w` |
| selected checkpoint（trainer_state） | `.../checkpoint-218000` |
| selection strategy | **`OUTPUT_DIR/trainer_state.json` → `best_model_checkpoint`**（与脚本 `OK:root_state` 分支一致） |
| merge 输出目录 | `/home/wangchengjun/huangziyi/reseg/output/source/rrsisd_public_semantic_v2_refaware_exclusion_only_28w_merged_best` |
| 风险判断 | **低–中**：若曾手动删改 `trainer_state` 或 best 字段会改变选中 ckpt；当前文件与 `checkpoint-218000` 存在性自洽。 |

---

## 6. Eval / Test 链路

| 检查项 | 结论 |
|--------|------|
| `eval.py` 是否被修改 | **否**（git diff 空） |
| eval 是否使用 refaware merge 后模型 | **路径设计为是**：`run_eval_infer "${MERGED_DIR}"`；metrics 落在 `*_eval_best` 与 `merged_best` 共存 |
| eval 是否注入 semantic prior | **否**（`eval.py` 无 `concept_` 相关参数） |
| `concept_refaware_prior` / `concept_public_semantic_library` / `concept_match_strict` | eval 命令均未传 |
| test_results / metrics 是否独立目录 | refaware 使用 `..._28w_eval_best/test_results` 与独立 `rrsisd_test_metrics.json`，与 baseline/v2 根目录命名区分 |
| 误读其他实验结果风险 | **低**（路径分离；仍建议以绝对路径归档） |

**设计说明（专家常问）**：v2 / refaware 在 **训练时**改变 human prompt；**推理 eval** 回到无库 prompt，依赖权重内化学到的归纳偏置。故 eval 不传 prior **符合当前设计**，不是实现遗漏。

---

## 7. 三组指标对比

数据来源：`rrsisd_test_metrics.json`（详见 `experiment_metric_summary.csv`）。三组 **`num_samples=3480`**，`skipped_empty_gt=1`，`missing_pred=0`。

| 指标 | baseline_raw_7w | public_semantic_v2_7w | refaware_exclusion_only_28w | v2−baseline | refaware−baseline | refaware−v2 |
|------|-----------------|------------------------|-------------------------------|-------------|-------------------|-------------|
| mIoU / gIoU | 0.664690 | 0.669801 | 0.668313 | +0.005111 | +0.003623 | −0.001488 |
| oIoU | 0.782134 | 0.783064 | 0.780843 | +0.000930 | −0.001291 | −0.002221 |
| oIoU_percent | 78.213 | 78.306 | 78.084 | +0.093 | −0.129 | −0.222 |
| mDice | 0.742122 | 0.747022 | 0.744909 | +0.004900 | +0.002787 | −0.002113 |
| mPrecision | 0.754938 | 0.761665 | 0.757109 | +0.006727 | +0.002171 | −0.004556 |
| mRecall | 0.759295 | 0.761746 | 0.762407 | +0.002451 | +0.003112 | +0.000661 |

**解读**：差值均在 **千分位量级**；在步数不对照（7w vs 28w）前提下，**不支持**「refaware 带来稳健整体增益」的强结论。

---

## 8. 当前结果解释（克制）

1. **refaware_exclusion_only_28w** 在整体 test 指标上 **未** 相对 baseline / v2 呈现明显优势。
2. 三组曲线/数值在宏观上 **高度重叠** 的判断，与本地 JSON **一致**。
3. **不能**声称 refaware 显著优于 baseline 或 v2。
4. **不能**声称「语义库方向已失败」：当前仅说明 **该注入策略 + 该训练预算 + 无 prior 推理** 未转化为整体指标收益。
5. 更合理解释包括：整体均值掩盖 per-class 变化；训练期 rich prompt 与推理分布偏移；库覆盖/条目质量限制等。
6. 若继续：需 **baseline_28w / v2_28w** 对齐步数；**per-category** 与 **per-sample** 分析；复核 merge/eval 路径；冻结脚本默认基座。

---

## 9. 专家问答（Q1–Q15）

每个问题：**简短回答** | **证据** | **风险等级**（低/中/高）

| ID | 简答 | 证据 | 风险 |
|----|------|------|------|
| Q1 改模型结构？ | **否** | 无 model 目录 diff；LoRA 训练 | 低 |
| Q2 改 mask decoder？ | **否** | 无相关 Python diff；yaml 路径脚本一致 | 低 |
| Q3 改 loss？ | **否** | 未改 loss 计算路径 | 低 |
| Q4 改 `token_refer_id`？ | **否** | `dataset.py` 仍 `preprocess_referring_instruction(instruction)` | 低 |
| Q5 改 inference/eval？ | **否** | `eval.py` 无 diff、无 concept 参数 | 低 |
| Q6 改 merge？ | **否** | merge 脚本无 diff | 低 |
| Q7 改语义库 JSON？ | **否** | git diff 空 | 低 |
| Q8 训练是否启用 refaware？ | **是**（设计上） | 训练命令含 `--concept_refaware_prior True`；实现分支存在 | 低 |
| Q9 推理无 semantic prior 是否正常？ | **正常** | 设计一致；eval 无库参数 | 低 |
| Q10 可能训了但 eval 没用上？ | **当前证据不支持** | 存在 `merged_best` + `eval_best` metrics；adapter 基座路径一致 | 低 |
| Q11 可能 merge 错 ckpt？ | **当前以 trainer_state 为准风险低** | `best_model_checkpoint` → `checkpoint-218000` | 中（若人工改 state） |
| Q12 可能读错 test_results？ | **路径分离降风险** | 目录命名不同；仍需操作规范 | 低 |
| Q13 指标支持有效提升？ | **不支持** | §7 差值小 + 步数不对照 | 高（结论外推） |
| Q14 还能作为论文创新点？ | **需更强证据** | 当前整体指标不构成单独卖点 | 高 |
| Q15 下一步最该查？ | **步数对齐 + per-class/sample** | 控制 max_steps 与基座后再比较 | — |

---

## 10. 风险与未证实项

1. **7w vs 28w**：最大混淆因素；任何「refaware 无效/有效」结论都应先对齐训练步数。
2. **脚本默认基座漂移（未提交）**：与已训练 `adapter_config` 不一致；未来重训前必须统一环境变量。
3. **W&B 与本地 JSON**：本报告以 **本地 `rrsisd_test_metrics.json`** 为准；未再对 W&B 导出做数值交叉验证（可选增强）。

---

## 11. 下一步建议

1. 跑 **同 max_steps（建议 280k）** 的 `baseline_raw` 与 `public_semantic_v2`，再与 refaware 对比。
2. 输出 **per-category** mIoU / oIoU 表（可用现有 `eval_val_metrics` 或离线脚本，**不改**训练代码前提下新增只读分析亦可）。
3. 冻结 `run_refaware_exclusion_only_28w_full.sh` 默认 `MODEL_NAME_OR_PATH` 与 **实际** 7w 实验一致，或强制在文档中写「必须通过 env 显式传入」。
4. 对 refaware 抽 **小样本** 打开 `debug_concept_refaware_prior`（独立 debug 跑，非正式训练）以动态验证 §3.5 样例。

---

## 附录：交付物路径

| 文件 |
|------|
| `docs/REFAWARE_EXCLUSION_ONLY_EXPERT_REVIEW.md`（本文件） |
| `outputs/source/refaware_exclusion_only_expert_review/experiment_metric_summary.csv` |
| `outputs/source/refaware_exclusion_only_expert_review/experiment_path_summary.json` |
| `outputs/source/refaware_exclusion_only_expert_review/code_diff_summary.txt` |
