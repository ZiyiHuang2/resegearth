# 训练侧接入 `concept_public_semantic_library_v2` — 报告

## 任务一：链路检查结论（书面回答）

1. **`train.py` 原先是否有 `--concept-public-semantic-library`**  
   **否。** 本次在 `DataArguments` 中新增 **`--concept_public_semantic_library`**（HuggingFace 风格下划线参数名；等价于命令行 `--concept_public_semantic_library`）。

2. **与 `process_units_jsonl` 的关系**  
   `process_units_jsonl` 的同名参数仅作用于离线增强流水线；**训练脚本此前未读取该 JSON**，二者已独立确认。

3. **训练样本文本来源**  
   `RRSISDDataset.__getitem__` 中 `instruction` 来自 refs 的 **`sentences[0].sent`**，否则 **`raw`**，再否则回退到类别模板句；**不是** `enhanced_training` / `enhanced_expression`（本任务未生成该数据）。

4. **prompt 拼接位置**  
   在 **`resegearth+tgi/segearth_r2/datasets/dataset.py`** 的 **`RRSISDDataset.__getitem__`**：构造 `sources` → **`RS_Base_Dataset.preprocess_llama2`** 调用 `conversation_lib.default_conversation.get_prompt()`（见 `segearth_r2/model/mipha/conversation.py`）→ **`tokenizer_special_tokens`** 将 `<image>` / `<refer>` 转为特殊 index。

5. **labels / loss mask 构造**  
   同一 **`preprocess_llama2`**：对 human 侧（含 user 指令与占位符）将对应 **`labels` 置为 `IGNORE_INDEX`（-100）**；assistant 侧保留可训练 token。多模态阶段在 **`llava_phi.SegEarthR2.prepare_inputs_labels_for_multimodal`** → **`concat_image_seg_cls_embeds`**：图像展开位置与 **`<refer>` 展开为 refer 嵌入序列** 的片段 **`labels` 均为 -100**；普通文本 chunk 携带 `preprocess_llama2` 得到的 label 切片。

6. **v2 prior 最稳注入层**  
   **Dataset 层（`RRSISDDataset`）在 tokenization 之前写入 human 字符串**，与现有 “user 文本 + 特殊 token + assistant” 管线一致；**不**改 `llava_trainer` / mask decoder 数学，**不**改 `[SEG]` 在 assistant 中的角色。

---

## 任务二～四：实现摘要

- **检索键**：`retrieve_matched_public_grounding` 使用 **与数据相同的 `instruction`（优先 sent 的 raw 语义）**，未使用任何 enhanced 字段。
- **注入侧**：仅 **human / user 拼接串**；**非** system；**非** assistant。
- **顺序（v2 开启时）**：`expression 文本` →（若有命中）**grounding JSON 附录** → `<|vision_bos|> <image> ...` → `<refer>` → assistant。满足「**在可见 raw 文本之后、image token 之前**」。
- **无匹配**：`format_grounding_appendix` 返回空，不插入 prior 块。
- **多概念 / 父子抑制**：沿用 `resegearth+source` 中 `retrieve_concept_semantics` 的 v2 逻辑。
- **`token_refer_id`（与 baseline 一致）**：**baseline 与 v2 均使用** `RRSISDDataset.preprocess_referring_instruction(instruction)`，即 **instruction 的 tokenizer ids + `[SEG]`**，与原始 SegEarth-R2 RRSIS-D 行为一致；**仅 v2 在 human 串中额外注入** grounding JSON（可能同时出现明文 instruction，与 refer 嵌入中的词面重复属于有意保留的原始 refer 策略）。
- **eval / inference / merge**：**未修改**。

---

## 任务三：loss mask

- **v2 JSON 与引导说明**位于 **human 文本**中 → `preprocess_llama2` 将其划入 **masked user 区** → **`labels = -100`**（与原有 user instruction 一致）。
- **图像 token、refer 展开段**：`concat_image_seg_cls_embeds` 中 **`labels` 仍为 -100**（原逻辑不变）。
- **assistant `[SEG]`**：仍在 **gpt** 轮次中参与训练；**mask decoder 路径未改**。

> 说明：若未来改动 `preprocess_llama2` 的 human/assistant 切分规则，需重新核对 v2 块是否仍落在 human 半轮。

---

## 任务五：Smoke 脚本

- 路径：`resegearth+tgi/tools/debug_train_prompt_with_v2.py`  
- 依赖：`conda` 环境 **`reseg`**（含 `torch` / `transformers`）。  
- 功能：baseline / v2 **human** 对比、unified diff、`matched_concepts`、**`token_refer_id` 两侧一致性**、`labels != -100` 计数、`visual_evidence` 子词 span 是否全 `-100`、`[SEG]` 是否在 assistant 可训练区、私域字段名扫描。  
- **已执行（不训练）**：  
  `conda run -n reseg python tools/debug_train_prompt_with_v2.py --help`  
  `conda run -n reseg python tools/debug_train_prompt_with_v2.py --model-name-or-path ... --library-v2 ...`（4 条样例：water / river / lake / multi）。

---

## 任务六：训练脚本

- `resegearth+tgi/run_train_baseline_raw.sh`：不传 `--concept_public_semantic_library`，`output_dir=.../checkpoints/rrsisd_baseline_raw`。  
- `resegearth+tgi/run_train_public_semantic_v2.sh`：仅多 `--concept_public_semantic_library` 指向 v2 JSON，`output_dir=.../checkpoints/rrsisd_public_semantic_v2`。  
- 其余超参块与 `train_mstva_loss_w01_7w.sh` 中 deepspeed 段 **对齐**（含 `lora_r`、MSTVA、save_steps 等）；**未**默认附带 `--report_to wandb`（与当前仓库内该段一致的可删项）。

---

## 任务七：验收表（13 项）

| # | 项 | 状态 |
|---|----|------|
| 1 | train 接口已接通 | 是（`DataArguments.concept_public_semantic_library`） |
| 2 | 变更文件 | `train.py`、`datasets/dataset.py`、`utils/concept_public_grounding_train.py`、`tools/debug_train_prompt_with_v2.py`、`run_train_*.sh`、本文档 |
| 3 | 注入函数 | `build_rrsisd_supervised_human_value` + `RRSISDDataset.__getitem__` |
| 4 | user 内 raw 后、image 前 | 是（v2 布局） |
| 5 | 检索基于 raw/sent | 是（`instruction` 字段） |
| 6 | prior 不在 assistant | 是 |
| 7 | prior tokens `labels=-100` | 是（human 区 + 展开规则） |
| 8 | `[SEG]` / mask decoder 未变 | 是（未改 `llava_phi` 拼接与 loss 结构） |
| 9 | inference/eval 未改 | 是 |
| 10 | 未调 API | 是 |
| 11 | 未跑 `process_units_jsonl` | 是 |
| 12 | 未生成 `enhanced_training.json` | 是 |
| 13 | 两脚本除 v2 与 output 外一致 | 是（`MASTER_PORT` 默认已对齐；见下方 `diff -u`） |

---

## 最终验收声明（必须项）

1. **v2 prior 仅在 user/human 输入侧**：通过 `build_rrsisd_supervised_human_value` 写入 human 串；**不**写入 gpt/assistant 的 `value`。  
2. **v2 prior 不在 assistant target**：gpt 轮仍为 `\n[SEG]`；prior JSON **仅**出现在 human。  
3. **v2 prior 对应 token 的 `labels` 均为 -100**：`preprocess_llama2` 将 human 侧 mask；`visual_evidence` 子词 span 经验证 **全为 -100**（见 smoke 输出）。  
4. **baseline 与 v2 的 `token_refer_id` 策略一致**：均为 **`encode(instruction) + [SEG]`**；**已移除**此前 v2 专用的「仅 `[SEG]`」分支。  
5. **两训练脚本除 v2 参数与 `OUTPUT_DIR` 外 CLI 一致**：`diff -u` 仅体现 **文件头注释、`CONCEPT_LIB`、`OUTPUT_DIR`、多一行 `--concept_public_semantic_library`**（见下）。

### `diff -u run_train_baseline_raw.sh run_train_public_semantic_v2.sh`

```diff
--- run_train_baseline_raw.sh
+++ run_train_public_semantic_v2.sh
@@ -1,6 +1,6 @@
-# RRSIS-D train — baseline_raw (no --concept_public_semantic_library).
-# All other CLI hyperparameters match run_train_public_semantic_v2.sh (same MASTER_PORT default).
+# RRSIS-D train — public_semantic_v2 (adds --concept_public_semantic_library only).
+# All other CLI hyperparameters match run_train_baseline_raw.sh (same MASTER_PORT default).
@@ -9,6 +9,8 @@
+CONCEPT_LIB="/home/wangchengjun/huangziyi/reseg/resegearth+source/configs/concept_public_semantic_library_v2.json"
@@ -16,7 +18,7 @@
-OUTPUT_DIR=".../rrsisd_baseline_raw"
+OUTPUT_DIR=".../rrsisd_public_semantic_v2"
@@ -64,6 +66,7 @@
+  --concept_public_semantic_library "${CONCEPT_LIB}" \
```

（完整路径以仓库内脚本为准。）

### Smoke 摘要（`conda run -n reseg`）

- **`token_refer_id`**：各 case 下 baseline 与 v2 **张量相等**（例如 `find water` 为长度 3 的 id 序列）。  
- **`labels != -100` 计数**：baseline 与 v2 **均为 5**（仅 assistant / `[SEG]` 等可训练片段；v2 仅增加 human 长度，不增加可训练 token 数）。  
- **`[SEG]`**：仍在 **gpt 轮**可训练 decode 片段中。  
- **未**：训练、API、`process_units_jsonl`、`enhanced_training.json`、inference/eval 修改。

---

## 关于 `run_train_merge_test.sh`

仓库内 **`resegearth+bseg/run_train_merge_test.sh`** 等脚本指向 **另一套 `REPO_DIR`（SegEarth-R2）**，与 **`resegearth+tgi`** 当前 `train.py` **不是同一条训练入口**；本次改动集中在 **`resegearth+tgi/segearth_r2/train/train.py` + 同树数据集**。

---

## Inference 说明（任务四）

是否在 **推理 / 评测** 注入 v2 prior 作为 **独立实验因子**，本次 **未实现**；需后续在 eval 侧复用同一 `build_rrsisd_supervised_human_value` 思路时再开分支。
