# `resegearth+source`：训练侧接入 `concept_public_semantic_library_v2` — 报告

## 一、链路结论

1. **CLI 参数**  
   `segearth_r2/train/train.py` 的 `DataArguments` 提供 **`--concept_public_semantic_library`**（HF 下划线风格）。

2. **与 `process_units_jsonl` 的关系**  
   离线 `process_units_jsonl` 与训练读取的 v2 JSON **相互独立**；训练不依赖 `enhanced_training.json`。

3. **训练样本文本来源**  
   `RRSISDDataset.__getitem__` 中 `instruction` 来自 **`sentences[0].sent`**，否则 **`raw`**，再否则类别模板；**不使用** `enhanced_expression`。

4. **prompt 与 tokenization**  
   `RRSISDDataset.__getitem__` 构造 `sources` → `RS_Base_Dataset.preprocess_llama2` → `tokenizer_special_tokens`（`<image>` / `<refer>` 等）。

5. **labels / loss mask**  
   `preprocess_llama2` 将 **human 整段**（含 v2 附录）对应 **`labels = IGNORE_INDEX`（-100）**；**gpt** 侧保留 **`[SEG]`** 等可训练 token。多模态展开阶段逻辑未改。

6. **注入层**  
   **`segearth_r2/utils/concept_public_grounding_train.build_rrsisd_supervised_human_value`**，由 **`RRSISDDataset.__getitem__`** 写入 **human** 的 `value`；检索经 **`tools/rrsisd_explicitization_pipeline.py`**（`retrieve_matched_public_grounding`）。

---

## 二、实现摘要

- **检索键**：与样本相同的 **`instruction`**（sent/raw）。
- **注入侧**：仅 **human**；非 system；非 assistant。
- **顺序（有 v2 库路径时）**：说明 + **raw expression** →（若有）**grounding JSON** → **`<|vision_bos|> <image> ...`** → **`<refer> <|assistant|>`** → gpt 仍为 **`\\n[SEG]`**。
- **无匹配**：不插入 JSON 附录块（或仅 expression + image 布局，由 `format_grounding_appendix` 空串控制）。
- **`token_refer_id`（重要）**  
  **baseline_raw（不传库）与 public_semantic_v2（传库）均使用同一策略**：**`preprocess_referring_instruction(instruction)`**，即 **`encode(instruction) + [SEG]`**（与历史 RRSIS-D 行为一致）。**v2 仅多注入 human 内的 grounding prior**，**不存在**「v2 下 `token_refer_id` 仅 `[SEG]`」的特殊分支（若旧文档曾写此条，视为作废）。

---

## 三、Smoke / Debug 工具（source）

- **路径**：`tools/debug_train_prompt_with_v2.py`  
- **依赖**：`conda run -n reseg`（或已安装 `torch` / `transformers` 的环境）。  
- **命令**：
  ```bash
  cd /home/wangchengjun/huangziyi/reseg/resegearth+source
  conda run -n reseg python tools/debug_train_prompt_with_v2.py --help
  conda run -n reseg python tools/debug_train_prompt_with_v2.py
  ```
  默认 `--model-name-or-path` 指向常见 merged 模型目录；`--library-v2` 默认 `configs/concept_public_semantic_library_v2.json`。可用环境变量 `DEBUG_MODEL_PATH`、`DEBUG_LIBRARY_V2` 覆盖。

---

## 四、本地验收记录（2026-05-04，`conda run -n reseg`）

在 **`resegearth+source`** 下执行完整 debug，**`ALL CASES OK: True`**。

| 检查项 | 结果 |
|--------|------|
| 四类样本 water / river / lake / multi | 均 **`CASE … overall OK: True`** |
| baseline vs v2 **`token_refer_id`** | 各 case **`torch.equal`，一致**（water/river/lake 长度 3；multi 长度 9） |
| **`labels != -100` 数量** | baseline / v2 **均为 5**（可训区一致） |
| **`visual_evidence` / `mask_scope` / `exclusion_rule`** 子串 span | **全部为 -100** |
| **`Category-level` 至首个可训 token 前** 区间 | **全部为 -100** |
| **`[SEG]`** | 在 **assistant 可训 decode** 中 |
| **v2 prior 仅在 human** | gpt 分支仅为 **`\\n[SEG]`**；prior 关键词未出现在 gpt 串 |
| **private/internal 子串扫描** | **无命中** |
| **river / lake / water 互斥**（matched **concept** 集合） | water 无 river/lake；river/lake 的 matched 中 **无 `water` 概念行**（JSON 内 `parent: water` 不计为命中 water 概念） |
| **multi** | 命中 **vehicle、bridge、river**；**未**将 **water** 作为独立 matched concept |

---

## 五、涉及文件（source 树）

| 文件 | 作用 |
|------|------|
| `segearth_r2/train/train.py` | `DataArguments.concept_public_semantic_library` |
| `segearth_r2/datasets/dataset.py` | `RRSISDDataset` 调用 `build_rrsisd_supervised_human_value`；统一 `token_refer_id` |
| `segearth_r2/utils/concept_public_grounding_train.py` | 加载库、检索、拼 human 串 |
| `configs/concept_public_semantic_library_v2.json` | v2 语义库 |
| `tools/debug_train_prompt_with_v2.py` | 无训练本地验收 |

---

## 六、Inference / Eval

**未**在推理或评测脚本中注入 v2；若需同一 prior，应在 eval 侧单独复用 `build_rrsisd_supervised_human_value` 思路。

---

## 七、禁止项确认（本任务范围）

未执行：训练、API、`process_units_jsonl`、`enhanced_training.json` 生成；未改 inference/eval 代码路径。
