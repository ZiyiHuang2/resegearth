# Concept semantic library：角色与降级说明（pipeline）

## 1. `concept_semantic_library_verified_7.json`

- **定位**：工程链路验证产物（API 草稿 + validator + self-check 曾跑通的一版），**不是**最终面向 MLLM 的 public semantic library。
- **内容**：含完整 private 字段（如 `visual_form_options`、`forbidden_auto_infer_tokens`、`slot_guidance` 等），其中部分形态相关 forbidden 曾为迎合历史校验规则而保留，**不应**再视为「语义洁净的对外知识」。

## 2. `concept_semantic_library_draft_next_10.json`

- **定位**：API 草稿扩展集（next_10 seed），**不是**最终 public semantic library。
- **用途**：开发与审计、与旧 generator/validator 行为对照；**不**作为对外的极简语义源。

## 3. `concept_public_semantic_library_v0.json`（推荐）

- **定位**：手写、极简、**唯一推荐**进入 expression enhancement **MLLM user prompt** 的 public semantic 源。
- **条目结构**：每条仅 `concept` / `type` / `boundary`（英文一句），无 `visual_form_options`、无 `forbidden_auto_infer_tokens`、无 seed `generation_hint` 等。

## 4. 运行时策略（摘要）

- **MLLM**：优先加载 `--concept-public-semantic-library`；若与 `--concept-semantic-library`（完整库）同时传入，则 **prompt 用 public**，**post-check 仍可用完整库**（`matched_concepts_full`）。
- **Private / audit**：完整库中的 `visual_form_options`、`forbidden_auto_infer_tokens` 等仅作 **private/audit**，**不作为** MLLM 输入。

详见 `tools/rrsisd_explicitization_pipeline.py` 中 `retrieve_concept_semantics`、`build_llm_messages`、`concept_public_semantic_library_self_check`。
