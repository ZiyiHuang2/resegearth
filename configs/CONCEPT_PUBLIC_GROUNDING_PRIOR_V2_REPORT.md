# RRSIS-D public grounding prior library v2 — 任务报告

生成时间：2026-05-04（本地自检，无 API、未运行 `process_units_jsonl`、未训练、未生成 `enhanced_training.json`）。

## 1. v2 文件路径

- `resegearth+source/configs/concept_public_semantic_library_v2.json`
- 说明文档：`resegearth+source/configs/concept_public_semantic_library_v2_NOTES.md`

## 2. 17 个 concept 是否齐全

是。`concept_key` 集合与既有 `PUBLIC_SEMANTIC_LIBRARY_V17_KEYS` 一致（含 `bare land`、`parking lot` 等多词键）。

## 3. 每条字段是否正确

是。每条仅含：`concept`、`type`、`parent`、`visual_evidence`、`mask_scope`、`exclusion_rule`；`type` ∈ {`stuff`,`thing_or_facility`}；文本字段非空且长度 ≤ 180。

## 4. match_policy 是否存在并正确

是。与规范逐项比对：`word_boundary`、`continuous_phrase`、`longest_first_non_overlapping`、`child_suppresses_parent`、`inject_all_non_overlapping_matches`、`fallback=empty`。

## 5. parent 层级是否正确

是。仅 `river`、`lake` 的 `parent` 为 `"water"`；其余为 JSON `null`。

## 6. river / lake / water 检索是否互斥（子类抑制父类）

是。`find river` / `find lake` / `find water` 分别只注入对应子类或父类；`vehicles on the bridge over the river` 注入 `river` 而不注入 `water`。

## 7. 多概念共现是否按「全部注入」处理

是。贪心最长非重叠匹配后保留多概念；`exclusion_rule` 不合并。

## 8. riverbank 是否不误命中 river

是。整词边界下 `riverbank` 不匹配 `river`。

## 9. fallback 无匹配是否返回空

是。例如 `unknown target` → `matched_concepts == []`。

## 10. 是否没有 private/internal 字段名进入条目 JSON

结构自检扫描禁止字段名与禁用子串；当前库通过。

## 11. 是否没有固定结论式形态模板及绝对词

结构自检禁止 `always` / `must` / `definitely`（整词）及短语 `appears as`、`is always`、`must be`；当前库通过。

## 12. prompt_leakage_self_check 是否通过（v2 路径）

是。对默认 `find water near bridge` + v2 库：`prompt_leakage_self_check_passed: true`（对 grounding matched JSON 跳过形态污染词子串检查，与 v2 设计一致）。

## 13. concept_public_context_length_check 是否通过（v2 路径）

是。`concept_public_context_length_check_passed: true`（含字符预算与每字段 ≤180 检查）。

## 14. 是否未调用 API

是。仅执行：`concept_public_grounding_prior_structure_check`、`concept_public_grounding_prior_retrieval_check`（内部调用上述只读检查），无 HTTP LLM 调用。

## 15. 是否未运行 process_units_jsonl

是。未执行该子命令。

## 代码变更摘要

- `retrieve_concept_semantics`：识别 `library_kind == public_grounding_prior`，应用别名扩展（单 token 简单复数）、父子抑制、注入六字段公共行。
- `build_llm_messages`：根据 matched 形态切换「grounding prior」与「minimal boundary」说明文案。
- 新 CLI：`concept_public_grounding_prior_structure_check`、`concept_public_grounding_prior_retrieval_check`。
