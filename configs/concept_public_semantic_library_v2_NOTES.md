# concept_public_semantic_library_v2（public_grounding_prior）

1. **覆盖范围**：当前 JSON 中的 17 个 `concept_key` 是 RRSIS-D 语义实验候选类别。若真实数据或表达中出现未覆盖类别，检索结果为 `matched_concepts = []`，**无**通用 fallback 注入。

2. **与 v0 边界表的分工**：v2 不提供「类别名替换防护」句式；主目标是 **pixel / mask 层面的 grounding prior**（`visual_evidence`、`mask_scope`、`exclusion_rule`），而非改写规则。

3. **标签与文本管线**：若后续中间步骤会**显式生成**文本标签或规范化类别名，应**另行**增加 label-preservation / 防替换护栏；v2 本身不承担该职责。

4. **形态词**：允许在 `visual_evidence` 等字段中使用形态或纹理词作为**辅助**视觉线索，但禁止固定结论式模板（见库内自检与 JSON 约束）。

5. **多概念共现**：命中多个不重叠概念时 **全部注入** `matched_concepts`；**不合并** `exclusion_rule`，**不做**冲突推理；长度由 `concept_public_context_length_check` 与上游 prompt 预算约束。

6. **匹配策略**：见文件内 `match_policy`；实现与 `retrieve_concept_semantics` 一致：整词/连续短语、`longest_first_non_overlapping`、`child_suppresses_parent`（river/lake 命中时抑制 water）、无命中则空列表。
