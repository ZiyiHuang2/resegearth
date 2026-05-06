0. 角色说明见同目录 `concept_semantic_library_pipeline_notes.md`（verified_7 为工程验证版，非最终 public 库）。
1. verified_7 包含 water / bridge / road / building / vehicle / ship / airport。
2. 该版本通过 concept_library_audit 与 concept_semantic_self_check。
3. water 属于 stuff/region 类。
4. 当前 water.forbidden_auto_infer_tokens 中保留部分结构形态词，是为了兼容当前 generator 的 visual_form_options 内容词覆盖校验。
5. 后续如引入 concept_type=stuff_region，可清理 water 的结构词 forbidden，使语义设计更干净。
6. 该版本用于验证语义库生成链路，不代表最终全量语义库。
