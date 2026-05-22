# Semantic Library and Code Path Audit

**审计日期**：2026-05-12  
**约束**：只读；未修改任何 `.py` / `.json` / `.sh` / checkpoint；未训练、merge、全量 eval。  
**产物目录**：`outputs/source/semantic_library_audit/`（JSON/CSV）与本文件。

---

## 0. 一页结论

| 问题 | 结论 |
|------|------|
| 当前能否默认语义库没问题？ | **不能默认**。库在工程上被一致加载且 schema 闭合，但与 RRSIS-D **20 类 GT 名称不对齐**（15 个 GT 类无库条目），且 **reference 的 exclusion 文本信息量大**，存在内容与匹配层面的风险。 |
| 最大风险是内容、匹配、projector，还是训练混杂？ | **证据排序（由高到低）**：① **SEG_token_projector 瓶颈**（Phase 2：`mean_signal_survival≈0.22`）；② **Probe B 的 7w vs 28w 混杂**（不可因果归因）；③ **语义库与 GT/语料对齐缺口 + exclusion 过详**（中等）。 |
| 是否建议立刻改模型结构？ | **否**。无证据表明需改架构；应先做 **同预算 merged 对照** 再决定。 |
| 是否建议先改语义库？ | **可作为并行项，但不宜越过对照实验单独承担主责**。优先缩短/弱化 reference exclusion 的「视觉枚举」，并为高频无库 GT（如 overpass）设计 **alias 或独立条目**（仅建议，本次未改 JSON）。 |
| 是否建议先做同预算 baseline_28w？ | **是**。在解释 refaware 之前，需要 **baseline_28w / v2_28w / refaware_28w** 三方 merged 控制，否则 Probe B 类退化无法归因。 |

**最终判断（必选一项）**：**B** — 语义库内容存在**轻微到中等**风险，但在当前证据链上**尚不足以排他地认定为主因**；主因仍由 projector 数值与训练混杂解释更稳。

---

## 1. 审计范围

- **语义库**：`configs/concept_public_semantic_library_v2.json`（只读）
- **代码链路**：`concept_public_grounding_train.py`、`tools/rrsisd_explicitization_pipeline.py`、`segearth_r2/datasets/dataset.py`、`segearth_r2/train/train.py`、`tools/probe_refaware_seg_hidden_effect.py`（只读 grep/read）
- **数据类别名**：`/home/wangchengjun/huangziyi/data/RRSISD/rrsisd/instances.json` → `categories[].name`
- **先验命中分析**：`outputs/source/public_semantic_v2_prior_hit_analysis/per_sample_prior_hit_diff.jsonl`
- **100-sample 探针**：`outputs/source/refaware_hidden_probe_100/*`
- **未找到**：`/home/wangchengjun/huangziyi/data/RRSISD` 下 **`refs(unc).p`**（glob 无匹配）；类别名以 `instances.json` 为准。

---

## 2. 代码链路审计

### 2.1 semantic library 在哪里加载？

| 环节 | 路径 | 函数 | 结论 |
|------|------|------|------|
| 检索入口 | `segearth_r2/utils/concept_public_grounding_train.py` | `retrieve_matched_public_grounding` 调用 `_load_pipeline()` → `load_concept_public_semantic_library(library_path)` | **风险：低** — 路径由调用方传入，无隐式硬编码文件名（训练脚本通常传 CLI 的 `concept_public_semantic_library`）。 |

```38:43:segearth_r2/utils/concept_public_grounding_train.py
def retrieve_matched_public_grounding(raw_expression: str, library_path: str) -> List[Dict[str, Any]]:
    mod = _load_pipeline()
    lib = mod.load_concept_public_semantic_library(library_path)
    ctx = mod.retrieve_concept_semantics(raw_expression, lib, private_library_for_audit=None)
    rows = ctx.get("matched_concepts") or []
    return [r for r in rows if isinstance(r, dict)]
```

底层 JSON 读取在 `tools/rrsisd_explicitization_pipeline.py`：

```812:816:tools/rrsisd_explicitization_pipeline.py
def load_concept_public_semantic_library(path: str) -> Dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("concept public semantic library must be a JSON object.")
    return payload
```

### 2.2 训练 v2 / refaware 是否使用同一 JSON？

- **训练侧**：`DataArguments.concept_public_semantic_library` 为 **Optional[str] 路径**（`segearth_r2/train/train.py`），具体 JSON 由启动命令决定；**代码不强制** v2 与 refaware 使用同一物理文件。  
- **实践**：若两次训练传入同一 `--concept_public_semantic_library` 路径，则一致；**本次审计无法读取历史 shell 命令**，只能结论：**设计上可分叉，默认依赖实验脚本传参**。

```57:61:segearth_r2/train/train.py
    concept_public_semantic_library: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional path to concept_public_semantic_library_v2.json; RRSIS-D injects matched grounding priors in the human message (expression first, then priors, then image, then refer)."
        },
    )
```

### 2.3 `retrieve_matched_public_grounding` 匹配规则？

委托 `retrieve_concept_semantics`（`public_grounding_prior` 分支）：

- 对每个 `concepts` 条目构建 **alias 列表**（concept_key + cfg["concept"] + 英文简单复数规则），在原始 expression 上做 **词边界正则** 匹配 `_find_phrase_matches`。  
- **候选排序**：`(-length, start)` — **最长短语优先**。  
- **非重叠占用**：`occupied` 位图贪心选取。  
- **父子抑制**：`library_kind == public_grounding_prior` 时 `_suppress_parent_grounding_matches` 去掉与已匹配子概念同时出现的父概念（如 river/lake 命中时抑制 water）。

```1416:1477:tools/rrsisd_explicitization_pipeline.py
def retrieve_concept_semantics(
    raw_expression: str,
    library: Dict[str, Any],
    *,
    private_library_for_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ...
    candidates.sort(key=lambda x: (-x["length"], x["start"]))
    occupied = [False] * max(len(raw_expression), 1)
    picked: List[Dict[str, Any]] = []
    for c in candidates:
        s, e = c["start"], c["end"]
        if any(occupied[i] for i in range(s, e)):
            continue
        picked.append(c)
        for i in range(s, e):
            occupied[i] = True

    if is_public_grounding_prior_library(library):
        picked = _suppress_parent_grounding_matches(picked, concepts)
```

```1078:1082:tools/rrsisd_explicitization_pipeline.py
def _find_phrase_matches(text: str, phrase: str) -> List[Tuple[int, int]]:
    p = _phrase_pattern(phrase)
    if p is None:
        return []
    return [(m.start(), m.end()) for m in p.finditer(text)]
```

**风险：中** — 匹配完全依赖 **英文字面** 与 **词边界**；RRSIS-D 表达若用词与 `concept` 字符串不一致，则 **零命中**。

### 2.4 refaware 如何 split target / reference？

```70:90:segearth_r2/utils/concept_public_grounding_train.py
def split_matched_concepts_target_and_reference(
    matched_concepts: List[Any],
    category_name: Optional[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ...
    want = normalize_concept_name(category_name)
    ...
        if want and normalize_concept_name(_concept_label_from_matched_row(row)) == want:
            target_concepts.append(row)
        else:
            reference_concepts.append(row)
```

**结论**：与 **GT `category_name` 归一化相等** 的 matched 行 → **target**；其余 → **reference**。若 `category_name` 为空，则 **全部为 reference**（见 docstring）。

### 2.5 reference concept 是否只用 exclusion_rule？

```124:144:segearth_r2/utils/concept_public_grounding_train.py
def format_reference_exclusion_prior_section(reference_concepts: List[Dict[str, Any]]) -> str:
    """
    Minimal exclusion block for reference concepts only (no visual_evidence / mask_scope).
    """
    ...
        rule = str(row.get("exclusion_rule") or "").strip()
        if rule:
            lines.append(f"- {name}: {rule}")
        else:
            lines.append(f"- {name}: Exclude {name}.")
```

**结论**：reference 块 **不注入** `visual_evidence` / `mask_scope`，**仅** `exclusion_rule`（或缺省句）。**风险：中** — exclusion 原文仍可能很长、列举多类视觉对象。

### 2.6 v2（非 refaware）是否仍完整注入六字段？

`build_rrsisd_supervised_human_value` 使用 `format_grounding_appendix`，将 matched 行 **JSON 序列化** 注入；对 `public_grounding_prior` 库，public row 由 `build_grounding_prior_public_row` 构造，含 **concept/type/parent/visual_evidence/mask_scope/exclusion_rule**。

```147:182:segearth_r2/utils/concept_public_grounding_train.py
def build_rrsisd_refaware_exclusion_only_human_value(...):
    ...
    if target_concepts:
        tgt = format_grounding_appendix(target_concepts)
```

```970:987:tools/rrsisd_explicitization_pipeline.py
def build_grounding_prior_public_row(concept_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    ...
    out: Dict[str, Any] = {
        "concept": str(cfg.get("concept", concept_key)).strip(),
        "type": str(cfg.get("type", "")).strip(),
        "parent": parent_out,
        "visual_evidence": str(cfg.get("visual_evidence", "")).strip(),
        "mask_scope": str(cfg.get("mask_scope", "")).strip(),
        "exclusion_rule": str(cfg.get("exclusion_rule", "")).strip(),
    }
```

### 2.7 `token_refer_id` 是否仍来自原始 instruction？

**是**。`human_value` 可含 prior，但 `token_refer_id` 单独由 **`preprocess_referring_instruction(instruction)`** 编码 **原始 expression + [SEG]**，与 prior 块无关。

```405:456:segearth_r2/datasets/dataset.py
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
        token_refer_id = self.preprocess_referring_instruction(instruction)
```

### 2.8 prior token 是否仍 `labels=-100`（IGNORE_INDEX）？

`preprocess_llama2` 将 **human 轮中「指令部分」** 对应 token 的 `labels` 置为 `IGNORE_INDEX`（常量 **-100**），仅监督 assistant 段。

```125:203:segearth_r2/datasets/dataset.py
        targets = input_ids.clone()
        ...
                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
```

```7:7:segearth_r2/utils/constants.py
IGNORE_INDEX = -100
```

**结论**：prior 插入在 human `sentence["value"]` 内，与 expression 同属 **被 mask 的监督区域**（不直接对 prior 文本做 LM loss）。**风险：低**（与既有 LLaVA 式训练一致）。

### 2.9 prompt 中 prior 插入位置是否固定？

`build_rrsisd_refaware_exclusion_only_human_value` / `build_rrsisd_supervised_human_value` 固定顺序：

1. 头部模板 +「按 expression 做 Reasoning Segmentation」  
2. **原始 expression**  
3. **appendix**（v2 JSON 和/或 `[Exclusion Prior]`）  
4. `<|vision_bos|> <image> <|vision_eos|>\n<refer> <|assistant|>`

**结论**：**固定**在 **图像 token 之前**、expression 之后。

### 2.10 100-sample probe 是否复用训练同源的 refaware 构造函数？

**是**。`tools/probe_refaware_seg_hidden_effect.py` 中 `build_refaware_fields` 直接调用 `build_rrsisd_refaware_exclusion_only_human_value`（与 `dataset.py` 同源 utils）。

（见探针内 `build_refaware_fields` → `build_rrsisd_refaware_exclusion_only_human_value` 的调用链；与上文 `dataset.py` 405–415 行逻辑一致。）

---

## 3. 语义库 Schema 审计

详见 **`outputs/source/semantic_library_audit/semantic_library_schema_audit.json`**。

摘要：

- **`concept_count`**：17  
- **顶层**：`library_kind=public_grounding_prior`，`version=public_semantic_v2`，**存在** `match_policy`（与 pipeline 期望一致）。  
- **每个条目键集合**：与 `GROUNDING_PRIOR_ENTRY_KEYS` 一致（`concept,type,parent,visual_evidence,mask_scope,exclusion_rule`）。  
- **父子**：仅 `river`/`lake` → `parent: water`；其余 `parent: null`。  
- **重复 / key 不一致**：当前 JSON **无** duplicate concept 字符串、**无** concept_key 与 `cfg["concept"]` 文本不一致项（见 schema JSON 空列表）。  
- **超长字段**：未发现单字段 `>800` 字符（`long_fields_over_800_chars` 为空）；pipeline 另有 **单行 JSON 900 字符**上限（`GROUNDING_PRIOR_ROW_JSON_MAX_CHARS`），当前库未触发。

---

## 4. Concept 与 RRSIS-D Category 对齐

- **RRSIS-D `instances.json`**：`categories` 共 **20** 个 `name`。  
- **语义库**：**17** 个 `concepts` 条目。

**无库条目的 GT 类别（15 个）**（归一化比较）：

`Expressway-Service-area`, `Expressway-toll-station`, `airplane`, `baseballfield`, `basketballcourt`, `chimney`, `dam`, `golffield`, `groundtrackfield`, `overpass`, `stadium`, `storagetank`, `tenniscourt`, `trainstation`, `windmill`

**库中有、但不在 20 类 GT 精确名中（典型「语料触发、非 GT 名」）**：

`water`, `vegetation`, `bare land`, `farmland`, `grassland`, `forest`, `river`, `lake`, `parking lot`, `playground` 等（完整列表见 `semantic_library_audit_summary.json` 的 `library_concepts_not_in_dataset_gt_names`）。

对齐表：**`outputs/source/semantic_library_audit/concept_category_alignment.csv`**。

**高风险库外类别（用户关心）**：`overpass`、`chimney`、`golffield`、`basketballcourt` 等 **无库条目** → 表达中若匹配到 **vehicle/bridge/road** 等，则在 refaware 下 **几乎总是 reference**，靠 **exclusion 文本** 影响模型，**无法**提供与 GT 名对齐的「目标先验 JSON」。

---

## 5. Concept 匹配分布（prior-hit jsonl + 100-sample probe）

**`per_sample_prior_hit_diff.jsonl`（3480 行）中 `matched_concepts` 实际出现过的库概念仅 5 个标签**（小写计数）：

| concept | jsonl 命中次数 |
|---------|----------------|
| vehicle | 626 |
| bridge | 290 |
| airport | 219 |
| ship | 190 |
| harbor | 32 |

其余库概念在该 jsonl 的 `matched_concepts` 字段中 **计数为 0**（不代表全数据集永远为零，但说明 **该分析子集/语料切片** 几乎不触发这些词）。

逐概念统计（含 probe 100 共现、Probe A/B 均值）：**`outputs/source/semantic_library_audit/concept_match_distribution.csv`**。

---

## 6. 逐 concept 内容质量审计

启发式评分（1–5）与 `recommended_action`：**`outputs/source/semantic_library_audit/concept_quality_audit.csv`**。

共性观察：

- **`exclusion_rule`** 普遍 **多逗号列举** → 在 reference 模式下仍构成 **强负向视觉列表**（`exclusion_rule_risk` 常标 `high` → 建议 `simplify_exclusion_rule` 或 `mark_reference_high_risk`）。  
- **`mask_scope`** 对 **airport / harbor** 等允许「整设施 / 庇护水域」——语义上 **偏宽**，与实例级分割易产生张力。

---

## 7. 高风险 concept（用户指定 10 项）

**`outputs/source/semantic_library_audit/risky_concepts_report.csv`**

与 100-sample probe 交叉要点：

- **vehicle / bridge**：在 probe 子集中 **reference_count ≫ target_count**（vehicle 20 vs 11），且 **mean_probe_b_mask_iou** 较低（≈0.66 / 0.65），**severe 共现高**。  
- **airport**：全为 target_prior（50/50），但 **severe+moderate 仍高** → 与「仅语义库有错」不完全一致，**更支持权重差异 + projector + 宽 mask_scope 叠加**。  
- **ship / harbor**：probe 中 mean Probe A IoU 仍高，但 **Probe B mean IoU 尤其 ship 低** → **28w 混杂**信号强。

---

## 8. 与 100-sample Probe 结果结合

数据来源：`outputs/source/refaware_hidden_probe_100/hidden_probe_summary.json`。

1. **severe/moderate 是否集中在某些 concept？**  
   **共现层面**：`vehicle`、`bridge`、`ship` 在 degradation 样本中共现次数高（见 `concept_match_distribution.csv`）。**airport** 同时有大量 target prior，仍有高退化计数 → **不能单用「库错」解释**。

2. **是否集中在 prior_group？**  
   `degradation_by_prior_group`：**target_prior**（severe 12, moderate 11）与 **mixed**（severe 8）均显著；**reference_only** 也有 severe 6。**不是**仅 reference 组问题。

3. **signal_survival 是否因 concept 不同？**  
   在 100 样本上：`airport` mean ≈0.210，`ship` ≈0.183，`vehicle` ≈0.230，`bridge` ≈0.247 — **有差异但同量级**，与「projector 普遍压缩」一致。

4. **target_prior 是否比 reference_only 更能影响 mask？**  
   Probe A mean mask IoU：`target_prior`≈0.927 vs `reference_only`≈0.903（summary 中 `prior_group_summary`）。**略高**并不等价「影响更小」；且 airport 在 target_prior 下仍有明显退化 → **不能简化因果**。

5. **airport 退化与 prior 内容 vs 28w？**  
   **证据分裂**：Probe A mask IoU 仍 ~0.93（先验未「搞砸」mask 对齐）；Probe B 对 airport mean IoU ~0.78 且 severe 多 → **更支持 28w 权重与任务难度**，语义库 **可能放大** 宽 scope，但 **非唯一主因**。

6. **overpass 退化与 vehicle/road/bridge reference？**  
   probe 中 overpass 样本常共现 **vehicle**（`top_categories_probe`）；库中 **无 overpass 条目** → **reference 驱动的 exclusion 栈** 合理假设为 **风险放大器**（置信度中等）。

7. **no_prior 组也有退化？**  
   severe 3 / moderate 2 / stable 5 → 说明 **Probe B 退化不完全依赖 matched prior**（基线差异、类别难度、随机 patch 均可能）。**不支持「一切坏在语义库」叙事**。

8. **A/B/C/D 证据包**  
   - **D（三者都有）** 在叙事上最完整，但用户强制 **单选 A–D**：选 **B** 更贴近「库有风险但非排他主因」。  
   - **A** 过强（无法忽视 GT/库不对齐）。  
   - **C** 过强（Probe A 未显示库「单独」灾难性失败）。

---

## 9. 当前证据支持的失败原因排序

1. **Projector 瓶颈（B）**：`mean_signal_survival≈0.22`，与 Phase 2 设计一致。  
2. **训练步数 / 权重混杂（C）**：Probe B 7w vs 28w；no_prior 仍有退化。  
3. **语义库内容与对齐（A）**：GT 无库、reference exclusion 过长、vehicle-bridge 高频共现。

**置信度（主观但可复核）**：对排序 **0.72**（见 `semantic_library_audit_summary.json`）。

---

## 10. 下一步建议

1. **必做**：同预算 **merged** 三联体实验 + 小规模机制探针复用本脚本。  
2. **库方向（不改文件前提下）**：准备 **overpass → road/vehicle/bridge alias 策略** 的评审稿；压缩 **reference exclusion** 句长。  
3. **分析方向**：对 `per_sample_prior_hit_diff.jsonl` 全量重扫 `matched_concepts`（若需覆盖 water/road 等）以验证「零命中」是否为该 jsonl 过滤特性。

---

## 最终必选判断

**选择：B** — 语义库内容存在**轻微到中等**风险，但在当前证据下**不宜认定为主因**。

| 字段 | 内容 |
|------|------|
| **evidence_for** | GT/库不对齐；reference exclusion 信息量大；vehicle/bridge 高频 reference；overpass 无库条目。 |
| **evidence_against** | 代码路径一致；Probe A mask IoU 仍高；hook/stability 正常；no_prior 仍有 Probe B 退化。 |
| **confidence** | **0.72** |
| **recommended_next_action** | **先做 baseline_28w / v2_28w / refaware_28w merged 对照**；语义库修订与「降权 reference exclusion」作为 **并行评审项**，不替代对照实验。 |

---

## 附录：本次生成的文件

1. `docs/SEMANTIC_LIBRARY_AND_CODE_AUDIT.md`（本文件）  
2. `outputs/source/semantic_library_audit/semantic_library_schema_audit.json`  
3. `outputs/source/semantic_library_audit/concept_category_alignment.csv`  
4. `outputs/source/semantic_library_audit/concept_match_distribution.csv`  
5. `outputs/source/semantic_library_audit/concept_quality_audit.csv`  
6. `outputs/source/semantic_library_audit/risky_concepts_report.csv`  
7. `outputs/source/semantic_library_audit/semantic_library_audit_summary.json`  

---

## Top 10「风险 concept」（按 jsonl 命中 + probe 共现综合）

1. **vehicle**（626 / reference 为主 / Probe B 低）  
2. **bridge**（290 / 纯 reference 在 probe 子集 / Probe B 低）  
3. **airport**（219 / 纯 target / 仍多 severe）  
4. **ship**（190 / mixed target-ref / Probe B 最低均值之一）  
5. **harbor**（32）  
6. **road**（库内概念；**该 jsonl 切片计数 0** — 风险为「潜在未覆盖」）  
7. **water**（同上）  
8. **building**（同上）  
9. **parking lot**（GT 名无精确 match；库内用于表达触发）  
10. **playground**（同上）

---

## 是否建议先改语义库还是先改模型？

- **先改模型（结构）**：**不建议**。  
- **先改语义库**：可作为 **与对照实验并行** 的「降风险」路径，但 **不应跳过 28w 对照** 以免再次混淆因果。

## 是否建议继续分叉 B（实验线）？

**建议继续** — 在 **对照 merged 权重** 与 **可控 ablation**（例如缩短 reference exclusion、或 strict/无 refaware）齐备后，分叉才有判别力；当前 Probe B 不足以宣告胜负。
