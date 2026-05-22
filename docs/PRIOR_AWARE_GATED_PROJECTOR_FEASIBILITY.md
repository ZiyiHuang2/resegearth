# Prior-aware Gated Projector Feasibility Audit

**审计类型**：只读源码 + 架构可行性；**未**改代码、未训练、未改 JSON、未跑 probe。  
**依据代码版本**：`resegearth+source` 当前树（`llava_phi.py`、`dataset.py`、`llava_trainer.py`、`merge_lora_weights_and_save_hf_model.py`、`probe_refaware_seg_hidden_effect.py`）。

---

## 0. 一页结论

| 问题 | 结论 |
|------|------|
| **是否可实现？** | **可实现**：在数学上 `q = q_base + gate * q_prior` 且 `q_prior∈ℝ^{256}`、`q∈ℝ^{256}`，**无需**改 `predictor` / mask decoder 输入维度假设。 |
| **风险等级 A/B/C/D** | **B（可实现，中风险）**。主要风险在 **prior token 与 `last_hidden_state` 对齐**、**多模态拼接后序列变长**、以及 **DDP/DeepSpeed 未使用参数** 与 **Trainer `model(**inputs)` 参数契约**。 |
| **是否需要改 mask decoder？** | **不需要**（当前 `predictor(..., SEG_embedding)` 已吃 `[B,1,HIDDEN_DIM]`，`HIDDEN_DIM` 来自 yaml，典型 **256**）。 |
| **zero-gate 等价？** | **设计上可保证**：`gate=0` 且 **`q_prior` 不参与累加（或恒为 0 向量）** 时 `q=q_base`。**工程上**需避免 `0 * NaN`、以及 dtype 混用带来的极小数值差；建议 **无 prior 时短路不算 `q_prior`** 或 **`h_prior=0` 且 `prior_projector` bias=False 且初始 weight=0** 等组合验收。 |
| **prior span 训练时能否可靠传递？** | **当前不能**（dataset 未产出 token 级 mask）。**最小增量改造后可传递**：在 **最终 `input_ids` 定长之后** 构造 `prior_token_mask`（与 `SEG_token_embedding_indices` 同步 pad），经 **collator** 进入 **batch**，再经 **`forward`/`eval_seg` 显式形参** 传入（因 `model(**inputs)` **无任意 kwargs 通道**）。 |
| **raw baseline 无 prior 时能否动态跳过？** | **可以**：`prior_token_count==0` 或 mask 全零时 **`q=q_base`**。注意 **DDP**：硬跳过导致 `prior_projector` 未进图时可能触发 **unused parameter**，需 **`find_unused_parameters=True`** 或 **恒走 prior 分支用零张量乘 gate=0**。 |
| **是否建议进入 Stage 0？** | **建议**。先做 **zero-gate 数值等价** 与 **梯度可达性**，再考虑短训 sanity。 |

---

## 1. 当前 SEG projection 路径

### 1.1 模块定义

`SEG_token_projector` 为 **`nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)`**：

```152:152:segearth_r2/model/language_model/llava_phi.py
        self.SEG_token_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
```

### 1.2 `get_SEG_embedding`

对 batch 内每条序列，取 `SEG_embedding_indices.bool()` 为真的 hidden 行，再 `cat` 并 `unsqueeze(1)`：

```598:603:segearth_r2/model/language_model/llava_phi.py
    def get_SEG_embedding(self, hidden_states, SEG_embedding_indices):
        SEG_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, SEG_embedding_indices):
            current_refer_state = current_hidden_state[current_token_indice.bool()]
            SEG_embedding_list.append(current_refer_state)
        return torch.cat(SEG_embedding_list, dim=0).unsqueeze(1)
```

**结论（问题 1）**：

- **`h_seg` shape**：`[B, 1, config.hidden_size]`（与 Phase2 probe 一致，例如 2560）。  
- **`q_base` shape**：`[B, 1, HIDDEN_DIM]`，典型 **256**（由 `MASK_FORMER.HIDDEN_DIM` 决定，见 mask yaml）。  
- **送入 `predictor` 的张量**：`SEG_embedding`（即投影后的 query），同上 **`[B,1,256]`**。  
- **下游是否只接受该形状**：`forward` / `eval_seg` 中直接传入 `predictor(..., SEG_embedding)`，未见在调用点 reshape 为其他 query 数（object query 仍由 predictor 内部 `query_feat` 等负责）。  
- **若改输出维度**：需同步 `MASK_FORMER.HIDDEN_DIM`、预训练 mask2former 权重加载形状、`predictor` 与 `SEG_token_projector` 权重矩阵 — **影响面大**，故 **禁止改 256** 的约束是合理的。

### 1.3 `prepare_inputs_labels_for_multimodal` 位置

在 `forward` 与 `eval_seg` 中，于 LLM `self.model(...)` 之前调用，用于把 **image token 展开为 embed** 并 **对齐 labels / SEG_token_embedding_indices / image_features_indices**：

```642:644:segearth_r2/model/language_model/llava_phi.py
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, SEG_token_embedding_indices, image_features_indices = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images_clip,
                token_refer_id=token_refer_id, SEG_token_embedding_indices=SEG_token_embedding_indices)
```

```794:796:segearth_r2/model/language_model/llava_phi.py
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, SEG_token_embedding_indices, image_features_indices = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images_clip,
            token_refer_id=token_refer_id, SEG_token_embedding_indices=SEG_token_embedding_indices)
```

---

## 2. 拟议结构（相对当前代码）

**当前（`forward` L657-671）**：

```657:671:segearth_r2/model/language_model/llava_phi.py
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        ...
        SEG_embedding = self.SEG_token_projector(self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices))
        ...
        mask_outputs = self.predictor(multi_scale_features, mask_features, None, None, SEG_embedding)
```

**拟议**：保留 `SEG_token_projector(h_seg)` 为 **`q_base`**；新增 **`prior_projector: Linear(H,256)`** 与 **`gate`**（标量 / 每通道 / 每样本向量，见 §7）；  
`h_prior = pool(last_hidden_state, prior_token_mask)`；`q_prior = prior_projector(h_prior)`；`q = q_base + gate * q_prior`。

**与约束对照**：

1. **`q` 仍为 `[B,1,256]`** — 是。  
2. **mask decoder / predictor 接口不变** — 是（仍传最后一个 arg）。  
3. **原 `SEG_token_projector` 保留** — 是。  
4. **`prior_projector` 输出 256** — 需实现为 `Linear(H,256)`。  
5. **`gate` 初值 0** — 推荐 `nn.Parameter(torch.zeros(...))`。  
6. **raw baseline 无 prior → `q=q_base`** — 需工程保证（§5）。  
7. **不改 256** — 是。  
8. **旧 merged checkpoint**：新参数 **missing keys** — **必然**；需 **`strict=False` 加载** 或 **初始化后保存新 checkpoint**；merge 脚本当前保存 `state_dict` 全键（见 §8）。

---

## 3. 训练时 prior token span 传递方案（问题 8 详答）

### 3.1 现状：`dataset.py` **未**记录 prior 段

`RRSISDDataset.__getitem__` 构造 `human_value`（含 v2/refaware appendix）后 `preprocess_llama2` 得到 `input_ids` / `labels`，并计算 `SEG_token_embedding_indices`、`token_refer_id`（`dataset.py` 约 L405-L477），**没有** `prior_token_mask` / span 字段。

### 3.2 `token_refer_id` 与 prior 的关系

`token_refer_id = preprocess_referring_instruction(instruction)` — **仅原始 instruction**，**不包含** prior 文本（`dataset.py` L456）。

### 3.3 `labels=-100` 与 prior 段

`preprocess_llama2` 将 human 侧（含 prior）对应位置 `labels` 置 `IGNORE_INDEX`（-100）（`dataset.py` L125-L203；常量 `IGNORE_INDEX=-100`）。

### 3.4 多模态路径对 span 的干扰

`prepare_inputs_labels_for_multimodal` / `concat_image_seg_cls_embeds` 会 **改变序列长度** 并将 `SEG_token_embedding_indices` **右侧 pad 对齐**（`llava_phi.py` L543-551 附近）。因此：

- **字符级** prior 起止在 `human_value` 上可算，但 **映射到最终 `inputs_embeds` 行** 必须与 **post-multimodal** 的 `input_ids`/索引一致。  
- **最稳妥**：在 **`preprocess_llama2` 输出 `input_ids` 之后**，用 **稳定 marker** 或 **对「appendix 子串」二次 tokenize 对齐** 生成 `prior_token_mask`（与 `input_ids` 等长）。

### 3.5 Collator 是否可透传额外字段

`DataCollatorForCOCODatasetV2` 目前显式处理 `token_refer_id`、`SEG_token_embedding_indices`、`mask_num` 等（`dataset.py` L687-L701）。**未**处理自定义字段 — **需扩展** `__call__` 以对 `prior_token_mask` 做 `pad_sequence`（padding_value=0），并 `batch['prior_token_mask']=...`。

### 3.6 Trainer → model 的参数契约

`LLaVATrainer.compute_loss` 使用 `outputs = model(**inputs)`（`llava_trainer.py` L226-L239）。`SegEarthR2.forward` 形参列表为 **显式命名参数**（`llava_phi.py` L605-L623），**不含 `**kwargs`**：

- **结论**：batch 中 **多出的 key** 若直接 `**inputs` 传入，**可能** `TypeError: unexpected keyword argument`。  
- **对策（最小之一）**：在 `compute_loss` 内 `inputs.pop` 出 prior 相关张量并作为显式参数传入；或 **扩展** `forward(..., prior_token_mask=None, ...)`。

### 3.7 `llava_trainer.py` 专节（审计对象二.6）

- **`compute_loss`**（`llava_trainer.py` L226-L239）：在 `label_smoother` 分支会从 `inputs` 中 `pop("labels")`；随后 `inputs['global_step'] = global_step`；**整包 `inputs` 以关键字形式** 传给 `model(**inputs)`。因此 **`forward` 必须显式声明** 所有 batch 张量键（含未来的 `prior_token_mask`），或 **在 `compute_loss` 开头 pop 掉非模型参数** 再传入。  
- **`evaluate`**（同文件 L362-L376）：**不**走 `forward`，而是显式调用 `model.eval_seg(input_ids=..., attention_mask=..., images=..., images_clip=..., seg_info=..., token_refer_id=..., SEG_token_embedding_indices=..., labels=..., mask_num=...)`。**新增 prior 相关参数时**，此处必须与 `eval_seg` 签名同步扩展，否则评估阶段永远等价「无 prior」或运行时报错。

### 3.8 建议字段名

- `prior_token_mask`：`LongTensor`/`BoolTensor`，与 **padding 后** `input_ids` 对齐。  
- `prior_token_count`：`int` 或 `LongTensor`，便于日志与 `gate` 逻辑。  
- （可选）`prior_token_span`：`(start,end)` **仅在能证明与 pad 后序列一致时使用**。

### 3.9 `offset_mapping` / tokenize 落点（问题 8 补充）

- **当前树**：prior span **未**在 `dataset.py` 中记录；是否在别处启用 `return_offsets_mapping` 需以实际 `preprocess_llama2` / tokenizer 调用为准。**可靠默认策略**：在 **`preprocess_llama2` 已得到与训练一致的 `input_ids` 之后**，用 **模板内稳定子串边界** 或 **marker token 位置** 在 `input_ids` 上标 1，再经 `prepare_inputs_labels_for_multimodal` 与 `SEG_token_embedding_indices` **同一套 pad 规则** 对齐（与 §3.4 一致）。  
- **collator vs dataset**：**dataset 产出单样本张量 + collator pad** 与现有 `SEG_token_embedding_indices` 模式一致，**优先**在 `__getitem__` 生成与 `input_ids` 等长的 mask，**collator 只做 pad**，避免在 collator 中重复 tokenize。

---

## 4. 推理 / probe 时 prior span（问题 9）

- **推理**：与训练相同需求 — 需在 **最终 token 序列** 上定义 mask。  
- **现有 probe**：`prior_token_count = max(0, ref_tok_n - raw_tok_n)`，其中 `ref_tok_n` / `raw_tok_n` 为 `count_valid_tokens(...)`（`probe_refaware_seg_hidden_effect.py` **L795-L798**）— **长度差启发式**，**不是** token 级 `prior_token_mask`，**不足以**单独作为训练对齐依据（未区分 pad、image token 展开、特殊 token 重复等）。  
- **可复用部分**：probe 已复现 **同源** `build_refaware_fields` / `collator` / `forward_eval_seg_tensors` 路径，适合作为 **Stage0A** 的实验脚手架（未来允许改代码时）。

---

## 5. raw baseline 无 prior 与动态跳过（问题 6–7）

- **无 prior**：`prior_token_mask` 全零或 `prior_token_count=0` → **`q_prior` 置零** 且 **`gate=0` → `q=q_base`**。  
- **不引入随机性**：不调用 `dropout` 于 prior 分支（默认可）；推理 `eval` 模式。  
- **动态分支与 DDP / DeepSpeed**：若 `if prior_count>0:` **跳过** `prior_projector`，在部分 batch 上参数未参与 loss，可能 **unused params**（DDP/ ZeRO 等报错或静默不同步）。可选方案：  
  - **A（恒在图内）**：`h_prior` 置零、`q_prior = prior_projector(h_prior)`，再 `q = q_base + gate * q_prior`；**`gate=0` 且 `prior_projector` 零初始化（含 bias=0）** 时与「不算 `q_prior`」数值一致（需 Stage0A 验收）。  
  - **B**：**`find_unused_parameters=True`**（或等价配置），允许整段分支被跳过（**训练步均摊开销上升**）。  

**Stage 0 vs 正式 28w（问题 7 结论）**：  
- **Stage 0**：优先 **方案 A**，减少分布式变量，便于 **zero-gate 数值回归** 与 **梯度探针** 稳定复现。  
- **28w**：若 batch 内 **混有** no-prior / prior，且希望 **省算力**，可评估 **条件分支 + `find_unused_parameters=True`**；若追求 **实现简单与梯度稳定**，可继续 **方案 A**（prior 很短时 `prior_projector` 额外 FLOPs 相对整网通常较小，需 profile 后决策）。

**结论**：**可实现动态跳过**；**DeepSpeed 梯度同步本身不禁止 if 分支**，风险在 **未使用参数** 与 **不同 rank 分支不一致**（若存在）— 需统一策略或 `find_unused_parameters`。

---

## 6. zero-gate equivalence（问题 4）

- **严格 `q=q_base`**：`gate=0` **且** `q_prior` 为 **精确零张量**（或不累加）时成立。  
- **`SEG_token_projector` 为 `nn.Linear`**，默认无 `LayerNorm`/dropout 包在模块外；**dtype** 一致时数值稳定性较好。  
- **验收**：Stage0A（见 `outputs/source/prior_aware_gated_projector_feasibility/stage0_test_plan.md`）：`projected_query_cosine>0.9999`、`mask_iou>0.999`、`pred_area_delta≈0`。

---

## 7. prior hidden 来源与 pooling（问题 5）

- **同层**：与 `h_seg` 一致，取 **`outputs.last_hidden_state`** 最符合当前架构（`forward` L657）。  
- **Pooling**：`mean` 最简；`attention` pooling 需额外 query（增参/增风险）。  
- **形状**：mask 选 `K` 个 prior token 时，`h_token ∈ℝ^{K×H}` → mean → `ℝ^{H}` → `unsqueeze(1)` → `[1,1,H]` / batch `[B,1,H]`。  
- **`prior_token_mask` 必要性**：**强烈建议**，否则难以定义「哪些 token 属于 public prior」 vs 指令/模板噪声。

---

## 8. Marker 是否必须（问题 10）

- **当前 prompt**：`build_rrsisd_refaware_exclusion_only_human_value` 使用固定模板 + `format_grounding_appendix`（JSON）+ `[Exclusion Prior]` 文本头（`concept_public_grounding_train.py` L114-L144）。**已有** `"[Exclusion Prior]"` 可作为 **弱 marker**；**完整 JSON 边界**无单一 bracket（依赖 `Category-level public grounding...` 头 + JSON）。  
- **最小侵入**：在 **Python 拼接模板** 中加 **普通 ASCII marker 行**（不改 JSON 文件）以辅助 span — **需改代码**而非改 `concept_public_semantic_library_v2.json`。  
- **新 special token**：需 `resize_token_embeddings` + 全链路对齐 — **侵入大**，不推荐作为第一步。

---

## 9. 训练信号 / 梯度（问题 11）

- **mask loss / dice** 经 `predictor(..., q)` 回传至 `q`，再至 `q_base` 与 `gate*q_prior` 分支。  
- **`labels=-100` 于 prior token**：只屏蔽 **LM CE** 路径；**不阻止** hidden 参与 mask 分支梯度（hidden 仍由整条序列前向产生）。  
- **若 grad 为 0**：可能 `gate` 被裁剪为非参数常量、`h_prior` 全零、`mask` 全零、或 `loss` 未连到 prior 分支（实现 bug）。

---

## 10. 对 baseline_raw 的影响（问题 12）

- **无 library / 无 prior**：应保持 **`q=q_base`**；新增参数若从不训练可能 **欠拟合**，但不会破坏旧行为 — **前提是 gate 初值 0 且等价性测试通过**。  
- **`prior_projector` bias**：若 zero-init 权重与 bias=0，输入全零 → 输出全零，利于等价性。  
- **`gate` 形态**：**标量**最易分析与初始化；**每通道 `[256]`** 更灵活；**每样本动态**需额外网络 — 风险与复杂度更高。首版建议 **标量或可学习 per-channel 向量 + 强 L2 正则/小步长**。

---

## 11. 对 train / eval / merge 的影响（问题 13）

### 11.1 `merge_lora_weights_and_save_hf_model.py`

- `train_module_list` 当前包含 **`SEG_token_projector`**（`merge_lora_weights_and_save_hf_model.py` L109-L111）。  
- **新增** `prior_projector` / `gate`：**不在** LoRA `target_modules` 自动发现里 — 需 **显式训练** 或 **扩展 LoRA 目标策略**。  
- `save_pretrained(..., state_dict=...)` 会保存 **当前模型全部参数张量**（L141-L146）— **可**携带新键，但 **旧 checkpoint 加载** 需 `strict=False` 或初始化。

### 11.2 `eval`

`llava_trainer.evaluate` 调 `model.eval_seg(...)` 显式传参（`llava_trainer.py` L366-L376），**需同步**增加 `prior_token_mask` 等，或 **在 evaluate 前 pop 并默认 None**（等价无 prior）。

### 11.3 仅训 LoRA 不训 prior 分支

- **风险**：prior 分支永远随机 init → **等效噪声** 若 `gate` 非严格 0。  
- **建议**：`gate` 锁 0 冻结 prior 直到 warmup 结束，或 **始终联合训练** 小模块。

---

## 12. 最小改动文件清单（问题 14）

见 **`outputs/source/prior_aware_gated_projector_feasibility/required_code_changes.csv`**。

---

## 13. Stage 0 / Stage 1（问题 15–16）

见 **`outputs/source/prior_aware_gated_projector_feasibility/stage0_test_plan.md`**。

**Stage1 短训**：**建议必要**；用于观察 `gate` 是否离开 0、`signal_survival` 是否改善，再投入 7w/28w 主训。

---

## 14. 风险清单（技术 / 实验）

| 风险 | 说明 |
|------|------|
| **Span 对齐错误** | multimodal 展开后索引漂移 → `h_prior` 污染指令 token → mask 劣化。 |
| **Trainer kwargs** | `model(**inputs)` 与 `forward` 签名不一致导致运行期错误。 |
| **DDP unused** | 条件分支跳过新参数 → 分布式报错。 |
| **merge / 加载** | missing keys、LoRA 未覆盖新线性层。 |
| **实验** | `gate` 学不出 / 梯度爆炸 / 仅拟合先验文本。 |

---

## 15. 与语义库审计、Phase2 结论的关系

- Semantic audit **B**：库不是唯一主因；Phase2 显示 **projector 衰减显著**。  
- **Prior-aware gated projector** 在方向上 **对齐「结构性瓶颈」假设**，但 **不能替代** 「同预算 merged 对照」对因果的解释。

---

## 16. 最终裁决

**B — 可实现，中风险。**

- **最大技术风险**：**prior token ↔ `last_hidden_state` 索引** 在多模态序列中的一致性 + **Trainer 参数传递**。  
- **最大实验风险**：**gate 与 prior 分支训练不动或破坏 zero-init 等价**。  
- **是否建议进入 Stage 0 代码改造**：**建议**（先 Stage0A–C）。  
- **是否建议先改语义库**：**不必须作为前置**；可与 Stage0 并行评审短 exclusion。  
- **是否继续 prompt-only 路线**：在 **Stage0 未通过前**，**仍应保留** prompt-only 作为低风险基线。

---

## 附录：关键源码锚点

| 主题 | 文件:行 |
|------|---------|
| `SEG_token_projector` | `llava_phi.py` ~152 |
| `get_SEG_embedding` | `llava_phi.py` 598-603 |
| `forward` 投影与 `predictor` | `llava_phi.py` 657-671 |
| `eval_seg` | `llava_phi.py` 770-825 |
| `prepare_inputs_labels_for_multimodal` | `llava_phi.py` 457+ |
| `DataCollator` 批处理 | `dataset.py` 599-702 |
| `Trainer.compute_loss` | `llava_trainer.py` 226-239 |
| LoRA train list 含 `SEG_token_projector` | `merge_lora_weights_and_save_hf_model.py` 109-111 |
| probe `prior_token_count` | `probe_refaware_seg_hidden_effect.py` 795-798 |
