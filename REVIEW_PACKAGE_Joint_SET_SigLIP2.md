# Joint-SET-SigLIP2 代码审核包

> **目标**: 审核 SegEarth-R2 / LaSeRS 的 Joint-SET-SigLIP2 代码修改，判断机制设计和代码实现是否真的满足实验目标。
>
> **审核重点**: 这次修改是否真的把 SET 从 frozen 弱调制，升级成 joint training 中的主路径 mask-execution controller。

---

## 1. 背景

### 1.1 旧方案：A3 Frozen SET 5w

| 项目 | 值 |
|------|-----|
| 路径 | `/root/rivermind-data/huangziyi/reseg/output/set/a3-frozen-lasers-set-5w` |
| 脚本 | [`run_a3_set_lasers_5w_from_baseline.sh`](run_a3_set_lasers_5w_from_baseline.sh) |
| 起点 | baseline merged_model (SigLIP1) |
| 可训练 | 仅 `set_conditioner` / `count_head` / `category_set_head` |
| LoRA | `False` |
| SET layers | 1 |
| gate_init | 0.0 |

**主要发现**:
1. A3 frozen 可以稳定训练，SET gate 从约 0.018 增长到约 0.336
2. `generated_SEG_count` 在 655 个交集样本上完全不变 (0/655 changed)
3. `model_answer` 几乎不变 (3/655 有差异)
4. 445/655 样本 `merged_IoU` 有变化 → SET 确实能影响 mask execution
5. Multiple 子集收益很弱，equal-count mean merged IoU 约 +0.005
6. **结论**: frozen A3 只是 mask-side 弱调制，不能作为完整 [SET] 方法

### 1.2 新方案目标：Joint-SET-SigLIP2

从 Mipha-3B + SigLIP2 开始 joint training：
- SET 在训练中强参与 [SEG] mask execution
- LoRA / SEG_token_projector / predictor / pixel_decoder / SET modules 一起学习
- count/category supervision 有机会影响目标集合表达

---

## 2. 新旧方案关键差异对比

| 维度 | Frozen A3 (旧) | Joint SET (新) |
|------|---------------|----------------|
| 脚本 | [`run_a3_set_lasers_5w_from_baseline.sh`](run_a3_set_lasers_5w_from_baseline.sh) | [`run_train_set_joint_from_scratch.sh`](run_train_set_joint_from_scratch.sh) |
| 起点 | baseline merged_model (SigLIP1) | **Mipha-3B** (from scratch) |
| Vision Tower | `siglip-so400m-patch14-384` (SigLIP1) | **`siglip2-so400m-patch14-384`** (SigLIP2) |
| LoRA | `False` | **`True`** (r=8, alpha=16) |
| `a3_train_only_set_modules` | `True` | **`False`** |
| SET layers | 1 | **2** |
| SET heads | 4 | 4 |
| gate_init | 0.0 | **0.001** |
| 训练步数 | 50k | **80k** |
| 可训练参数 | 仅 SET modules | **LoRA + SEG_token_projector + predictor + pixel_decoder + SET modules** |
| 视觉 backbone | 冻结 | 冻结 (`train_clip_backbone=False`, `train_swin_backbone=False`) |
| merge 方式 | `--a3_only --no-lora --baseline_model_path` | **LoRA merge (标准流程)** |

---

## 3. 代码链路逐段审核

### 3.1 训练入口 → 模型初始化

**文件**: [`segearth_r2/train/train.py`](segearth_r2/train/train.py)

```python
# Line 312-324: 模型加载
model = SegEarthR2.from_pretrained(
    model_args.model_name_or_path,  # Mipha-3B, 不是 merged_model
    mask_decoder_cfg=mask_cfg,
    add_cross_attn=True,
)

if not model.is_train_mask_decode:
    # Mipha-3B 没有 mask_decode_train=True
    # → 走 initial_mask_module，加载 mask2former 权重
    model.initial_mask_module(mask2former_ckpt, model_args)
else:
    model.init_set_conditioning_modules(model_args)
```

**审核**: ✅ `initial_mask_module()` 内部 (line 173) 调用 `self.init_set_conditioning_modules(model_args)`，SET 模块在 mask 初始化时一并创建。

**关键**: `initial_mask_module` 中 `pretrained_path` 不为 None → 从 mask2former pkl 加载 pixel_decoder/predictor 权重。

---

### 3.2 训练参数冻结策略

**文件**: [`segearth_r2/train/train.py`](segearth_r2/train/train.py), line 407-443

```python
# a3_train_only_set_modules=False 分支 (line 407-443)
train_module_list = [
    "lm_head", "pixel_decoder", "predictor", "SEG_token_projector",
]
if a3_modules_active:
    train_module_list.extend(["set_conditioner", "count_head", "category_set_head"])

# LoRA 启用: 对 LLM 的 q_proj/v_proj 加 LoRA
if training_args.lora_enable:
    lora_target_modules = find_linear_layers(model, train_module_list=train_module_list)
    lora_config = LoraConfig(r=8, lora_alpha=16, ...)
    model = get_peft_model(model, lora_config)

# 再将 train_module_list 中的模块设为 requires_grad=True
for n, p in model.named_parameters():
    if any(x in n for x in train_module_list):
        p.requires_grad = True
```

**审核**: ✅ 可训练参数范围正确：
- LoRA 作用于 LLM 的 q_proj/v_proj
- `lm_head`, `pixel_decoder`, `predictor`, `SEG_token_projector` 全参数可训练
- `set_conditioner`, `count_head`, `category_set_head` 全参数可训练
- 视觉 backbone (`vision_tower`, `vision_tower_mask`) 冻结

---

### 3.3 SET 在主路径中的位置

**文件**: [`segearth_r2/model/language_model/llava_phi.py`](segearth_r2/model/language_model/llava_phi.py), line 782-815

```python
# forward() 中的关键路径:
hidden_states = outputs.last_hidden_state                          # LLM 输出
SEG_embedding = self.SEG_token_projector(
    self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices)
)                                                                   # [SEG] token → projector

# ⭐ SET conditioning 在这里介入
SEG_embedding, loss_set_count, loss_set_category, ... = (
    self._apply_set_conditioning(SEG_embedding, mask_num, category_set_labels)
)

# 然后 refined SEG_embedding 进入 predictor
mask_outputs = self.predictor(multi_scale_features, mask_features, ..., SEG_embedding)
```

**审核**: ✅ SET 在 `SEG_token_projector` 之后、`predictor` 之前。refined SEG embedding 确实进入了 mask decoder 主路径。

**关键**: 不是只在 loss 里用 q_set，refined SEG_embedding 替代了原始 SEG_embedding 进入 predictor。

---

### 3.4 SET Conditioner 内部设计

**文件**: [`segearth_r2/model/set_conditioner.py`](segearth_r2/model/set_conditioner.py), line 206-279

```
输入: seg_embedding [sum(K_i), C] + mask_num
  ↓
regroup → [B, Kmax, C] + valid_mask
  ↓
pool_q_set: attention pooling → q_set [B, C]
  ↓
set_proj(q_set) → 加到每个 SEG embedding
  ↓
TransformerEncoder (2 layers, 4 heads, norm_first)
  ↓
output_proj → delta
  ↓
gate = sigmoid(gate_proj([seg_orig, q_set_expand]))
  ↓
refined = seg_orig + residual_gate * gate * delta
  ↓
flatten → [sum(K_i), C]
```

**初始化策略**:
| 组件 | 初始化 | 效果 |
|------|--------|------|
| `set_proj` | zeros | q_set 投影初始为零，不干扰 |
| `output_proj` | zeros | delta 初始为零 |
| `gate_proj.weight` | zeros | - |
| `gate_proj.bias` | **-4.0** | sigmoid(-4.0) ≈ 0.018，gate 初始很低 |
| `residual_gate` | **0.001** | 缩放 delta 路径 |

**审核**: ✅ 初始化策略合理——初始时 SET 几乎不改变主路径（gate≈0.018, delta≈0, residual_gate=0.001），随着训练逐步放开。相比 frozen 版的 `gate_init=0.0`，新版的 `gate_init=0.001` 和 `gate_proj.bias=-4.0` 提供了更合理的冷启动。

---

### 3.5 Count / Category Loss 接入

**文件**: [`segearth_r2/model/language_model/llava_phi.py`](segearth_r2/model/language_model/llava_phi.py), line 700-719

```python
# count loss: 从 mask_num (GT [SEG] 数量) 作为 target
loss_set_count, target_count_acc = compute_set_count_loss(
    self.count_head, q_set, mask_num, self.set_max_count
)

# category loss: 从 GT answer 的 <p>...</p> 提取，multi-hot
if category_set_labels is not None:
    loss_set_category = compute_set_category_loss(
        self.category_set_head, q_set, category_set_labels
    )
```

**数据流** ([`segearth_r2/datasets/dataset.py`](segearth_r2/datasets/dataset.py), line 511-516):
```python
category_labels, ... = build_category_set_labels_from_answer(
    answer,                          # GT answer (含 <p>...</p>)
    self.lasers_category_vocab,      # 191 类
)
data_dict['category_set_labels'] = category_labels
```

**审核**: ✅
- count target = `mask_num`（每个样本的 GT [SEG] 数量）
- category target = 从 GT answer `<p>...</p>` 提取的 multi-hot
- vocab 是 `lasers_category_vocab.json`，**191 类**（非 fallback 34 类）
- loss 权重: `lambda_set_count=0.05`, `lambda_set_category=0.1`

---

### 3.6 总 Loss 组合

**文件**: [`segearth_r2/model/language_model/llava_phi.py`](segearth_r2/model/language_model/llava_phi.py), line 901-905

```python
loss = llm_loss + mask_loss + 0.01 * loss_attention
if loss_set_count is not None:
    loss = loss + self.lambda_set_count * loss_set_count
if loss_set_category is not None:
    loss = loss + self.lambda_set_category * loss_set_category
```

**审核**: ✅ LLM loss + Mask loss + SET count loss + SET category loss 联合训练。

---

### 3.7 Merge (LoRA + SET 权重合并)

**文件**: [`segearth_r2/train/merge_lora_weights_and_save_hf_model.py`](segearth_r2/train/merge_lora_weights_and_save_hf_model.py)

调用方式 (line 184-193 of shell script):
```bash
python merge_lora_weights_and_save_hf_model.py \
    --model_path "${ckpt}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${save_dir}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lasers_category_vocab_path "${VOCAB_PATH}"
```

**脚本内部** (line 124-133):
```python
if model.is_train_mask_decode:
    init_args = model.config
    if not getattr(init_args, "use_set_conditioner", False):
        init_args.use_set_conditioner = True
        ...
    model.init_set_conditioning_modules(init_args)
```

**然后** (line 172-174):
```python
model = load_state_dict_from_zero_checkpoint(model, model_path)
if hasattr(model, "merge_and_unload"):
    model = model.merge_and_unload()  # 合并 LoRA
```

**审核**: ✅
- 从 DeepSpeed checkpoint 加载完整权重
- `merge_and_unload()` 将 LoRA 权重合并到基础模型
- SET 模块权重在 checkpoint 中，一并加载
- config 中保留 `use_set_conditioner=True`

---

### 3.8 Merge 后验证

**Shell 脚本** (line 196-209):
```python
model = SegEarthR2.from_pretrained(merged_dir, ...)
if getattr(model.config, "use_set_conditioner", False) and getattr(model, "set_conditioner", None) is None:
    model.init_set_conditioning_modules(model.config)
assert getattr(model.config, "use_set_conditioner", False)
assert model.set_conditioner is not None
```

**审核**: ✅ 验证了 merged model 包含 SET config 和 SET 模块。

---

### 3.9 Eval 加载路径

**文件**: [`segearth_r2/utils/builder.py`](segearth_r2/utils/builder.py), line 47-48

```python
model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=mask_cfg, **kwargs)

if getattr(model.config, "use_set_conditioner", False) and getattr(model, "set_conditioner", None) is None:
    model.init_set_conditioning_modules(model.config)
```

**审核**: ✅ Eval 时自动检测 config 中的 `use_set_conditioner`，如果 SET 模块未加载则初始化。

---

### 3.10 Eval 中 SET 是否启用

**文件**: [`segearth_r2/model/language_model/llava_phi.py`](segearth_r2/model/language_model/llava_phi.py), line 964-965

```python
# eval_seg() 方法中:
SEG_embedding = self.SEG_token_projector(self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices))
SEG_embedding, _, _, _, _ = self._apply_set_conditioning(SEG_embedding, mask_num)
```

**审核**: ✅ `eval_seg()` 中调用了 `_apply_set_conditioning`，eval 时 SET 确实生效。

---

### 3.11 Diagnostic 链路

**文件**: [`segearth_r2/eval/eval_lasers_diagnostic.py`](segearth_r2/eval/eval_lasers_diagnostic.py)

```
load_pretrained_model (builder.py) → 自动 init SET
  ↓
greedy_generate_answer → 自回归生成 model_answer
  ↓
build_seg_batch → 用 model_answer 中的 [SEG] 构建 batch
  ↓
model.eval_seg() → 内部调用 _apply_set_conditioning
  ↓
compute_iou_diagnostics → 计算 IoU matrix / merged IoU
  ↓
输出 JSONL: generated_SEG_count, model_answer, merged_IoU, ...
```

**审核**: ✅ Diagnostic 完整覆盖了 generation + SET-conditioned mask execution。

---

## 4. 设计层面审核回答

### Q1: 新代码是否真的从 A3 frozen 变成 Joint SET？

**是。** 确认：
- 脚本中 `--a3_train_only_set_modules False`（非 `True`）
- 脚本中 `--lora_enable True`（非 `False`）
- 起点是 `Mipha-3B`，不是 `merged_model`

### Q2: 新训练是否从 Mipha-3B + SigLIP2 开始？

**是。**
- `model_name_or_path` = `/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B`
- `vision_tower` = `/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384`

### Q3: SET 是否仍在主路径中？

**是。** 路径：
```
hidden_states → get_SEG_embedding → SEG_token_projector → SetConditioner → predictor
```

### Q4: SET 是否不只是 loss head？

**不是。** `refined_seg_embedding` 替代原始 `SEG_embedding` 进入 `predictor()`。count_head/category_set_head 是额外的 loss head，但 SET 的核心作用在 mask execution 主路径中。

### Q5: SET 的影响力是否足够？

相比 frozen 版有明显提升：
- SET layers: 1 → **2**（更强表达能力）
- gate_init: 0.0 → **0.001**（非零初始值允许早期梯度流）
- gate_proj bias: **-4.0**（sigmoid ≈ 0.018，冷启动）
- 同时训练 predictor + pixel_decoder + SEG_token_projector → SET 可以影响这些模块的表示学习

### Q6: count/category supervision 是否正确接入？

**是。**
- count target: 来自 `mask_num`（GT [SEG] 数量）
- category target: 来自 GT answer 的 `<p>...</p>` 提取
- vocab: 191 类（`lasers_category_vocab.json`）
- merge/load 时传入 `lasers_category_vocab_path`

### Q7: 是否有过度声明风险？

**需要注意。** SET 不直接修改 autoregressive logits，因此：
- 不能声称 SET 直接控制 `generated_SEG_count`
- 只能说 joint training 可能让 set supervision 通过梯度影响 LLM/SEG representation
- 如果训练后 `generated_SEG_count` 发生变化，那是 joint training 的间接效果
- 如果 `generated_SEG_count` 不变但 equal-count IoU 提升，那是 mask-side SET 的效果

---

## 5. 发现的潜在问题

### ⚠️ 问题 1: `set_conditioner.forward()` 类型注解与实际返回值不匹配

**文件**: [`segearth_r2/model/set_conditioner.py`](segearth_r2/model/set_conditioner.py), line 254-258

类型注解声明返回 `Tuple[Tensor, Tensor, Tensor, Tensor]`（4 个），但实际返回 5 个值。调用方 [`_apply_set_conditioning`](segearth_r2/model/language_model/llava_phi.py:696) 正确解包了 5 个值，所以不影响运行，但类型注解不准确。

**严重程度**: 低（不影响功能）

---

### ⚠️ 问题 2: WANDB_NAME 与 OUTPUT_DIR 步数不一致

**文件**: [`run_train_set_joint_from_scratch.sh`](run_train_set_joint_from_scratch.sh), line 20 vs 67

- `WANDB_NAME` 默认值: `joint-set-siglip2-from-scratch-50k`
- `OUTPUT_DIR`: `joint-set-siglip2-from-scratch-80k`

实际训练步数是 80k，但 wandb 名称写的是 50k。建议统一为 80k。

**严重程度**: 低（不影响训练，仅影响 wandb 命名）

---

### ⚠️ 问题 3: Diagnostic 的 `eval_seg` 调用没有传 `category_set_labels`

**文件**: [`segearth_r2/eval/eval_lasers_diagnostic.py`](segearth_r2/eval/eval_lasers_diagnostic.py), line 396-406

`eval_seg()` 内部调用 `_apply_set_conditioning(SEG_embedding, mask_num)` 时没有传 `category_set_labels`。但这是 inference 模式，category loss 不需要计算，所以实际上没问题——`_apply_set_conditioning` 中 category loss 只在 `category_set_labels is not None` 时才计算。

**严重程度**: 低（inference 时不需要 loss）

---

## 6. 静态验证建议

在 GPU 不可用时，至少运行以下验证：

```bash
cd /root/rivermind-data/huangziyi/reseg/segearth+set
PYTHON=/root/rivermind-data/miniconda3/envs/reseg/bin/python

# 1. 语法检查
$PYTHON -m py_compile \
  segearth_r2/model/set_conditioner.py \
  segearth_r2/model/language_model/llava_phi.py \
  segearth_r2/train/train.py \
  segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
  segearth_r2/datasets/dataset.py

# 2. Shell 脚本语法检查
bash -n run_train_set_joint_from_scratch.sh

# 3. 检查 trainable 参数范围（不冻结模式）
$PYTHON scripts/check_a3_trainable_params.py \
  --model_name_or_path /root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B \
  --use_set_conditioner \
  --a3_train_only_set_modules  # 注意：这个脚本默认是 True，需要改为 False 来验证 joint 模式
```

---

## 7. 训练后必须审核的指标

代码已支持输出以下指标：

### 标准评估
- [x] LaSeRS Table2（9 benchmarks）
- [x] overall weighted metrics
- [x] Multiple / Long / Instance 子集
- [x] RRSIS-D / RefSegRS / RISBench / EarthReason（并行评估）

### Diagnostic
- [x] `generated_SEG_count`（是否变化）
- [x] `under_generate` rate
- [x] `equal_count` rate
- [x] `model_answer`（是否变化）
- [x] `merged_mask_IoU`
- [x] `per_SEG_IoU_with_each_GT`（duplicate best-match rate）
- [x] `all_gt_covered` rate（可通过 merged IoU 推断）
- [x] rescued / damaged（analyze 脚本支持）
- [x] big_gain / big_loss（analyze 脚本支持）
- [x] 关闭 `use_set_conditioner` 后的 eval 对照（需手动运行）

### 最关键的机制判断
| 现象 | 结论 |
|------|------|
| `generated_SEG_count` 变化 | joint SET 可能间接影响 generation-side |
| `generated_SEG_count` 不变，但 equal-count Multiple IoU/coverage 明显提升 | mask-side SET 有效 |
| 两者都没有明显变化 | 机制不成立 |

---

## 8. 审核结论

### 1. 总体判定: **PASS**（可以开始训练）

### 2. 设计是否满足目标

| 检查项 | 状态 |
|--------|------|
| 从 frozen A3 变成 joint SET | ✅ `a3_train_only_set_modules=False`, `lora_enable=True` |
| 扩大了 SET 对主路径的影响 | ✅ 2 layers, gate_init=0.001, 同时训练 predictor/pixel_decoder |
| 不只是小辅助头 | ✅ refined SEG embedding 进入 predictor 主路径 |
| 从 Mipha-3B + SigLIP2 开始 | ✅ |
| 191 类 category vocab | ✅ |

### 3. 代码链路完整性

| 链路 | 状态 |
|------|------|
| Train (forward + loss) | ✅ |
| Merge (LoRA + SET 权重) | ✅ |
| Load (builder.py 自动 init SET) | ✅ |
| Eval (eval_seg 调用 _apply_set_conditioning) | ✅ |
| Diagnostic (generation + SET mask) | ✅ |

### 4. 最大风险

1. **SigLIP2 归因混淆**: 如果效果提升，无法区分是 SigLIP2 的贡献还是 SET 的贡献。建议保留一组 SigLIP1 的 joint SET 对照，或至少对比 baseline SigLIP1 的结果。

2. **SET 权重在 merge 中丢失**: 已通过 `verify_merged_set` 验证，但 merge 后 `init_set_conditioning_modules` 可能重新初始化（如果 config 中缺少必要字段）。当前代码在 merge 脚本 line 126-132 做了防御性设置。

3. **Trainable 参数范围**: 当前 `train_module_list` 包含 `lm_head`，这意味着整个 LM head（51200 × hidden_dim）都在训练。这是预期行为，但参数量较大。

4. **generated_SEG_count 可能仍然不变**: 因为 SET 不直接修改 autoregressive logits，joint training 能否改变 generation behavior 取决于 count/category loss 的梯度能否通过 LLM 反向传播影响 token 分布。LoRA 提供了这条路径，但效果不确定。

### 5. 是否可以开始全量训练

**可以**，但建议：

1. 先跑一个 500-1000 step 的 smoke test，确认 forward/backward 不崩
2. Smoke test 后检查 wandb 中的 `set_gate_value` 和 `target_count_acc` 是否正常变化
3. 全量训练完成后，务必跑 diagnostic 并对比 baseline
4. 如有条件，保留一组 SigLIP1 + joint SET 对照以分离 SigLIP2 的贡献

### 6. 机制定位

**当前实现是 mask-side SET controller**：
- SET 作用于 `SEG_token_projector` 输出和 `predictor` 输入之间
- 不直接修改 LLM 的 autoregressive logits
- count/category loss 通过梯度反向传播**间接**影响 LLM 参数（通过 LoRA）

**论文表述建议**：
- 区分 "generation-side target planning"（LLM 决定生成几个 [SEG]）和 "mask-side set execution"（SET 改善多个 mask 的质量）
- 如果 `generated_SEG_count` 在训练后发生变化，可表述为 "joint training enables set-level supervision to influence generation behavior through gradient propagation"
- 如果不变，应诚实表述为 "SET operates as a mask-side refinement mechanism, improving mask quality without altering generation planning"

---

## 附录: 关键文件索引

| 文件 | 用途 |
|------|------|
| [`run_train_set_joint_from_scratch.sh`](run_train_set_joint_from_scratch.sh) | 新训练脚本 |
| [`run_a3_set_lasers_5w_from_baseline.sh`](run_a3_set_lasers_5w_from_baseline.sh) | 旧 frozen 脚本（对比） |
| [`segearth_r2/model/set_conditioner.py`](segearth_r2/model/set_conditioner.py) | SET 核心模块 |
| [`segearth_r2/model/language_model/llava_phi.py`](segearth_r2/model/language_model/llava_phi.py) | 主模型 forward/eval_seg |
| [`segearth_r2/train/train.py`](segearth_r2/train/train.py) | 训练入口 + 参数冻结 |
| [`segearth_r2/train/a3_training_utils.py`](segearth_r2/train/a3_training_utils.py) | A3 冻结/检查工具 |
| [`segearth_r2/train/merge_lora_weights_and_save_hf_model.py`](segearth_r2/train/merge_lora_weights_and_save_hf_model.py) | LoRA merge + SET 权重保存 |
| [`segearth_r2/utils/builder.py`](segearth_r2/utils/builder.py) | Eval 模型加载 |
| [`segearth_r2/eval/eval.py`](segearth_r2/eval/eval.py) | 标准评估 |
| [`segearth_r2/eval/eval_lasers_diagnostic.py`](segearth_r2/eval/eval_lasers_diagnostic.py) | Generation diagnostic |
| [`segearth_r2/datasets/dataset.py`](segearth_r2/datasets/dataset.py) | 数据加载 + category label 构建 |
| [`segearth_r2/model/lasers_category_vocab.json`](segearth_r2/model/lasers_category_vocab.json) | 191 类 category vocab |
| [`scripts/check_a3_trainable_params.py`](scripts/check_a3_trainable_params.py) | Trainable 参数检查 |
| [`scripts/analyze_a3_set_diagnostic.py`](scripts/analyze_a3_set_diagnostic.py) | Diagnostic 对比分析 |
| [`run_train_merge_test.sh`](run_train_merge_test.sh) | 参考：base 训练流程 |