### 基于中间层熵的 gpt-oss 动态解码说明

本说明文档介绍在 `transformers` 中如何为 `GptOssForCausalLM` 启用**基于中间层熵的动态解码算法**，以及相关输出的含义。

---

### 一、算法概述

- **目标**：在解码阶段，不再总是使用末层的 logits，而是根据各层的预测分布信息熵，动态选择“更合适的层”进行解码，从而实现“早退 / 内层解码”等效果。
- **支持模型**：目前仅针对 `gpt-oss` 模型（`GptOssForCausalLM`），且仅在非 beam 的解码（`num_beams == 1`）下生效。
- **集成位置**：
  - 核心逻辑在 `GenerationMixin._entropy_decoding` 中实现；
  - 通过 `GenerationConfig` 的新字段在 `generate` 方法中进行启用与控制。

---

### 二、可选策略

通过 `GenerationConfig.entropy_decoding` 选择解码策略：

- **`None`（默认）**：  
  - 不启用内层熵解码，`generate` 逻辑与原版一致，始终使用末层 logits 解码。

- **`"trough"`（熵谷策略）**：  
  - 每一步解码时，对所有层的 logits 计算信息熵；
  - 对于每个样本，选择**信息熵最小的那一层**（即“熵谷”）作为本步解码所用的层。

- **`"random_after"`（熵谷之后随机）**：  
  - 首先同 `"trough"` 一样找到熵谷层索引 `i_trough`；
  - 然后在区间 `[i_trough, L-1]`（从熵谷层到末层）中 **均匀随机** 选择一层进行解码；
  - 该随机选择在整个解码过程中是按照固定策略执行的（不再使用概率 `p_strategy` 混合末层），即一旦设置 `"random_after"`，每一步都会在对应区间做一次均匀采样。

> 注意：上述策略仅影响**选用哪一层的 logits**，后续仍会正常走 `logits_processor`（温度、top-k、top-p、惩罚等）以及采样 / 贪心逻辑。

---

### 三、如何在 `generate` 中启用

新算法是通过 `GenerationConfig` 控制的，你可以有两种主流用法：

#### 1. 直接在 `generate` 中传入参数（推荐）

```python
from transformers import AutoTokenizer, GptOssForCausalLM

model_name = "your-gpt-oss-checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GptOssForCausalLM.from_pretrained(model_name)

prompt = "你好，请简单介绍一下熵解码。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=True,
    # ---- 新增参数：基于内层熵解码 ----
    entropy_decoding="trough",          # 或 "random_after" / None
    entropy_record_tokens=True,         # 是否记录各层 token 与最终 token
    return_dict_in_generate=True,       # 为了拿到额外统计信息，需返回字典
)

generated = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
print(generated[0])
```

#### 2. 通过 `generation_config` 预先配置

```python
gen_config = model.generation_config
gen_config.entropy_decoding = "random_after"
gen_config.entropy_record_tokens = True
gen_config.return_dict_in_generate = True

outputs = model.generate(**inputs, generation_config=gen_config)
```

---

### 四、输出中的新增字段说明

当你设置：

- `return_dict_in_generate=True` 且
- `generation_config.entropy_decoding` 不为 `None` 且
- 模型为 `GptOssForCausalLM`（decoder-only, `model.config.model_type == "gpt_oss"`），

解码结果类型为 `GenerateDecoderOnlyOutput`，在原有字段基础上新增三个可选字段：

- **`entropy_per_layer_token_ids: Optional[tuple[torch.LongTensor]]`**
  - 形状：长度为 `num_generated_tokens` 的 tuple；
  - 第 `t` 个元素是一个形状为 `[batch_size, num_layers]` 的张量；
  - 含义：在第 `t` 步解码时，每个样本、每一层对应的 **原始 logits argmax token_id**。  
    - 最后一列（`[..., -1]`）即为末层的 argmax；
    - 其余列则是各中间层的 argmax。

- **`entropy_selected_token_ids: Optional[tuple[torch.LongTensor]]`**
  - 形状：同样是长度为 `num_generated_tokens` 的 tuple；
  - 第 `t` 个元素是形状为 `[batch_size]` 的张量；
  - 含义：第 `t` 步最终真正 **被解码出来的 token_id**（即在选层 + logits_processor + 采样 / 贪心之后的结果）；  
    它与 `outputs.sequences` 中新增位置的 token 对应，用于和各层原始 argmax 做对比分析。

- **`entropy_selected_layer_indices: Optional[tuple[torch.LongTensor]]`**
  - 形状：长度为 `num_generated_tokens` 的 tuple；
  - 第 `t` 个元素是形状为 `[batch_size]` 的张量；
  - 含义：第 `t` 步解码时，每个样本实际**选择的层索引**（0 到 `num_layers-1`）；  
    该字段在启用 `entropy_decoding` 时**始终可用**（无需设置 `entropy_record_tokens=True`），用于分析解码过程中各步实际使用的层分布。
  - 示例值：
    - 如果某步某个样本的值为 `5`，表示该步使用了第 5 层（0-indexed，即第 6 层）的 logits 进行解码；
    - 如果值为 `num_layers-1`，表示使用了末层（与默认解码行为一致）。

示例读取方式：

```python
out = outputs  # GenerateDecoderOnlyOutput

per_step_layer_tokens = out.entropy_per_layer_token_ids    # tuple[step] of [B, L] (需 entropy_record_tokens=True)
per_step_selected = out.entropy_selected_token_ids         # tuple[step] of [B] (需 entropy_record_tokens=True)
per_step_layer_indices = out.entropy_selected_layer_indices  # tuple[step] of [B] (始终可用)

step_idx = 0
batch_idx = 0

# 查看各层 token（需 entropy_record_tokens=True）
if per_step_layer_tokens is not None:
    layer_tokens_step0 = per_step_layer_tokens[step_idx][batch_idx]  # [L]
    selected_token_step0 = per_step_selected[step_idx][batch_idx]    # scalar
    print("第 0 步各层 argmax token_ids:", layer_tokens_step0)
    print("第 0 步最终策略选择 token_id:", selected_token_step0)

# 查看选择的层索引（始终可用）
selected_layer_idx_step0 = per_step_layer_indices[step_idx][batch_idx]  # scalar
print("第 0 步实际选择的层索引:", selected_layer_idx_step0.item())

# 分析整个解码过程的层选择分布
all_layer_indices = torch.stack([idx for idx in per_step_layer_indices], dim=0)  # [T, B]
print("所有步骤的层索引分布:", all_layer_indices)
```

---

### 五、设计细节与限制

- **仅作用于非 beam 解码**：  
  当 `num_beams == 1` 且模式为普通 `greedy` 或 `sample` 时才会使用 `_entropy_decoding`，否则仍走原有 beam / group beam / constrained beam 等逻辑。

- **只对 gpt-oss 生效**：  
  内部通过 `config.model_type == "gpt_oss"` 与存在 `lm_head` 属性做检查，其他模型即便传入 `entropy_decoding` 也会自动退化为普通解码。

- **策略为全程固定**：  
  - 一旦在 `GenerationConfig` 中指定了 `entropy_decoding="trough"` 或 `"random_after"`，整个解码过程中都会按照该策略选层；
  - 不再使用 vLLM 实现中的 `p_strategy` 按概率混合末层策略。

- **熵与选层依据**：
  - 每一步，对于所有层的最终位置 hidden state `h_i`（经过 norm 处理），使用同一个 `lm_head` 计算 logits；
  - 对每层 logits **先应用 logits_processor**（温度、top-k、top-p 等），得到处理后的 scores；
  - 基于处理后的 scores 计算信息熵（默认按词表归一化、以 2 为底），得到 `[B]` 形状的熵向量；
  - 借助 `_select_layers_from_entropies` 对不同层熵序列做逐样本“熵谷”选择，得到基础层索引；
  - 在此基础上根据 `entropy_decoding` 决定是否进一步进行 `"random_after"` 等策略变换。

- **输出字段的可用性**：
  - `entropy_selected_layer_indices`：只要启用 `entropy_decoding` 且 `return_dict_in_generate=True`，该字段就会自动填充，**无需设置 `entropy_record_tokens`**；
  - `entropy_per_layer_token_ids` 和 `entropy_selected_token_ids`：需要额外设置 `entropy_record_tokens=True` 才会填充，用于更详细的 token 级别分析。

---

### 六、迁移自 vLLM 的部分说明

- 辅助函数 `calculate_multi_layer_entropy_selection` 已经从 vLLM 风格的接口适配为适合 Transformers 的版本：
  - 输入为当前 step 各层的 hidden states（已裁剪为 `[B, H]`，且经过 norm 处理）与 `lm_head`；
  - 内部直接计算每层 logits 与熵，并使用 `_select_layers_from_entropies` 得到熵谷层索引；
  - 返回：`(selected_idx_trough, entropies_per_layer, logits_per_layer)`，供 `_entropy_decoding` 使用。
- **重要变更**：在 `_entropy_decoding` 中，熵计算和层选择基于**处理后的 scores**（即经过 `logits_processor` 处理后的 logits），而非 raw logits，以确保熵值反映实际用于解码的分布特征。
- vLLM 中与 `logits_processor`、`sampling_metadata` 相关的显存友好实现不再直接使用，转而由 `generate` 内部的标准流程负责裁剪与约束。

如需进一步扩展策略（例如加入一致性因子、embedding 相似度等），可以在保持现有接口的前提下，在 `_entropy_decoding` 中基于 `entropies_per_layer` 与 `logits_per_layer` 做更复杂的打分与选层。 


