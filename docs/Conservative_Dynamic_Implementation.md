# Conservative Dynamic 实现说明

本文档详细说明 Conservative Dynamic 策略的实现细节和代码修改。

## 目录

- [实现概述](#实现概述)
- [代码修改](#代码修改)
- [核心算法](#核心算法)
- [使用示例](#使用示例)
- [性能考虑](#性能考虑)

## 实现概述

Conservative Dynamic 是一种基于熵的动态层选择策略，从末层向前查找熵最低的层，使用该层的预测作为输出。

### 策略原理

根据 `Dynamic_Decoding.md` 中的描述：

- **思想**：从末层向前查看，只要发现更早层的熵更低，就用该层；一旦不再降低，立即停止查看。
- **数学原理**：
  - 记第 i 层熵为 H_i。自 L-1 到 0 依次检查，维护当前最佳 H_best 与层索引 j：
    - 若 H_i < H_best 则更新 H_best ← H_i, j ← i
    - 否则对该样本早停。最终选择层为 j。

## 代码修改

### 1. Engine 类扩展 (`llms/engine.py`)

#### 新增方法：`get_per_layer_logits`

```python
def get_per_layer_logits(self, prompt: str, max_length: int = 1024) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    获取每一层的logits输出（用于Conservative Dynamic策略）。
    
    通过手动遍历每一层，计算每层的hidden states，然后通过lm_head得到logits。
    """
```

**实现特点**：
- 手动遍历每一层，避免使用hooks（更可靠）
- 支持多种模型架构（GPT、LLaMA、Qwen等）
- 返回所有层的logits和最终logits

**支持的模型架构**：
- GPT-style: `transformer.h` + `transformer.wte`
- LLaMA/Qwen-style: `transformer.layers` + `transformer.embed_tokens`
- Decoder-style: `transformer.decoder.layers` + `transformer.decoder.embed_tokens`

#### GPU支持扩展

新增了以下功能：
- `device_id` 参数：直接指定GPU ID
- `device_map` 参数：支持多GPU自动分配
- `get_available_gpus()` 函数：获取GPU信息
- `print_gpu_info()` 函数：打印GPU信息

详见 [GPU使用指南](./GPU_Usage.md)

### 2. 新增生成函数 (`generate.py`)

#### 新增函数：`generate_conservative_dynamic`

```python
def generate_conservative_dynamic(
    engine: Engine, 
    prompt: str, 
    max_length: int = 1024, 
    max_new_tokens: int = 512, 
    output_dict: bool = False, 
    verbose: bool = False, 
    **kwargs
) -> str | tuple[str, dict]:
```

**功能**：
- 逐token生成，每个token都选择最优层
- 实现Conservative Dynamic策略
- 支持返回详细信息（选择的层、熵值等）

## 核心算法

### 算法流程

```
对于每个要生成的token:
  1. 获取所有层的logits
  2. 计算每层的熵值
  3. 从最后一层（L-1）开始向前遍历:
     - 如果当前层熵 < 最佳熵:
       → 更新最佳熵和选中层索引
     - 否则:
       → 早停，使用当前最佳层
  4. 输出选中层预测的token
```

### 代码实现

```python
# Conservative Dynamic策略：从末层向前查找熵最低的层
selected_layer_idx = num_layers - 1  # 默认选择最后一层
best_entropy = layer_entropies[selected_layer_idx]  # 初始化为最后一层的熵

for layer_idx in range(num_layers - 2, -1, -1):  # 从倒数第二层向前
    if layer_entropies[layer_idx] < best_entropy:
        best_entropy = layer_entropies[layer_idx]
        selected_layer_idx = layer_idx
    else:
        # 一旦熵不再降低，立即停止查看（早停）
        break

# 使用选中层的预测
selected_token_id = layer_token_ids[selected_layer_idx]
```

### 关键点

1. **早停机制**：一旦发现熵不再降低，立即停止查找，提高效率
2. **从末层开始**：优先考虑深层（通常质量更高）
3. **熵值比较**：使用信息熵衡量不确定性，选择最确定的层

## 使用示例

### 基本使用

```python
from extensions.llms.engine import Engine
from extensions.generate import generate_conservative_dynamic

# 初始化Engine
engine = Engine(
    model_name="meta-llama/Llama-2-7b-hf",
    device_id=0
)

# 使用Conservative Dynamic生成
response = generate_conservative_dynamic(
    engine=engine,
    prompt="What is machine learning?",
    max_new_tokens=512,
    verbose=True  # 打印每层的选择信息
)

print(response)
```

### 获取详细信息

```python
response, info = generate_conservative_dynamic(
    engine=engine,
    prompt="Explain quantum computing",
    max_new_tokens=512,
    output_dict=True  # 返回详细信息
)

# info包含：
# - selected_layers: 每个token选择的层索引列表
# - layer_entropies: 每个token的各层熵值列表
# - all_layer_tokens: 每个token的各层预测列表
# - generated_text: 生成的文本
# - generated_token_ids: 生成的token ID列表

print(f"选择的层: {info['selected_layers']}")
print(f"第一层熵值: {info['layer_entropies'][0]}")
```

### 分析层选择模式

```python
import numpy as np

response, info = generate_conservative_dynamic(
    engine=engine,
    prompt="Your question",
    max_new_tokens=512,
    output_dict=True
)

# 统计层选择
selected_layers = info['selected_layers']
layer_counts = {}
for layer in selected_layers:
    layer_counts[layer] = layer_counts.get(layer, 0) + 1

print("层选择统计:")
for layer, count in sorted(layer_counts.items()):
    print(f"  Layer {layer}: {count}次 ({count/len(selected_layers)*100:.1f}%)")

# 计算平均熵值
avg_entropies = [np.mean(entropies) for entropies in info['layer_entropies']]
print(f"\n平均熵值: {np.mean(avg_entropies):.4f}")
```

## 性能考虑

### 计算开销

1. **逐token生成**：每个token都需要计算所有层的logits
2. **内存占用**：需要存储所有层的logits
3. **时间开销**：比标准生成慢（需要遍历所有层）

### 优化建议

1. **批量处理**：如果可能，考虑批量生成多个样本
2. **缓存机制**：对于相同的prompt，可以缓存各层logits
3. **早停优化**：当前实现已有早停机制，减少不必要的层遍历

### 性能对比

```python
import time

# 标准生成
start = time.time()
response1 = generate_simple(engine, prompt, max_new_tokens=512)
time1 = time.time() - start

# Conservative Dynamic生成
start = time.time()
response2 = generate_conservative_dynamic(engine, prompt, max_new_tokens=512)
time2 = time.time() - start

print(f"标准生成: {time1:.2f}秒")
print(f"Conservative Dynamic: {time2:.2f}秒")
print(f"速度比: {time2/time1:.2f}x")
```

## 错误处理

### 模型结构不支持

如果模型结构不被识别，会抛出 `ValueError`：

```python
try:
    per_layer_logits, _ = engine.get_per_layer_logits(prompt)
except ValueError as e:
    print(f"错误: {e}")
    # 回退到使用最终层
```

### 获取各层logits失败

`generate_conservative_dynamic` 会自动回退到使用最终层：

```python
# 在generate_conservative_dynamic内部
try:
    per_layer_logits, final_logits = engine.get_per_layer_logits(current_prompt)
except Exception as e:
    if verbose:
        print(f"Warning: Failed to get per-layer logits: {e}. Falling back to final layer.")
    # 回退到使用最终层
    per_layer_logits = [final_logits]
```

## 实现细节

### 熵计算

使用 `calculate_information_entropy` 函数计算熵：

```python
from extensions.utils.entropy import calculate_information_entropy

# logits shape: [batch_size, vocab_size]
entropy = calculate_information_entropy(logits).item()
```

公式：$H = -\sum p(x) \log_2 p(x)$

### 层遍历顺序

从末层（L-1）向前遍历到第0层：

```python
for layer_idx in range(num_layers - 2, -1, -1):  # 从倒数第二层向前
    # 处理逻辑
```

**为什么从倒数第二层开始？**
- 最后一层已经作为初始最佳值
- 从倒数第二层开始比较，一旦熵不再降低就早停

## 与标准生成的对比

### 标准生成

```python
from extensions.generate import generate_simple

response = generate_simple(
    engine=engine,
    prompt=prompt,
    max_new_tokens=512
)
```

- 使用最终层的预测
- 速度快
- 无法利用中间层的信息

### Conservative Dynamic生成

```python
from extensions.generate import generate_conservative_dynamic

response = generate_conservative_dynamic(
    engine=engine,
    prompt=prompt,
    max_new_tokens=512
)
```

- 动态选择最优层
- 可能选择更早的层（如果熵更低）
- 速度较慢，但可能质量更高

## 评估方法

### 准确性评估

```python
# 在测试集上运行
results = []
for prompt, expected in test_set:
    response = generate_conservative_dynamic(engine, prompt, max_new_tokens=512)
    accuracy = evaluate(response, expected)
    results.append(accuracy)

print(f"平均准确率: {np.mean(results):.2%}")
```

### 层选择分析

```python
response, info = generate_conservative_dynamic(
    engine, prompt, max_new_tokens=512, output_dict=True
)

# 分析哪些层被选择最多
from collections import Counter
layer_counts = Counter(info['selected_layers'])
print("最常选择的层:", layer_counts.most_common(5))
```

## 相关文档

- [GPU使用指南](./GPU_Usage.md)
- [Engine API文档](./Engine_API.md)
- [Conservative Dynamic使用指南](../gpqa/baselines/extensions/CONSERVATIVE_DYNAMIC_USAGE.md)
- [Dynamic Decoding原理](../src/analysis/tokens/Dynamic_Decoding.md)
