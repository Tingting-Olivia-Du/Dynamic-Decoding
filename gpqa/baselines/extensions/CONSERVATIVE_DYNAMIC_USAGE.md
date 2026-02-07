# Conservative Dynamic 使用指南

## 概述

Conservative Dynamic 是一种基于熵的动态层选择策略，从末层向前查找熵最低的层，使用该层的预测作为输出。

## 策略原理

根据 `Dynamic_Decoding.md` 中的描述：

- **思想**：从末层向前查看，只要发现更早层的熵更低，就用该层；一旦不再降低，立即停止查看。
- **数学原理**：
  - 记第 i 层熵为 H_i。自 L-1 到 0 依次检查，维护当前最佳 H_best 与层索引 j：
    - 若 H_i < H_best 则更新 H_best ← H_i, j ← i
    - 否则对该样本早停。最终选择层为 j。

## 使用方法

### 基本用法

```python
from extensions.generate import generate_conservative_dynamic
from extensions.llms.engine import Engine, print_gpu_info

# 查看可用GPU
print_gpu_info()

# 初始化引擎 - 方式1: 自动选择GPU
engine = Engine(model_name="your-model-name")

# 初始化引擎 - 方式2: 指定GPU ID
engine = Engine(model_name="your-model-name", device_id=0)  # 使用GPU 0

# 初始化引擎 - 方式3: 直接指定设备字符串
engine = Engine(model_name="your-model-name", device="cuda:1")  # 使用GPU 1

# 初始化引擎 - 方式4: 使用device_map（多GPU自动分配）
engine = Engine(model_name="your-model-name", device_map="auto")

# 生成回复
prompt = "你的问题"
response = generate_conservative_dynamic(
    engine=engine,
    prompt=prompt,
    max_new_tokens=512,
    verbose=True  # 打印每层的选择信息
)

print(response)
```

### 获取详细信息

```python
response, info = generate_conservative_dynamic(
    engine=engine,
    prompt=prompt,
    max_new_tokens=512,
    output_dict=True  # 返回详细信息
)

# info 包含：
# - selected_layers: 每个token选择的层索引列表
# - layer_entropies: 每个token的各层熵值列表
# - all_layer_tokens: 每个token的各层预测列表
# - generated_text: 生成的文本
# - generated_token_ids: 生成的token ID列表

print(f"选择的层: {info['selected_layers']}")
print(f"第一层熵值: {info['layer_entropies'][0]}")
```

### 参数说明

- `engine`: Engine对象，已初始化的LLM引擎
- `prompt`: 输入提示文本
- `max_length`: 最大输入长度（默认1024）
- `max_new_tokens`: 最大生成token数（默认512）
- `output_dict`: 是否返回详细信息（默认False）
- `verbose`: 是否打印调试信息（默认False）
- `**kwargs`: 其他参数（当前未使用）

## 实现细节

### 核心函数

1. **`Engine.get_per_layer_logits()`**: 获取每一层的logits
   - 手动遍历每一层，计算每层的hidden states
   - 通过lm_head获取每层的logits

2. **`generate_conservative_dynamic()`**: 实现Conservative Dynamic策略
   - 逐token生成
   - 对每个token，计算所有层的熵
   - 从末层向前查找熵最低的层
   - 使用选中层的预测

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

## 注意事项

1. **性能**: 由于需要计算所有层的logits，比标准生成慢
2. **内存**: 需要存储所有层的logits，内存占用较大
3. **模型兼容性**: 支持常见的transformer架构（GPT、LLaMA、Qwen等）
4. **错误处理**: 如果获取各层logits失败，会自动回退到使用最终层

## 示例输出

当 `verbose=True` 时，会打印类似以下信息：

```
Token 0: Selected layer 22/23, entropy=8.5234, token='The'
Token 1: Selected layer 23/23, entropy=7.8912, token=' answer'
Token 2: Selected layer 22/23, entropy=8.1234, token=' is'
...
```

## 与标准生成的对比

```python
# 标准生成
from extensions.generate import generate_simple
response_standard = generate_simple(engine, prompt, max_new_tokens=512)

# Conservative Dynamic生成
response_dynamic = generate_conservative_dynamic(engine, prompt, max_new_tokens=512)

# 比较结果
print("Standard:", response_standard)
print("Dynamic:", response_dynamic)
```

## 评估准确性

要评估Conservative Dynamic生成的准确性，可以：

1. 在测试集上运行生成
2. 收集 `output_dict=True` 的详细信息
3. 分析选择的层分布和熵值
4. 与标准生成结果对比准确率

```python
# 评估示例
results = []
for prompt in test_prompts:
    response, info = generate_conservative_dynamic(
        engine, prompt, max_new_tokens=512, output_dict=True
    )
    results.append({
        'prompt': prompt,
        'response': response,
        'selected_layers': info['selected_layers'],
        'avg_entropy': sum(info['layer_entropies']) / len(info['layer_entropies'])
    })
```
