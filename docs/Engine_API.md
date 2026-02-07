# Engine API 文档

本文档详细说明 `Engine` 类的 API 和使用方法。

## 目录

- [Engine 类概述](#engine-类概述)
- [初始化参数](#初始化参数)
- [主要方法](#主要方法)
- [辅助函数](#辅助函数)
- [使用示例](#使用示例)

## Engine 类概述

`Engine` 类是用于加载和使用大语言模型（LLM）的核心类，支持单GPU、多GPU和CPU运行。

**位置**: `gpqa/baselines/extensions/llms/engine.py`

## 初始化参数

### `__init__` 方法

```python
Engine(
    model_name: str = "gpt2",
    device: str = None,
    device_id: int = None,
    dtype = torch.bfloat16,
    device_map: str = None
)
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | str | `"gpt2"` | 模型名称或路径（HuggingFace模型ID或本地路径） |
| `device` | str | `None` | 设备字符串（'cuda', 'cpu', 'cuda:0', 'cuda:1'等）。如果为None，自动选择 |
| `device_id` | int | `None` | GPU设备ID（0, 1, 2等）。如果指定，会覆盖device参数中的GPU ID |
| `dtype` | torch.dtype | `torch.bfloat16` | 使用的torch数据类型 |
| `device_map` | str | `None` | 设备映射（用于多GPU，如'auto', 'balanced'等）。如果指定，会使用device_map而不是device参数 |

#### 参数优先级

1. **device_map**（最高优先级）- 如果指定，忽略device和device_id
2. **device_id** - 如果指定，覆盖device中的GPU ID
3. **device** - 直接指定设备，如果为None则自动选择

### 初始化示例

```python
from extensions.llms.engine import Engine
import torch

# 示例1: 自动选择GPU
engine1 = Engine(model_name="meta-llama/Llama-2-7b-hf")

# 示例2: 指定GPU ID
engine2 = Engine(
    model_name="meta-llama/Llama-2-7b-hf",
    device_id=0
)

# 示例3: 指定设备字符串
engine3 = Engine(
    model_name="meta-llama/Llama-2-7b-hf",
    device="cuda:1"
)

# 示例4: 多GPU自动分配
engine4 = Engine(
    model_name="meta-llama/Llama-2-70b-hf",
    device_map="auto"
)

# 示例5: 使用CPU
engine5 = Engine(
    model_name="gpt2",
    device="cpu"
)

# 示例6: 自定义数据类型
engine6 = Engine(
    model_name="meta-llama/Llama-2-7b-hf",
    device_id=0,
    dtype=torch.float16
)
```

## 主要方法

### `generate` 方法

进行文本生成。

```python
generate(
    prompt: str,
    max_length: int = 1024,
    max_new_tokens: int = 50,
    skip_special_tokens: bool = True,
    extra_output_keys: list[str] = None,
    **kwargs
) -> dict
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | - | 输入提示文本 |
| `max_length` | int | 1024 | 最大输入长度 |
| `max_new_tokens` | int | 50 | 最大生成token数目 |
| `skip_special_tokens` | bool | True | 解码时是否跳过特殊token |
| `extra_output_keys` | list[str] | None | 额外输出项的键列表 |
| `**kwargs` | - | - | 传递给model.generate()的其他参数 |

#### 返回值

返回包含以下键的字典：

- `text`: 完整生成的文本（包含输入）
- `sequence`: token ID序列
- `scores`: 每步的logits分数
- `generated_tokens`: 生成的token文本列表
- `generated_token_ids`: 生成的token ID列表
- 其他在`extra_output_keys`中指定的键

#### 示例

```python
result = engine.generate(
    prompt="What is the capital of France?",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)

print(result["text"])
print(result["generated_tokens"])
```

### `get_logits` 方法

获取模型的logits输出（用于单个前向传播）。

```python
get_logits(prompt: str, max_length: int = 1024) -> torch.Tensor
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | - | 输入提示文本 |
| `max_length` | int | 1024 | 最大输入长度 |

#### 返回值

`torch.Tensor`: 模型的logits输出，shape为 `[batch_size, seq_len, vocab_size]`

#### 示例

```python
logits = engine.get_logits("Hello, world!")
print(logits.shape)  # [1, seq_len, vocab_size]
```

### `get_per_layer_logits` 方法

获取每一层的logits输出（用于Conservative Dynamic策略）。

```python
get_per_layer_logits(
    prompt: str,
    max_length: int = 1024
) -> tuple[list[torch.Tensor], torch.Tensor]
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | - | 输入提示文本 |
| `max_length` | int | 1024 | 最大输入长度 |

#### 返回值

`tuple[list[torch.Tensor], torch.Tensor]`:
- 第一个元素：各层logits列表，每个元素的shape为 `[batch_size, seq_len, vocab_size]`
- 第二个元素：最终logits（与最后一层相同）

#### 实现细节

该方法通过手动遍历每一层，计算每层的hidden states，然后通过lm_head得到logits。支持以下模型架构：

- GPT-style (使用 `transformer.h` 和 `transformer.wte`)
- LLaMA/Qwen-style (使用 `transformer.layers` 和 `transformer.embed_tokens`)
- 其他使用 `decoder.layers` 的架构

#### 示例

```python
per_layer_logits, final_logits = engine.get_per_layer_logits("Hello, world!")
print(f"层数: {len(per_layer_logits)}")
print(f"每层logits shape: {per_layer_logits[0].shape}")
```

## 辅助函数

### `get_available_gpus` 函数

获取可用的GPU信息。

```python
get_available_gpus() -> list[dict]
```

#### 返回值

GPU信息字典列表，每个字典包含：

- `id`: GPU ID
- `name`: GPU名称
- `memory_total`: 总内存（GB）
- `memory_allocated`: 已分配内存（GB）
- `memory_reserved`: 已保留内存（GB）

#### 示例

```python
from extensions.llms.engine import get_available_gpus

gpus = get_available_gpus()
for gpu in gpus:
    print(f"GPU {gpu['id']}: {gpu['name']}")
    print(f"  总内存: {gpu['memory_total']:.2f} GB")
```

### `print_gpu_info` 函数

打印可用的GPU信息。

```python
print_gpu_info() -> None
```

#### 示例

```python
from extensions.llms.engine import print_gpu_info

print_gpu_info()
# 输出:
# 可用GPU数量: 2
#   GPU 0: NVIDIA GeForce RTX 3090
#     总内存: 24.00 GB
#     ...
```

## 使用示例

### 完整示例1: 基本使用

```python
from extensions.llms.engine import Engine, print_gpu_info

# 查看GPU信息
print_gpu_info()

# 初始化Engine
engine = Engine(
    model_name="gpt2",
    device_id=0
)

# 生成文本
result = engine.generate(
    prompt="The capital of France is",
    max_new_tokens=50
)

print(result["text"])
```

### 完整示例2: 使用Conservative Dynamic

```python
from extensions.llms.engine import Engine
from extensions.generate import generate_conservative_dynamic

# 初始化Engine
engine = Engine(
    model_name="meta-llama/Llama-2-7b-hf",
    device_id=0
)

# 使用Conservative Dynamic生成
response, info = generate_conservative_dynamic(
    engine=engine,
    prompt="What is machine learning?",
    max_new_tokens=512,
    output_dict=True,
    verbose=True
)

print(f"回复: {response}")
print(f"选择的层: {info['selected_layers']}")
```

### 完整示例3: 多GPU使用

```python
from extensions.llms.engine import Engine
from extensions.generate import generate_conservative_dynamic

# 使用多GPU自动分配
engine = Engine(
    model_name="meta-llama/Llama-2-70b-hf",
    device_map="auto",
    dtype=torch.bfloat16
)

# 生成文本
response = generate_conservative_dynamic(
    engine=engine,
    prompt="Explain quantum computing",
    max_new_tokens=512
)

print(response)
```

### 完整示例4: 获取各层logits

```python
from extensions.llms.engine import Engine
from extensions.utils.entropy import calculate_information_entropy

engine = Engine(model_name="gpt2", device_id=0)

# 获取各层logits
per_layer_logits, final_logits = engine.get_per_layer_logits("Hello, world!")

# 计算每层的熵
for i, logits in enumerate(per_layer_logits):
    # 获取最后一个位置的logits
    last_logits = logits[0, -1, :]
    entropy = calculate_information_entropy(last_logits.unsqueeze(0)).item()
    print(f"Layer {i}: entropy = {entropy:.4f}")
```

## 属性

### `device`

当前使用的设备字符串（如 `"cuda:0"`, `"cpu"`）。

```python
print(engine.device)  # "cuda:0"
```

### `tokenizer`

HuggingFace tokenizer对象。

```python
tokens = engine.tokenizer.encode("Hello, world!")
```

### `model`

HuggingFace模型对象。

```python
# 访问模型参数
for name, param in engine.model.named_parameters():
    print(f"{name}: {param.device}")
```

### `dtype`

使用的数据类型。

```python
print(engine.dtype)  # torch.bfloat16
```

## 错误处理

### GPU不存在

```python
try:
    engine = Engine(model_name="gpt2", device_id=999)
except ValueError as e:
    print(f"错误: {e}")  # GPU 999不存在
```

### CUDA不可用

```python
try:
    engine = Engine(model_name="gpt2", device="cuda:0")
except ValueError as e:
    print(f"错误: {e}")  # CUDA不可用
```

### 模型结构不支持

```python
try:
    per_layer_logits, _ = engine.get_per_layer_logits("test")
except ValueError as e:
    print(f"错误: {e}")  # 无法识别模型结构
```

## 注意事项

1. **内存管理**: 大模型会占用大量GPU内存，使用前确保有足够空间
2. **设备一致性**: 使用`device_map`时，模型可能分布在多个GPU上
3. **数据类型**: `bfloat16`通常比`float16`更稳定，但需要GPU支持
4. **模型兼容性**: `get_per_layer_logits`支持常见架构，但某些特殊模型可能需要适配

## 相关文档

- [GPU使用指南](./GPU_Usage.md)
- [Conservative Dynamic使用指南](../gpqa/baselines/extensions/CONSERVATIVE_DYNAMIC_USAGE.md)
