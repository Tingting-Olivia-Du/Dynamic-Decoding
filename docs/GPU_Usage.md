# GPU 使用指南

本文档说明如何在 Cautious Decoding 项目中指定和使用 GPU。

## 目录

- [查看可用GPU](#查看可用gpu)
- [Engine 初始化方式](#engine-初始化方式)
- [GPU 指定方法](#gpu-指定方法)
- [多GPU支持](#多gpu支持)
- [常见问题](#常见问题)

## 查看可用GPU

在初始化 Engine 之前，可以先查看系统中可用的 GPU：

```python
from extensions.llms.engine import print_gpu_info, get_available_gpus

# 方法1: 打印GPU信息
print_gpu_info()
# 输出示例:
# 可用GPU数量: 2
#   GPU 0: NVIDIA GeForce RTX 3090
#     总内存: 24.00 GB
#     已分配: 2.50 GB
#     已保留: 3.00 GB
#     可用: 21.00 GB
#   GPU 1: NVIDIA GeForce RTX 3090
#     ...

# 方法2: 获取GPU信息字典
gpus = get_available_gpus()
for gpu in gpus:
    print(f"GPU {gpu['id']}: {gpu['name']}")
    print(f"  总内存: {gpu['memory_total']:.2f} GB")
    print(f"  可用内存: {gpu['memory_total'] - gpu['memory_reserved']:.2f} GB")
```

## Engine 初始化方式

### 方式1: 自动选择GPU（推荐）

如果不指定设备，Engine 会自动选择可用的 GPU：

```python
from extensions.llms.engine import Engine

# 自动选择第一个可用GPU
engine = Engine(model_name="your-model-name")
```

### 方式2: 使用 device_id 参数

通过 `device_id` 参数指定 GPU ID（最简单的方式）：

```python
from extensions.llms.engine import Engine

# 使用GPU 0
engine = Engine(model_name="your-model-name", device_id=0)

# 使用GPU 1
engine = Engine(model_name="your-model-name", device_id=1)

# 使用GPU 2
engine = Engine(model_name="your-model-name", device_id=2)
```

**优点**：
- 简单直观
- 自动验证GPU是否存在
- 自动设置当前GPU设备

### 方式3: 使用 device 参数

直接指定设备字符串：

```python
from extensions.llms.engine import Engine

# 使用GPU 0
engine = Engine(model_name="your-model-name", device="cuda:0")

# 使用GPU 1
engine = Engine(model_name="your-model-name", device="cuda:1")

# 使用CPU
engine = Engine(model_name="your-model-name", device="cpu")

# 使用默认CUDA（第一个GPU）
engine = Engine(model_name="your-model-name", device="cuda")
```

### 方式4: 使用 device_map（多GPU自动分配）

对于大模型，可以使用 `device_map` 自动将模型分配到多个 GPU：

```python
from extensions.llms.engine import Engine

# 自动分配到多个GPU
engine = Engine(
    model_name="your-model-name",
    device_map="auto"  # 自动分配
)

# 其他device_map选项
engine = Engine(
    model_name="your-model-name",
    device_map="balanced"  # 平衡分配
)

engine = Engine(
    model_name="your-model-name",
    device_map="balanced_low_0"  # 平衡分配，优先使用GPU 0
)
```

## GPU 指定方法

### 参数优先级

当同时指定多个参数时，优先级如下：

1. **device_map**（最高优先级）
   - 如果指定了 `device_map`，会忽略 `device` 和 `device_id`
   - 用于多GPU自动分配

2. **device_id**
   - 如果指定了 `device_id`，会覆盖 `device` 参数中的GPU ID
   - 例如：`device="cuda:0"` + `device_id=1` → 实际使用 `cuda:1`

3. **device**
   - 直接指定设备字符串
   - 如果为 `None`，自动选择

### 完整示例

```python
from extensions.llms.engine import Engine, print_gpu_info

# 1. 查看可用GPU
print_gpu_info()

# 2. 方式1: 自动选择
engine1 = Engine(model_name="your-model-name")

# 3. 方式2: 指定GPU ID（推荐）
engine2 = Engine(model_name="your-model-name", device_id=0)

# 4. 方式3: 指定设备字符串
engine3 = Engine(model_name="your-model-name", device="cuda:1")

# 5. 方式4: 多GPU自动分配
engine4 = Engine(model_name="your-model-name", device_map="auto")

# 6. 组合使用（device_id会覆盖device中的GPU ID）
engine5 = Engine(
    model_name="your-model-name",
    device="cuda:0",      # 会被覆盖
    device_id=1,          # 实际使用GPU 1
    dtype=torch.bfloat16
)
```

## 多GPU支持

### device_map 选项

`device_map` 支持以下选项：

- **`"auto"`**: 自动将模型分配到所有可用GPU，尽量平衡内存使用
- **`"balanced"`**: 平衡分配，尽量让每个GPU的负载相等
- **`"balanced_low_0"`**: 平衡分配，但优先在GPU 0上放置更多层
- **`"sequential"`**: 按顺序分配，从GPU 0开始

### 多GPU使用示例

```python
from extensions.llms.engine import Engine

# 自动分配到多个GPU（适合大模型）
engine = Engine(
    model_name="meta-llama/Llama-2-70b-hf",
    device_map="auto",
    dtype=torch.bfloat16
)

# 使用engine进行生成
from extensions.generate import generate_conservative_dynamic

response = generate_conservative_dynamic(
    engine=engine,
    prompt="你的问题",
    max_new_tokens=512
)
```

### 注意事项

1. **device_map模式下的限制**：
   - 使用 `device_map` 时，模型会分布在多个GPU上
   - `get_per_layer_logits()` 方法可能需要特殊处理（当前实现假设所有层在同一设备）

2. **内存管理**：
   - 多GPU模式下，每个GPU的内存使用会更均衡
   - 适合无法在单GPU上加载的大模型

## 常见问题

### Q1: 如何检查当前使用的GPU？

```python
from extensions.llms.engine import Engine

engine = Engine(model_name="your-model-name", device_id=0)
print(f"当前设备: {engine.device}")

# 检查模型所在的设备
print(f"模型设备: {next(engine.model.parameters()).device}")
```

### Q2: GPU内存不足怎么办？

1. **使用更小的数据类型**：
   ```python
   engine = Engine(
       model_name="your-model-name",
       device_id=0,
       dtype=torch.float16  # 使用float16而不是bfloat16
   )
   ```

2. **使用多GPU**：
   ```python
   engine = Engine(
       model_name="your-model-name",
       device_map="auto"  # 自动分配到多个GPU
   )
   ```

3. **使用CPU（不推荐，速度慢）**：
   ```python
   engine = Engine(
       model_name="your-model-name",
       device="cpu"
   )
   ```

### Q3: 如何指定特定的GPU？

```python
# 方法1: 使用device_id（推荐）
engine = Engine(model_name="your-model-name", device_id=2)

# 方法2: 使用device字符串
engine = Engine(model_name="your-model-name", device="cuda:2")
```

### Q4: device_id 和 device 参数的区别？

- **device_id**: 只需要指定数字（0, 1, 2等），更简单
- **device**: 需要完整的设备字符串（"cuda:0", "cuda:1"等），更灵活（可以指定"cpu"）

推荐使用 `device_id`，因为它会自动验证GPU是否存在。

### Q5: 如何在不同GPU上运行多个模型？

```python
from extensions.llms.engine import Engine

# 在GPU 0上加载模型1
engine1 = Engine(model_name="model1", device_id=0)

# 在GPU 1上加载模型2
engine2 = Engine(model_name="model2", device_id=1)

# 可以同时使用两个模型
response1 = generate_conservative_dynamic(engine1, prompt1)
response2 = generate_conservative_dynamic(engine2, prompt2)
```

### Q6: 使用 device_map 时如何知道模型分配到了哪些GPU？

```python
from extensions.llms.engine import Engine

engine = Engine(
    model_name="your-model-name",
    device_map="auto"
)

# 检查模型的设备映射
for name, param in engine.model.named_parameters():
    print(f"{name}: {param.device}")
```

## 完整示例

```python
from extensions.llms.engine import Engine, print_gpu_info
from extensions.generate import generate_conservative_dynamic

# 1. 查看GPU信息
print("=== GPU信息 ===")
print_gpu_info()

# 2. 初始化Engine（使用GPU 0）
print("\n=== 初始化Engine ===")
engine = Engine(
    model_name="your-model-name",
    device_id=0,
    dtype=torch.bfloat16
)
print(f"使用设备: {engine.device}")

# 3. 使用Conservative Dynamic生成
print("\n=== 生成回复 ===")
response = generate_conservative_dynamic(
    engine=engine,
    prompt="你的问题",
    max_new_tokens=512,
    verbose=True
)
print(f"回复: {response}")
```

## 相关文档

- [Conservative Dynamic 使用指南](../gpqa/baselines/extensions/CONSERVATIVE_DYNAMIC_USAGE.md)
- [Engine API 文档](./Engine_API.md)（如果存在）
