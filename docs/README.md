# Cautious Decoding 文档

欢迎来到 Cautious Decoding 项目文档！

## 文档目录

### 核心功能

- **[GPU使用指南](./GPU_Usage.md)**
  - 如何指定和使用GPU
  - 单GPU和多GPU配置
  - 常见问题解答

- **[Engine API文档](./Engine_API.md)**
  - Engine类的完整API说明
  - 初始化参数和方法详解
  - 使用示例

### 算法文档

- **[Conservative Dynamic策略](../src/analysis/tokens/Dynamic_Decoding.md)**
  - 算法原理和数学描述
  - 两种策略对比

- **[Conservative Dynamic实现说明](./Conservative_Dynamic_Implementation.md)**
  - 实现细节和代码修改
  - 核心算法流程
  - 性能考虑和优化建议

- **[Conservative Dynamic使用指南](../gpqa/baselines/extensions/CONSERVATIVE_DYNAMIC_USAGE.md)**
  - 如何使用Conservative Dynamic生成
  - 参数说明和示例代码

### 分析文档

- **[层级选择机制](../src/analysis/tokens/SELECTION_MECHANISM.md)**
  - 熵谷解码的选择机制详解

- **[层级选择说明](../src/analysis/tokens/LAYER_SELECTION_EXPLAINED.md)**
  - 层级选择的基础知识

## 快速开始

### 1. 查看GPU信息

```python
from extensions.llms.engine import print_gpu_info
print_gpu_info()
```

### 2. 初始化Engine

```python
from extensions.llms.engine import Engine

# 使用GPU 0
engine = Engine(model_name="your-model", device_id=0)
```

### 3. 使用Conservative Dynamic生成

```python
from extensions.generate import generate_conservative_dynamic

response = generate_conservative_dynamic(
    engine=engine,
    prompt="你的问题",
    max_new_tokens=512,
    verbose=True
)
```

## 项目结构

```
Cautious_Decoding/
├── docs/                                    # 文档目录
│   ├── README.md                           # 本文档
│   ├── GPU_Usage.md                        # GPU使用指南
│   ├── Engine_API.md                       # Engine API文档
│   └── Conservative_Dynamic_Implementation.md  # Conservative Dynamic实现说明
├── gpqa/
│   └── baselines/
│       └── extensions/                     # 扩展功能
│           ├── generate.py                # 生成函数
│           └── llms/
│               └── engine.py             # Engine类
└── src/
    └── analysis/
        └── tokens/                         # 分析文档
```

## 主要功能

### 1. GPU管理

- 自动GPU选择
- 指定GPU ID
- 多GPU自动分配
- GPU信息查询

### 2. Conservative Dynamic策略

- 从末层向前查找熵最低的层
- 早停机制优化性能
- 详细的层选择信息

### 3. 层级分析

- 获取各层logits
- 计算每层熵值
- 层级选择统计

## 常见问题

### Q: 如何指定使用哪个GPU？

A: 使用 `device_id` 参数：
```python
engine = Engine(model_name="model", device_id=0)  # 使用GPU 0
```

详见 [GPU使用指南](./GPU_Usage.md)

### Q: 如何使用Conservative Dynamic策略？

A: 使用 `generate_conservative_dynamic` 函数：
```python
response = generate_conservative_dynamic(engine, prompt, max_new_tokens=512)
```

详见 [Conservative Dynamic使用指南](../gpqa/baselines/extensions/CONSERVATIVE_DYNAMIC_USAGE.md)

### Q: 如何获取各层的logits？

A: 使用 `get_per_layer_logits` 方法：
```python
per_layer_logits, final_logits = engine.get_per_layer_logits(prompt)
```

详见 [Engine API文档](./Engine_API.md)

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

请查看项目根目录的LICENSE文件。
