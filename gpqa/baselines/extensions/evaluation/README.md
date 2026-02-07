# Conservative Dynamic Perturbation Analysis

本目录包含用于验证Conservative Dynamic策略中"熵谷之后的层是否带来负面扰动"的评估工具。

## 功能概述

1. **分歧点收集** (`divergence_analysis.py`): 找出回退层与末层预测不同的token位置
2. **路径对比** (`path_comparison.py`): 从分歧点的两个token分别继续生成，对比结果
3. **扰动分析** (`perturbation_evaluation.py`): 分析扰动从第几层开始
4. **可视化** (`visualization.py`): 绘制熵值变化曲线和准确率对比图
5. **主评估脚本** (`run_evaluation.py`): 整合所有功能，运行完整评估流程

## 使用方法

### 基本使用

评估需在 GPU 上跑时，请用 `--device_id` 指定卡号（不指定则自动选卡）：

```bash
cd gpqa/baselines/extensions/evaluation
python run_evaluation.py \
    --model "your-model-name" \
    --data "path/to/gpqa_diamond.csv" \
    --output "evaluation_results" \
    --device_id 0 \
    --verbose
```

若希望只用某张物理卡（例如仅用 1 号卡），可先设置环境变量再运行：`CUDA_VISIBLE_DEVICES=1 python run_evaluation.py ... --device_id 0`（此时 0 即对应物理 1 号卡）。

### 参数说明

- `--model`: 模型名称或路径（必需）
- `--data`: GPQA-diamond数据集CSV文件路径（必需）
- `--output`: 输出目录（默认：evaluation_results）
- `--max_examples`: 最大评估样本数（默认：全部）
- `--max_tokens`: 最大生成token数（默认：512）
- `--device_id`: 指定 GPU 编号（0, 1, 2, ...）；不传则自动选卡。**需要在 GPU 上跑时请显式传入**，例如 `--device_id 0` 使用第一张卡
- `--seed`: 随机种子（默认：42）
- `--verbose`: 打印详细信息
- `--sampling_k`: 若大于 0，对前 N 个分歧点做 k 次采样路径对比（默认：0）
- `--sampling_max_divergence_points`: 参与多采样对比的最大分歧点数量（默认：20）
- `--sampling_temperature`: 多采样时的温度（默认：0.7）

### 输出文件

评估完成后，会在输出目录生成以下文件：

1. `divergence_points_*.json`: 所有分歧点的详细信息
2. `path_comparisons_*.json`: 路径对比结果
3. `perturbation_analyses_*.json`: 扰动分析结果
4. `evaluation_results_*.json`: 综合评估统计
5. `report_*.txt`: 文本报告
6. `entropy_trends.png`: 熵值变化曲线图
7. `accuracy_comparison.png`: 准确率对比图

## 代码示例

### 单独使用各个模块

```python
from llms.engine import Engine
from extensions.utils.gpqa_loader import load_gpqa_diamond
from extensions.evaluation.divergence_analysis import collect_divergence_points
from extensions.evaluation.path_comparison import compare_paths_for_divergence_point
from extensions.evaluation.perturbation_evaluation import analyze_perturbation_start, evaluate_perturbation_impact

# 1. 加载数据集
dataset = load_gpqa_diamond("path/to/gpqa_diamond.csv")

# 2. 初始化Engine
engine = Engine(model_name="your-model", device_id=0)

# 3. 收集分歧点
divergence_points = collect_divergence_points(
    engine=engine,
    dataset=dataset,
    max_new_tokens=512,
    verbose=True
)

# 4. 分析单个分歧点的扰动起始层
for div_point in divergence_points[:5]:  # 分析前5个
    analysis = analyze_perturbation_start(div_point)
    print(f"Perturbation starts at layer: {analysis['perturbation_start_layer']}")
```

## 评估指标

### Token级别
- 分歧点比例：回退层与末层预测不同的token占比
- Token差异数量：两个路径中不同token的数量
- 熵值变化幅度：从回退层到末层的熵值变化

### 答案级别
- 回退层路径准确率：使用回退层token生成的答案准确率
- 末层路径准确率：使用末层token生成的答案准确率
- 准确率提升/下降：两个路径的准确率差异

### 扰动分析
- 扰动起始层分布：扰动从哪些层开始
- 负面子集（Path A 对且 Path B 错）的扰动起始层分布
- 按 trough 深度（final_layer - selected_layer）分层：shallow / mid / deep 的正确率与起始层
- 扰动强度：熵值增加的幅度
- 受影响层数：从扰动起始层到末层的层数

## 文件结构

```
evaluation/
├── __init__.py                    # 模块初始化
├── divergence_analysis.py         # 分歧点分析
├── path_comparison.py             # 路径对比
├── perturbation_evaluation.py     # 扰动评估
├── visualization.py               # 可视化
├── run_evaluation.py              # 主评估脚本
└── README.md                      # 本文档
```

## 注意事项

1. **内存使用**: 评估过程需要计算所有层的logits，内存占用较大
2. **计算时间**: 每个样本需要多次前向传播，评估时间较长
3. **GPU要求**: 建议使用GPU加速，CPU运行会很慢
4. **数据集格式**: 确保GPQA数据集CSV文件包含以下列：
   - Question
   - Correct Answer
   - Incorrect Answer 1
   - Incorrect Answer 2
   - Incorrect Answer 3

## 故障排除

### 问题：无法找到分歧点

**可能原因**:
- 模型在大多数位置回退层和末层预测相同
- 生成token数太少

**解决方法**:
- 增加 `max_new_tokens` 参数
- 检查模型是否正常工作

### 问题：路径对比失败

**可能原因**:
- 上下文过长导致OOM
- Tokenizer编码问题

**解决方法**:
- 减少 `max_new_tokens`
- 检查tokenizer是否正确处理特殊字符

### 问题：可视化失败

**可能原因**:
- matplotlib未安装
- 数据格式不正确

**解决方法**:
```bash
pip install matplotlib numpy
```

## 可选：使用 transformers-dynamic + GptOss 收集分歧点

若需与 HF 的 `_entropy_decoding` 代码路径一致，可使用 `run_evaluation_hf_gpt_oss.py`（需将 transformers-dynamic 与 gpqa baselines 加入 PYTHONPATH）：

```bash
cd /path/to/Dynamic-Decoding
PYTHONPATH="transformers-dynamic:gpqa/baselines:$PYTHONPATH" python gpqa/baselines/extensions/evaluation/run_evaluation_hf_gpt_oss.py \
  --model /path/to/gpt-oss-checkpoint \
  --data gpqa/baselines/dataset/gpqa_diamond.csv \
  --output evaluation_results_hf \
  --device_id 0 \
  --max_examples 5 \
  --max_tokens 8192 \
  --run_path_comparison \
  --verbose
```

输出与主流程兼容的 `divergence_points_hf.json` 及可选的 `path_comparisons_hf.json`，可用于同一套扰动分析与可视化（需将 JSON 键与 gpqa 格式对齐）。

## 相关文档

- [Conservative Dynamic实现说明](../../../docs/Conservative_Dynamic_Implementation.md)
- [GPU使用指南](../../../docs/GPU_Usage.md)
- [Engine API文档](../../../docs/Engine_API.md)
