"""
Main evaluation script for perturbation analysis.

cd gpqa/baselines/extensions/evaluation
python run_evaluation.py \
    --model "your-model-name" \
    --data "path/to/gpqa_diamond.csv" \
    --output "evaluation_results" \
    --device_id 0 \
    --verbose

"""
import sys
import os
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from llms.engine import Engine
from extensions.utils.gpqa_loader import load_gpqa_diamond, create_zero_shot_prompt
from extensions.evaluation.divergence_analysis import collect_divergence_points
from extensions.evaluation.path_comparison import (
    compare_paths_for_divergence_point,
    compare_paths_with_sampling,
)
from extensions.evaluation.perturbation_evaluation import analyze_perturbation_start, evaluate_perturbation_impact
from extensions.evaluation.visualization import create_all_visualizations


def run_evaluation(
    model_name: str,
    data_path: str,
    output_dir: str = "evaluation_results",
    max_examples: int = None,
    max_new_tokens: int = 512,
    device_id: int = None,
    seed: int = 42,
    verbose: bool = True,
    sampling_k: int = 0,
    sampling_max_divergence_points: int = 20,
    sampling_temperature: float = 0.7,
):
    """
    运行完整的评估流程。
    
    :param model_name: 模型名称或路径
    :param data_path: GPQA数据集路径
    :param output_dir: 输出目录
    :param max_examples: 最大评估样本数（None表示全部）
    :param max_new_tokens: 最大生成token数
    :param device_id: GPU设备ID
    :param seed: 随机种子
    :param verbose: 是否打印详细信息
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if verbose:
        print("=" * 80)
        print("Conservative Dynamic Perturbation Analysis")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Dataset: {data_path}")
        print(f"Output directory: {output_dir}")
        print()
    
    # 1. 加载数据集
    if verbose:
        print("Step 1: Loading GPQA-diamond dataset...")
    dataset = load_gpqa_diamond(data_path, seed=seed)
    if max_examples:
        dataset = dataset[:max_examples]
    if verbose:
        print(f"Loaded {len(dataset)} examples")
        print()
    
    # 2. 初始化Engine
    if verbose:
        print("Step 2: Initializing Engine...")
    engine = Engine(
        model_name=model_name,
        device_id=device_id,
        dtype=torch.bfloat16 if device_id is not None else torch.float32
    )
    if verbose:
        print(f"Engine initialized on device: {engine.device}")
        print()
    
    # 3. 收集分歧点
    if verbose:
        print("Step 3: Collecting divergence points...")
    divergence_points, collect_stats = collect_divergence_points(
        engine=engine,
        dataset=dataset,
        max_new_tokens=max_new_tokens,
        verbose=verbose,
        return_stats=True,
    )
    
    if not divergence_points:
        print("No divergence points found!")
        return

    total_token_steps = collect_stats.get("total_token_steps", 0)
    divergence_rate = (len(divergence_points) / total_token_steps) if total_token_steps > 0 else 0.0
    
    # 保存分歧点
    divergence_file = output_path / f"divergence_points_{timestamp}.json"
    with open(divergence_file, 'w', encoding='utf-8') as f:
        json.dump(divergence_points, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"Saved {len(divergence_points)} divergence points to {divergence_file}")
        print()
    
    # 4. 对比生成路径
    if verbose:
        print("Step 4: Comparing generation paths...")
    path_comparisons = []
    
    for i, div_point in enumerate(divergence_points):
        if verbose and i % 10 == 0:
            print(f"  Processing divergence point {i+1}/{len(divergence_points)}...")
        
        # 构建完整上下文
        base_prompt = create_zero_shot_prompt(dataset[div_point['question_id']])
        context_before = "".join(div_point.get('context_before', []))
        full_context = base_prompt + context_before
        
        # 对比路径
        try:
            comparison = compare_paths_for_divergence_point(
                engine=engine,
                divergence_point=div_point,
                full_context_before=full_context,
                max_new_tokens=max_new_tokens,
                verbose=False
            )
            path_comparisons.append(comparison)
        except Exception as e:
            if verbose:
                print(f"  Error comparing paths for divergence point {i}: {e}")
            path_comparisons.append({
                'path_a': {'is_correct': False, 'answer': None, 'text': ''},
                'path_b': {'is_correct': False, 'answer': None, 'text': ''},
                'accuracy_diff': 0.0
            })
    
    # 保存路径对比结果
    comparison_file = output_path / f"path_comparisons_{timestamp}.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(path_comparisons, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"Saved path comparisons to {comparison_file}")
        print()

    # 4b. 可选：多采样路径对比
    sampling_results = None
    if sampling_k > 0:
        if verbose:
            print("Step 4b: Running multi-sample path comparison...")
        n_sampling = min(sampling_max_divergence_points, len(divergence_points))
        sampling_results = []
        for i in range(n_sampling):
            if verbose and i % 5 == 0:
                print(f"  Sampling divergence point {i+1}/{n_sampling}...")
            div_point = divergence_points[i]
            base_prompt = create_zero_shot_prompt(dataset[div_point['question_id']])
            context_before = "".join(div_point.get('context_before', []))
            full_context = base_prompt + context_before
            try:
                res = compare_paths_with_sampling(
                    engine=engine,
                    divergence_point=div_point,
                    full_context_before=full_context,
                    max_new_tokens=max_new_tokens,
                    k=sampling_k,
                    temperature=sampling_temperature,
                    seed=seed,
                    verbose=False,
                )
                sampling_results.append(res)
            except Exception as e:
                if verbose:
                    print(f"  Error sampling divergence point {i}: {e}")
        if sampling_results:
            path_a_better_count = sum(1 for r in sampling_results if r.get('path_a_better', False))
            if verbose:
                print(f"  Path A better in {path_a_better_count}/{len(sampling_results)} divergence points (k={sampling_k})")
            sampling_file = output_path / f"sampling_comparisons_{timestamp}.json"
            with open(sampling_file, 'w', encoding='utf-8') as f:
                json.dump(sampling_results, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"Saved sampling comparisons to {sampling_file}")
        print()
    
    # 5. 分析扰动起始点
    if verbose:
        print("Step 5: Analyzing perturbation start points...")
    perturbation_analyses = []
    for div_point in divergence_points:
        analysis = analyze_perturbation_start(div_point)
        perturbation_analyses.append(analysis)
    
    # 保存扰动分析
    perturbation_file = output_path / f"perturbation_analyses_{timestamp}.json"
    with open(perturbation_file, 'w', encoding='utf-8') as f:
        json.dump(perturbation_analyses, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"Saved perturbation analyses to {perturbation_file}")
        print()
    
    # 6. 综合评估
    if verbose:
        print("Step 6: Evaluating overall perturbation impact...")
    evaluation_results = evaluate_perturbation_impact(
        divergence_points=divergence_points,
        path_comparisons=path_comparisons
    )
    evaluation_results["total_token_steps"] = total_token_steps
    evaluation_results["divergence_rate"] = divergence_rate
    evaluation_results["num_examples_processed"] = collect_stats.get("num_examples_processed", 0)

    # 保存评估结果
    eval_file = output_path / f"evaluation_results_{timestamp}.json"
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # 7. 生成文本报告
    if verbose:
        print("Step 7: Generating text report...")
    report_file = output_path / f"report_{timestamp}.txt"
    generate_text_report(
        evaluation_results,
        divergence_points,
        path_comparisons,
        perturbation_analyses,
        report_file
    )
    if verbose:
        print(f"Saved report to {report_file}")
        print()
    
    # 8. 生成可视化
    if verbose:
        print("Step 8: Creating visualizations...")
    try:
        create_all_visualizations(
            divergence_points=divergence_points,
            evaluation_results=evaluation_results,
            output_dir=output_path
        )
        if verbose:
            print("Visualizations created successfully")
            print()
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to create visualizations: {e}")
            print()
    
    # 打印摘要
    if verbose:
        print("=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        print(f"Total divergence points: {evaluation_results['total_divergence_points']}")
        print(f"Divergence rate: {evaluation_results.get('divergence_rate', 0):.2%}")
        print(f"Path A (selected layer) accuracy: {evaluation_results['path_a_accuracy']:.2%}")
        print(f"Path B (final layer) accuracy: {evaluation_results['path_b_accuracy']:.2%}")
        print(f"Accuracy improvement: {evaluation_results['accuracy_improvement']:.2%}")
        print(f"Average entropy increase: {evaluation_results['avg_entropy_increase']:.4f}")
        print(f"Average affected layers: {evaluation_results['avg_affected_layers']:.2f}")
        print("=" * 80)


def generate_text_report(
    evaluation_results: Dict,
    divergence_points: List[Dict],
    path_comparisons: List[Dict],
    perturbation_analyses: List[Dict],
    output_file: Path
):
    """生成文本报告。"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Conservative Dynamic Perturbation Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        # 总体统计
        f.write("Overall Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total divergence points: {evaluation_results['total_divergence_points']}\n")
        total_steps = evaluation_results.get('total_token_steps', 0)
        div_rate = evaluation_results.get('divergence_rate', 0.0)
        f.write(f"Total token steps (across examples): {total_steps}\n")
        f.write(f"Divergence rate: {div_rate:.2%}\n")
        f.write(f"Path A (selected layer) accuracy: {evaluation_results['path_a_accuracy']:.2%}\n")
        f.write(f"Path B (final layer) accuracy: {evaluation_results['path_b_accuracy']:.2%}\n")
        f.write(f"Accuracy improvement: {evaluation_results['accuracy_improvement']:.2%}\n")
        f.write(f"Average entropy increase: {evaluation_results['avg_entropy_increase']:.4f}\n")
        f.write(f"Average affected layers: {evaluation_results['avg_affected_layers']:.2f}\n\n")
        
        # 扰动起始层分布
        f.write("Perturbation Start Layer Distribution\n")
        f.write("-" * 80 + "\n")
        for layer, count in sorted(evaluation_results['perturbation_start_distribution'].items()):
            f.write(f"Layer {layer}: {count} occurrences\n")
        f.write("\n")

        # Path A 对且 Path B 错 子集的扰动起始层分布
        neg_count = evaluation_results.get('negative_subset_count', 0)
        neg_dist = evaluation_results.get('negative_subset_perturbation_start_distribution', {})
        f.write("Negative Subset (Path A correct, Path B wrong) Perturbation Start Distribution\n")
        f.write("-" * 80 + "\n")
        f.write(f"Negative subset size: {neg_count}\n")
        for layer, count in sorted(neg_dist.items()):
            f.write(f"Layer {layer}: {count} occurrences\n")
        f.write("\n")

        # 按 trough 深度分层
        by_depth = evaluation_results.get('by_trough_depth', {})
        f.write("By Trough Depth (final_layer - selected_layer)\n")
        f.write("-" * 80 + "\n")
        for label in ['shallow', 'mid', 'deep']:
            if label not in by_depth:
                continue
            d = by_depth[label]
            f.write(f"  {label}: count={d['count']}, Path A acc={d['path_a_accuracy']:.2%}, Path B acc={d['path_b_accuracy']:.2%}\n")
        f.write("\n")
        
        # 层熵值相关性
        f.write("Layer-Entropy Correlation\n")
        f.write("-" * 80 + "\n")
        for layer_idx in sorted(evaluation_results['layer_entropy_correlation'].keys()):
            corr = evaluation_results['layer_entropy_correlation'][layer_idx]
            f.write(f"Layer {layer_idx}: mean={corr['mean_entropy']:.4f}, "
                   f"std={corr['std_entropy']:.4f}, count={corr['count']}\n")
        f.write("\n")
        
        # 典型案例
        f.write("Sample Cases\n")
        f.write("-" * 80 + "\n")
        for i in range(min(5, len(divergence_points))):
            div_point = divergence_points[i]
            comparison = path_comparisons[i]
            analysis = perturbation_analyses[i]
            
            f.write(f"\nCase {i+1}:\n")
            f.write(f"  Question ID: {div_point['question_id']}\n")
            f.write(f"  Token Position: {div_point['token_position']}\n")
            f.write(f"  Selected Layer: {div_point['selected_layer']}, Token: '{div_point['selected_token']}'\n")
            f.write(f"  Final Layer: {div_point['final_layer']}, Token: '{div_point['final_token']}'\n")
            f.write(f"  Path A Correct: {comparison['path_a']['is_correct']}\n")
            f.write(f"  Path B Correct: {comparison['path_b']['is_correct']}\n")
            f.write(f"  Perturbation Start Layer: {analysis.get('perturbation_start_layer', 'N/A')}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Conservative Dynamic perturbation")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--data", type=str, required=True, help="GPQA dataset path")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum new tokens")
    parser.add_argument("--device_id", type=int, default=None, help="指定 GPU 编号（0, 1, 2, ...）；不指定则自动选卡。需在 GPU 上跑时请显式传入，例如 --device_id 0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--sampling_k", type=int, default=0, help="If >0, run k-sample path comparison on first N divergence points")
    parser.add_argument("--sampling_max_divergence_points", type=int, default=20, help="Max divergence points to run sampling on (when --sampling_k > 0)")
    parser.add_argument("--sampling_temperature", type=float, default=0.7, help="Temperature for sampling (when --sampling_k > 0)")
    
    args = parser.parse_args()
    
    run_evaluation(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        max_examples=args.max_examples,
        max_new_tokens=args.max_tokens,
        device_id=args.device_id,
        seed=args.seed,
        verbose=args.verbose,
        sampling_k=args.sampling_k,
        sampling_max_divergence_points=args.sampling_max_divergence_points,
        sampling_temperature=args.sampling_temperature,
    )


if __name__ == "__main__":
    main()
