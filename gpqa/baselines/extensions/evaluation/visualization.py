"""
Visualization functions for perturbation analysis.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import Counter

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)


def plot_entropy_trends(
    divergence_points: List[Dict],
    output_file: Path,
    max_samples: int = 20
):
    """
    绘制从回退层到末层的熵值变化曲线。
    
    :param divergence_points: 分歧点列表
    :param output_file: 输出文件路径
    :param max_samples: 最多绘制的样本数
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Entropy Trends from Selected Layer to Final Layer', fontsize=16)
    
    # 选择前max_samples个样本
    samples = divergence_points[:max_samples]
    
    # 子图1: 所有样本的熵值曲线
    ax1 = axes[0, 0]
    for i, div_point in enumerate(samples):
        entropies = div_point.get('entropies', [])
        if entropies:
            layer_indices = [layer['layer_idx'] for layer in div_point.get('intermediate_layers', [])]
            if len(layer_indices) == len(entropies):
                ax1.plot(layer_indices, entropies, alpha=0.3, linewidth=1)
    
    # 计算平均熵值曲线
    all_layer_indices = set()
    for div_point in samples:
        for layer in div_point.get('intermediate_layers', []):
            all_layer_indices.add(layer['layer_idx'])
    
    if all_layer_indices:
        sorted_layers = sorted(all_layer_indices)
        avg_entropies = []
        for layer_idx in sorted_layers:
            entropies_at_layer = []
            for div_point in samples:
                for layer_info in div_point.get('intermediate_layers', []):
                    if layer_info['layer_idx'] == layer_idx and layer_info.get('entropy') is not None:
                        entropies_at_layer.append(layer_info['entropy'])
            if entropies_at_layer:
                avg_entropies.append(np.mean(entropies_at_layer))
            else:
                avg_entropies.append(None)
        
        valid_indices = [i for i, e in enumerate(avg_entropies) if e is not None]
        if valid_indices:
            ax1.plot([sorted_layers[i] for i in valid_indices], 
                    [avg_entropies[i] for i in valid_indices], 
                    'r-', linewidth=2, label='Average', marker='o')
    
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Entropy Trends (All Samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 平均熵值变化
    ax2 = axes[0, 1]
    if all_layer_indices:
        sorted_layers = sorted(all_layer_indices)
        mean_entropies = []
        std_entropies = []
        for layer_idx in sorted_layers:
            entropies_at_layer = []
            for div_point in samples:
                for layer_info in div_point.get('intermediate_layers', []):
                    if layer_info['layer_idx'] == layer_idx and layer_info.get('entropy') is not None:
                        entropies_at_layer.append(layer_info['entropy'])
            if entropies_at_layer:
                mean_entropies.append(np.mean(entropies_at_layer))
                std_entropies.append(np.std(entropies_at_layer))
            else:
                mean_entropies.append(None)
                std_entropies.append(None)
        
        valid_indices = [i for i, e in enumerate(mean_entropies) if e is not None]
        if valid_indices:
            layers_plot = [sorted_layers[i] for i in valid_indices]
            means_plot = [mean_entropies[i] for i in valid_indices]
            stds_plot = [std_entropies[i] for i in valid_indices]
            
            ax2.plot(layers_plot, means_plot, 'b-', linewidth=2, marker='o', label='Mean')
            ax2.fill_between(layers_plot, 
                           [m - s for m, s in zip(means_plot, stds_plot)],
                           [m + s for m, s in zip(means_plot, stds_plot)],
                           alpha=0.3, label='±1 Std')
    
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Average Entropy with Standard Deviation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 扰动起始层分布
    ax3 = axes[1, 0]
    from extensions.evaluation.perturbation_evaluation import analyze_perturbation_start
    perturbation_starts = []
    for div_point in divergence_points:
        analysis = analyze_perturbation_start(div_point)
        start_layer = analysis.get('perturbation_start_layer')
        if start_layer is not None:
            perturbation_starts.append(start_layer)
    
    if perturbation_starts:
        counter = Counter(perturbation_starts)
        layers = sorted(counter.keys())
        counts = [counter[l] for l in layers]
        ax3.bar(layers, counts, alpha=0.7)
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Perturbation Start Layer Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4: 熵值增加幅度分布
    ax4 = axes[1, 1]
    entropy_increases = []
    for div_point in divergence_points:
        analysis = analyze_perturbation_start(div_point)
        increase = analysis.get('entropy_increase_magnitude', 0.0)
        if increase > 0:
            entropy_increases.append(increase)
    
    if entropy_increases:
        ax4.hist(entropy_increases, bins=20, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Entropy Increase Magnitude')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Entropy Increases')
        ax4.axvline(np.mean(entropy_increases), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(entropy_increases):.4f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_comparison(
    evaluation_results: Dict,
    output_file: Path
):
    """
    绘制准确率对比图。
    
    :param evaluation_results: 评估结果
    :param output_file: 输出文件路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Accuracy Comparison: Selected Layer vs Final Layer', fontsize=16)
    
    # 子图1: 准确率对比柱状图
    ax1 = axes[0]
    categories = ['Path A\n(Selected Layer)', 'Path B\n(Final Layer)']
    accuracies = [
        evaluation_results['path_a_accuracy'],
        evaluation_results['path_b_accuracy']
    ]
    colors = ['green' if acc > evaluation_results['path_b_accuracy'] else 'red' 
              for acc in accuracies]
    
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    ax1.set_title('Path Accuracy Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom')
    
    # 子图2: 准确率改进
    ax2 = axes[1]
    improvement = evaluation_results['accuracy_improvement']
    color = 'green' if improvement > 0 else 'red'
    ax2.barh(['Accuracy\nImprovement'], [improvement], color=color, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Accuracy Difference')
    ax2.set_xlim([-0.5, 0.5])
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Selected Layer Advantage')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    ax2.text(improvement, 0, f'{improvement:.2%}',
             ha='left' if improvement > 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_perturbation_start_histogram(
    evaluation_results: Dict,
    output_file: Path,
) -> None:
    """
    绘制扰动起始层直方图（全量 + 负面子集对比）。

    :param evaluation_results: 含 perturbation_start_distribution 与 negative_subset_perturbation_start_distribution
    :param output_file: 输出文件路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Perturbation Start Layer Distribution', fontsize=14)

    # 全量分布
    ax1 = axes[0]
    dist = evaluation_results.get('perturbation_start_distribution', {})
    if dist:
        layers = sorted(dist.keys())
        counts = [dist[l] for l in layers]
        ax1.bar(layers, counts, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Frequency')
    ax1.set_title('All Divergence Points')
    ax1.grid(True, alpha=0.3, axis='y')

    # 负面子集（Path A 对且 Path B 错）
    ax2 = axes[1]
    neg_dist = evaluation_results.get('negative_subset_perturbation_start_distribution', {})
    if neg_dist:
        layers = sorted(neg_dist.keys())
        counts = [neg_dist[l] for l in layers]
        ax2.bar(layers, counts, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Negative Subset (Path A correct, Path B wrong)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_all_visualizations(
    divergence_points: List[Dict],
    evaluation_results: Dict,
    output_dir: Path
):
    """
    创建所有可视化图表。

    :param divergence_points: 分歧点列表
    :param evaluation_results: 评估结果
    :param output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制熵值趋势
    entropy_file = output_dir / "entropy_trends.png"
    plot_entropy_trends(divergence_points, entropy_file)
    print(f"Saved entropy trends to {entropy_file}")

    # 绘制准确率对比
    accuracy_file = output_dir / "accuracy_comparison.png"
    plot_accuracy_comparison(evaluation_results, accuracy_file)
    print(f"Saved accuracy comparison to {accuracy_file}")

    # 扰动起始层直方图（全量 + 负面子集）
    perturbation_hist_file = output_dir / "perturbation_start_histogram.png"
    plot_perturbation_start_histogram(evaluation_results, perturbation_hist_file)
    print(f"Saved perturbation start histogram to {perturbation_hist_file}")
