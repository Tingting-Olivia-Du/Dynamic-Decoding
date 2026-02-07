"""
Perturbation analysis and evaluation for Conservative Dynamic strategy.
"""
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from typing import Dict, List
import numpy as np
from collections import Counter


def analyze_perturbation_start(divergence_point: Dict) -> Dict:
    """
    分析扰动从第几层开始。
    
    :param divergence_point: 分歧点信息（来自collect_divergence_points）
    :return: 扰动分析结果
    """
    intermediate_layers = divergence_point.get('intermediate_layers', [])
    entropies = divergence_point.get('entropies', [])
    selected_layer = divergence_point['selected_layer']
    final_layer = divergence_point['final_layer']
    selected_token = divergence_point['selected_token']
    
    if not intermediate_layers:
        return {
            'perturbation_start_layer': None,
            'entropy_increase_layer': None,
            'prediction_change_layer': None,
            'entropy_trend': []
        }
    
    # 1. 找出预测开始变化的层
    prediction_change_layer = None
    for i, layer_info in enumerate(intermediate_layers):
        if layer_info['token'] != selected_token:
            prediction_change_layer = layer_info['layer_idx']
            break
    
    # 2. 找出熵值开始上升的层
    entropy_increase_layer = None
    if len(entropies) > 1:
        # 从回退层开始，熵值应该是最低的
        min_entropy = entropies[0] if entropies else None
        min_entropy_layer = selected_layer
        
        for i in range(1, len(entropies)):
            if entropies[i] > min_entropy * 1.05:  # 熵值上升超过5%
                entropy_increase_layer = intermediate_layers[i]['layer_idx']
                break
            if entropies[i] < min_entropy:
                min_entropy = entropies[i]
                min_entropy_layer = intermediate_layers[i]['layer_idx']
    
    # 3. 确定扰动起始层（取预测变化和熵值上升的较早者）
    perturbation_start_layer = None
    candidates = []
    if prediction_change_layer is not None:
        candidates.append(prediction_change_layer)
    if entropy_increase_layer is not None:
        candidates.append(entropy_increase_layer)
    
    if candidates:
        perturbation_start_layer = min(candidates)
    elif prediction_change_layer is not None:
        perturbation_start_layer = prediction_change_layer
    elif entropy_increase_layer is not None:
        perturbation_start_layer = entropy_increase_layer
    
    # 4. 计算熵值趋势
    entropy_trend = entropies.copy() if entropies else []
    
    return {
        'perturbation_start_layer': perturbation_start_layer,
        'entropy_increase_layer': entropy_increase_layer,
        'prediction_change_layer': prediction_change_layer,
        'entropy_trend': entropy_trend,
        'entropy_increase_magnitude': (entropies[-1] - entropies[0]) if len(entropies) > 1 else 0.0,
        'num_layers_affected': (final_layer - selected_layer) if perturbation_start_layer is None else (final_layer - perturbation_start_layer + 1)
    }


def evaluate_perturbation_impact(
    divergence_points: List[Dict],
    path_comparisons: List[Dict]
) -> Dict:
    """
    综合评估扰动的负面影响。
    
    :param divergence_points: 分歧点列表
    :param path_comparisons: 路径对比结果列表（与divergence_points一一对应）
    :return: 评估统计结果
    """
    if not divergence_points or not path_comparisons:
        return {
            'total_divergence_points': 0,
            'path_a_accuracy': 0.0,
            'path_b_accuracy': 0.0,
            'accuracy_improvement': 0.0,
            'perturbation_start_distribution': {},
            'layer_entropy_correlation': {}
        }
    
    # 1. 统计准确率
    path_a_correct = sum(1 for comp in path_comparisons if comp.get('path_a', {}).get('is_correct', False))
    path_b_correct = sum(1 for comp in path_comparisons if comp.get('path_b', {}).get('is_correct', False))
    total = len(path_comparisons)
    
    path_a_accuracy = path_a_correct / total if total > 0 else 0.0
    path_b_accuracy = path_b_correct / total if total > 0 else 0.0
    accuracy_improvement = path_a_accuracy - path_b_accuracy
    
    # 2. 分析扰动起始层分布
    perturbation_starts = []
    for div_point in divergence_points:
        analysis = analyze_perturbation_start(div_point)
        start_layer = analysis.get('perturbation_start_layer')
        if start_layer is not None:
            perturbation_starts.append(start_layer)
    
    perturbation_start_distribution = dict(Counter(perturbation_starts))
    
    # 3. 分析层与熵值的相关性
    layer_entropy_data = {}
    for div_point in divergence_points:
        intermediate_layers = div_point.get('intermediate_layers', [])
        for layer_info in intermediate_layers:
            layer_idx = layer_info['layer_idx']
            entropy = layer_info.get('entropy')
            if entropy is not None:
                if layer_idx not in layer_entropy_data:
                    layer_entropy_data[layer_idx] = []
                layer_entropy_data[layer_idx].append(entropy)
    
    # 计算每层的平均熵值
    layer_entropy_correlation = {
        layer_idx: {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'count': len(entropies)
        }
        for layer_idx, entropies in layer_entropy_data.items()
    }
    
    # 4. 统计扰动强度
    entropy_increases = []
    for div_point in divergence_points:
        analysis = analyze_perturbation_start(div_point)
        increase = analysis.get('entropy_increase_magnitude', 0.0)
        if increase > 0:
            entropy_increases.append(increase)
    
    avg_entropy_increase = np.mean(entropy_increases) if entropy_increases else 0.0
    
    # 5. 统计受影响层数
    affected_layers = []
    for div_point in divergence_points:
        analysis = analyze_perturbation_start(div_point)
        num_affected = analysis.get('num_layers_affected', 0)
        if num_affected > 0:
            affected_layers.append(num_affected)
    
    avg_affected_layers = np.mean(affected_layers) if affected_layers else 0.0
    
    return {
        'total_divergence_points': len(divergence_points),
        'path_a_accuracy': path_a_accuracy,
        'path_b_accuracy': path_b_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'perturbation_start_distribution': perturbation_start_distribution,
        'layer_entropy_correlation': layer_entropy_correlation,
        'avg_entropy_increase': avg_entropy_increase,
        'avg_affected_layers': avg_affected_layers,
        'path_a_correct_count': path_a_correct,
        'path_b_correct_count': path_b_correct,
        'total_comparisons': total
    }
