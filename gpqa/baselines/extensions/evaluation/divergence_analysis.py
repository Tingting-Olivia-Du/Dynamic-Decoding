"""
Divergence point analysis for Conservative Dynamic strategy.
"""
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from typing import List, Dict
from llms.engine import Engine
from extensions.generate import generate_conservative_dynamic
from extensions.utils.gpqa_loader import Example, create_zero_shot_prompt


def collect_divergence_points(
    engine: Engine,
    dataset: List[Example],
    max_new_tokens: int = 512,
    verbose: bool = False
) -> List[Dict]:
    """
    收集回退层与末层预测不同的token位置（分歧点）。
    
    :param engine: LLM引擎
    :param dataset: GPQA数据集（Example列表）
    :param max_new_tokens: 最大生成token数
    :param verbose: 是否打印详细信息
    :return: 每个分歧点的信息列表
    """
    divergence_points = []
    
    for question_id, example in enumerate(dataset):
        if verbose and question_id % 10 == 0:
            print(f"Processing question {question_id}/{len(dataset)}...")
        
        prompt = create_zero_shot_prompt(example)
        
        # 使用Conservative Dynamic生成，获取详细信息
        try:
            response, info = generate_conservative_dynamic(
                engine=engine,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                output_dict=True,
                verbose=False
            )
        except Exception as e:
            if verbose:
                print(f"Error processing question {question_id}: {e}")
            continue
        
        # 获取层数和token信息
        selected_layers = info.get("selected_layers", [])
        all_layer_tokens = info.get("all_layer_tokens", [])
        layer_entropies = info.get("layer_entropies", [])
        generated_token_ids = info.get("generated_token_ids", [])
        generated_tokens = info.get("generated_tokens", [])
        
        if not selected_layers or not all_layer_tokens:
            continue
        
        # 确定总层数（从第一个token的信息中获取）
        num_layers = len(all_layer_tokens[0]) if all_layer_tokens else 0
        final_layer_idx = num_layers - 1
        
        # 构建已生成的token序列（用于后续路径对比）
        generated_token_sequence = generated_tokens if generated_tokens else []
        
        # 遍历每个token位置，找出分歧点
        for token_pos, selected_layer in enumerate(selected_layers):
            if token_pos >= len(all_layer_tokens):
                break
            
            # 获取该位置所有层的预测
            layer_tokens = all_layer_tokens[token_pos]
            
            if len(layer_tokens) < num_layers:
                continue
            
            # 获取回退层和末层的预测
            selected_token = layer_tokens[selected_layer]
            final_token = layer_tokens[final_layer_idx]
            
            # 如果回退层和末层预测不同，记录为分歧点
            if selected_token != final_token:
                # 获取从回退层到末层的中间层信息
                intermediate_layers = []
                entropies = []
                
                # 从回退层到末层（包括）
                for layer_idx in range(selected_layer, final_layer_idx + 1):
                    if layer_idx < len(layer_tokens) and token_pos < len(layer_entropies):
                        layer_entropy = layer_entropies[token_pos][layer_idx] if layer_idx < len(layer_entropies[token_pos]) else None
                        intermediate_layers.append({
                            'layer_idx': layer_idx,
                            'token': layer_tokens[layer_idx],
                            'entropy': layer_entropy
                        })
                        if layer_entropy is not None:
                            entropies.append(layer_entropy)
                
                # 获取分歧点之前的token序列
                context_before = generated_token_sequence[:token_pos] if token_pos < len(generated_token_sequence) else []
                
                divergence_point = {
                    'question_id': question_id,
                    'token_position': token_pos,
                    'selected_layer': selected_layer,
                    'final_layer': final_layer_idx,
                    'selected_token': selected_token,
                    'final_token': final_token,
                    'intermediate_layers': intermediate_layers,
                    'entropies': entropies,
                    'question': example.question,
                    'correct_answer_index': example.correct_index,
                    'context_before': context_before,  # 分歧点之前的token序列
                    'base_prompt': prompt  # 原始prompt
                }
                
                divergence_points.append(divergence_point)
                
                if verbose:
                    print(f"  Found divergence at token {token_pos}: "
                          f"Layer {selected_layer}='{selected_token}' vs "
                          f"Layer {final_layer_idx}='{final_token}'")
    
    if verbose:
        print(f"\nTotal divergence points found: {len(divergence_points)}")
        print(f"Divergence rate: {len(divergence_points) / sum(len(info.get('selected_layers', [])) for _ in range(len(dataset))) * 100:.2f}%")
    
    return divergence_points
