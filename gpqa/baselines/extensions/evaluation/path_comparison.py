"""
Path comparison for divergence points analysis.
"""
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from typing import Dict, List
import torch
from llms.engine import Engine
from extensions.generate import generate_simple
from extensions.utils.gpqa_loader import create_zero_shot_prompt, extract_answer_from_text


def generate_from_prefix(
    engine: Engine,
    base_prompt: str,
    prefix_tokens: List[str],
    max_new_tokens: int = 512
) -> tuple[str, List[str]]:
    """
    从指定的token前缀继续生成。
    
    :param engine: LLM引擎
    :param base_prompt: 基础prompt（问题部分）
    :param prefix_tokens: 要添加的token前缀列表
    :param max_new_tokens: 最大生成token数
    :return: (生成的完整文本, token列表)
    """
    # 构建包含前缀的完整prompt
    prefix_text = "".join(prefix_tokens)
    full_prompt = base_prompt + prefix_text
    
    # 使用标准生成方法继续生成
    generated_text = generate_simple(
        engine=engine,
        prompt=full_prompt,
        max_new_tokens=max_new_tokens,
        output_dict=False
    )
    
    # 提取新生成的token（去掉前缀部分）
    full_text = generated_text
    if full_text.startswith(base_prompt + prefix_text):
        new_text = full_text[len(base_prompt + prefix_text):]
    else:
        new_text = full_text[len(base_prompt):]
    
    # 将文本分割为token（简单方法，实际应该用tokenizer）
    # 这里我们返回完整文本，让调用者处理token分割
    return generated_text, [new_text]


def compare_generation_paths(
    engine: Engine,
    divergence_point: Dict,
    base_prompt: str,
    max_new_tokens: int = 512,
    verbose: bool = False
) -> Dict:
    """
    从分歧点的两个token分别继续生成，对比结果。
    
    :param engine: LLM引擎
    :param divergence_point: 分歧点信息（来自collect_divergence_points）
    :param base_prompt: 基础prompt（问题部分，不包含已生成的token）
    :param max_new_tokens: 最大生成token数
    :param verbose: 是否打印详细信息
    :return: 对比结果字典
    """
    selected_layer = divergence_point['selected_layer']
    final_layer = divergence_point['final_layer']
    selected_token = divergence_point['selected_token']
    final_token = divergence_point['final_token']
    token_position = divergence_point['token_position']
    
    # 获取分歧点之前已生成的token（如果有的话）
    # 这里我们需要从generation_info中获取，但为了简化，我们假设base_prompt已经包含了之前的上下文
    # 实际使用时，需要传入完整的上下文
    
    # 路径A：使用回退层预测的token
    if verbose:
        print(f"Generating path A from layer {selected_layer} token '{selected_token}'...")
    
    try:
        path_a_text, path_a_tokens = generate_from_prefix(
            engine=engine,
            base_prompt=base_prompt,
            prefix_tokens=[selected_token],
            max_new_tokens=max_new_tokens
        )
        path_a_answer = extract_answer_from_text(path_a_text)
        path_a_correct = (path_a_answer is not None and 
                         ord(path_a_answer) - ord('A') == divergence_point['correct_answer_index'])
    except Exception as e:
        if verbose:
            print(f"Error generating path A: {e}")
        path_a_text = ""
        path_a_tokens = []
        path_a_answer = None
        path_a_correct = False
    
    # 路径B：使用末层预测的token
    if verbose:
        print(f"Generating path B from layer {final_layer} token '{final_token}'...")
    
    try:
        path_b_text, path_b_tokens = generate_from_prefix(
            engine=engine,
            base_prompt=base_prompt,
            prefix_tokens=[final_token],
            max_new_tokens=max_new_tokens
        )
        path_b_answer = extract_answer_from_text(path_b_text)
        path_b_correct = (path_b_answer is not None and 
                         ord(path_b_answer) - ord('A') == divergence_point['correct_answer_index'])
    except Exception as e:
        if verbose:
            print(f"Error generating path B: {e}")
        path_b_text = ""
        path_b_tokens = []
        path_b_answer = None
        path_b_correct = False
    
    # 计算token差异（简单方法：比较文本长度和内容）
    # 更精确的方法需要token级别的比较
    token_differences = []
    min_len = min(len(path_a_tokens), len(path_b_tokens))
    for i in range(min_len):
        if path_a_tokens[i] != path_b_tokens[i]:
            token_differences.append(i)
    
    # 如果长度不同，记录差异
    if len(path_a_tokens) != len(path_b_tokens):
        token_differences.append(min_len)
    
    result = {
        'path_a': {
            'tokens': path_a_tokens,
            'text': path_a_text,
            'answer': path_a_answer,
            'is_correct': path_a_correct
        },
        'path_b': {
            'tokens': path_b_tokens,
            'text': path_b_text,
            'answer': path_b_answer,
            'is_correct': path_b_correct
        },
        'token_differences': token_differences,
        'accuracy_diff': 1.0 if path_a_correct and not path_b_correct else (-1.0 if not path_a_correct and path_b_correct else 0.0)
    }
    
    return result


def compare_paths_for_divergence_point(
    engine: Engine,
    divergence_point: Dict,
    full_context_before: str,
    max_new_tokens: int = 512,
    verbose: bool = False
) -> Dict:
    """
    为分歧点比较两个生成路径（考虑完整上下文）。
    
    :param engine: LLM引擎
    :param divergence_point: 分歧点信息
    :param full_context_before: 分歧点之前的完整上下文（包括问题和已生成的token）
    :param max_new_tokens: 最大生成token数
    :param verbose: 是否打印详细信息
    :return: 对比结果
    """
    selected_token = divergence_point['selected_token']
    final_token = divergence_point['final_token']
    
    # 路径A：使用回退层token
    path_a_prompt = full_context_before + selected_token
    try:
        path_a_text = generate_simple(
            engine=engine,
            prompt=path_a_prompt,
            max_new_tokens=max_new_tokens,
            output_dict=False
        )
        path_a_answer = extract_answer_from_text(path_a_text)
        path_a_correct = (path_a_answer is not None and 
                         ord(path_a_answer) - ord('A') == divergence_point['correct_answer_index'])
    except Exception as e:
        if verbose:
            print(f"Error in path A: {e}")
        path_a_text = ""
        path_a_answer = None
        path_a_correct = False
    
    # 路径B：使用末层token
    path_b_prompt = full_context_before + final_token
    try:
        path_b_text = generate_simple(
            engine=engine,
            prompt=path_b_prompt,
            max_new_tokens=max_new_tokens,
            output_dict=False
        )
        path_b_answer = extract_answer_from_text(path_b_text)
        path_b_correct = (path_b_answer is not None and 
                         ord(path_b_answer) - ord('A') == divergence_point['correct_answer_index'])
    except Exception as e:
        if verbose:
            print(f"Error in path B: {e}")
        path_b_text = ""
        path_b_answer = None
        path_b_correct = False
    
    return {
        'path_a': {
            'text': path_a_text,
            'answer': path_a_answer,
            'is_correct': path_a_correct
        },
        'path_b': {
            'text': path_b_text,
            'answer': path_b_answer,
            'is_correct': path_b_correct
        },
        'accuracy_diff': 1.0 if path_a_correct and not path_b_correct else (-1.0 if not path_a_correct and path_b_correct else 0.0)
    }
