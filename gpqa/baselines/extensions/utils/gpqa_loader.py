"""
GPQA dataset loader for Conservative Dynamic evaluation.
"""
import pandas as pd
import random
from typing import List
from collections import namedtuple

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])


def load_gpqa_diamond(data_path: str, seed: int = 42, shuffle_choices: bool = True) -> List[Example]:
    """
    加载GPQA-diamond数据集。
    
    :param data_path: CSV文件路径
    :param seed: 随机种子（用于打乱选项顺序）
    :param shuffle_choices: 是否打乱选项顺序
    :return: Example列表
    """
    question_df = pd.read_csv(data_path)
    random.seed(seed)
    
    def create_example(row) -> Example:
        if shuffle_choices:
            list_choices = [
                row['Incorrect Answer 1'],
                row['Incorrect Answer 2'],
                row['Incorrect Answer 3'],
                row['Correct Answer']
            ]
            random.shuffle(list_choices)
            correct_index = list_choices.index(row['Correct Answer'])
        else:
            list_choices = [
                row['Incorrect Answer 1'],
                row['Incorrect Answer 2'],
                row['Incorrect Answer 3'],
                row['Correct Answer']
            ]
            correct_index = 3  # Correct Answer is always last if not shuffled
        
        example = Example(
            row['Question'],
            list_choices[0],
            list_choices[1],
            list_choices[2],
            list_choices[3],
            correct_index
        )
        return example
    
    return [create_example(row) for _, row in question_df.iterrows()]


def create_zero_shot_prompt(example: Example) -> str:
    """
    创建zero-shot prompt。
    
    :param example: Example对象
    :return: prompt字符串
    """
    prompt = f"{example.question}\n\n"
    prompt += f"(A) {example.choice1}\n"
    prompt += f"(B) {example.choice2}\n"
    prompt += f"(C) {example.choice3}\n"
    prompt += f"(D) {example.choice4}\n\n"
    prompt += "Answer:"
    return prompt


def extract_answer_from_text(text: str) -> str:
    """
    从生成的文本中提取答案（A/B/C/D）。
    
    :param text: 生成的文本
    :return: 提取的答案（'A', 'B', 'C', 'D'或None）
    """
    import re
    # 查找第一个出现的(A), (B), (C), (D)
    patterns = [
        r'\(A\)',
        r'\(B\)',
        r'\(C\)',
        r'\(D\)',
        r'\bA\b',
        r'\bB\b',
        r'\bC\b',
        r'\bD\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group().upper().replace('(', '').replace(')', '').strip()
            if letter in ['A', 'B', 'C', 'D']:
                return letter
    
    return None
