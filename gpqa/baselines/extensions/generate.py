import os
import sys
import torch
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

from llms.engine import Engine
from extensions.utils.entropy import calculate_information_entropy
from extensions.utils.format import make_raw_chat_prompt
from extensions.utils.vllm_utils import sample, LLM


def generate(engine: Engine, prompt: str, max_length: int = 1024, max_new_tokens: int = 512, use_dynamic_layers: bool = False, **kwargs) -> str:
    """
    生成回复（不含特殊 tokens）。
    """
    eos_token_id = engine.tokenizer.eos_token_id
    chat_prompt = make_raw_chat_prompt(prompt, "", engine.tokenizer)

    if not use_dynamic_layers:
        result = engine.generate(
            chat_prompt,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=False,
            pad_token_id=eos_token_id,
            **kwargs,
        )
        generated_token_ids = result["generated_token_ids"]

    else:
        generated_token_ids = []
        entropies = []

        current_prompt = chat_prompt

        # 动态层数
        n_dynamic_layers = 0

        # 熵差阈值
        entropy_threshold = 2.0

        while len(generated_token_ids) < max_new_tokens:
            # 调用 generate 方法，只生成一个 token
            result = engine.generate(
                current_prompt,
                max_length=max_length,
                max_new_tokens=1,
                skip_special_tokens=False,
                pad_token_id=eos_token_id,
                n_dynamic_layers=n_dynamic_layers,
                **kwargs,
            )

            # 解码生成的 token
            next_token = result["generated_tokens"][0]
            next_token_id = result["generated_token_ids"][0]

            # 获取 logits 并计算信息熵
            logits = result["scores"][0]
            entropy_value = calculate_information_entropy(logits).item()
            entropies.append(entropy_value)
            if len(entropies) > 1 and entropies[-2] - entropies[-1] > entropy_threshold:
                print(
                    f"Information entropy dropped by {entropies[-2] - entropies[-1]:.4f} bits, turning on shallow thinking mode.\n")
                n_dynamic_layers = -1
            else:
                n_dynamic_layers = 0

            # 添加到结果中
            generated_token_ids.append(next_token_id)
            current_prompt += next_token

            # 如果遇到 EOS，则终止生成
            if next_token_id == eos_token_id:
                break

    generated_text = engine.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return generated_text


def generate_simple(engine: Engine, prompt: str, max_length: int = 1024, max_new_tokens: int = 512, use_dynamic_layers: bool = False, entropy_criterion=None, output_dict: bool = False, **kwargs) -> str | tuple[str, dict]:
    """
    生成回复（不含特殊 tokens），基于魔改的transformers库。
    """
    eos_token_id = engine.tokenizer.eos_token_id
    chat_prompt = make_raw_chat_prompt(prompt, "", engine.tokenizer)

    result = engine.generate(
        chat_prompt,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        skip_special_tokens=False,
        pad_token_id=eos_token_id,
        use_dynamic_layers=use_dynamic_layers,
        entropy_criterion=entropy_criterion,
        **kwargs,
    )

    generated_token_ids = result["generated_token_ids"]
    generated_text = engine.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    if output_dict:
        return generated_text, result
    else:
        return generated_text


def generate_vllm(llm: LLM, tokenizer, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
    """
    生成回复（不含特殊 tokens），基于 vLLM 库。
    """
    chat_prompt = make_raw_chat_prompt(prompt, "", tokenizer)
    result = sample(llm, chat_prompt, n=1, max_tokens=max_new_tokens, **kwargs)
    generated_text = result[0][0]
    return generated_text


def generate_conservative_dynamic(engine: Engine, prompt: str, max_length: int = 1024, max_new_tokens: int = 512, 
                                   output_dict: bool = False, verbose: bool = False, **kwargs) -> str | tuple[str, dict]:
    """
    使用Conservative Dynamic策略生成回复。
    
    策略：从末层向前查看，只要发现更早层的熵更低，就用该层；一旦不再降低，立即停止查看。
    最终选择熵最低的层，输出该层的token预测。
    
    :param engine: LLM引擎
    :param prompt: 输入提示
    :param max_length: 最大输入长度
    :param max_new_tokens: 最大生成token数
    :param output_dict: 是否返回详细信息
    :param verbose: 是否打印调试信息
    :param kwargs: 其他参数
    :return: 生成的文本，或(文本, 详细信息)元组
    """
    eos_token_id = engine.tokenizer.eos_token_id
    chat_prompt = make_raw_chat_prompt(prompt, "", engine.tokenizer)
    
    generated_token_ids = []
    current_prompt = chat_prompt
    
    # 存储每个token的详细信息
    generation_info = {
        "selected_layers": [],
        "layer_entropies": [],
        "all_layer_tokens": []
    }
    
    while len(generated_token_ids) < max_new_tokens:
        # 获取各层的logits
        try:
            per_layer_logits, final_logits = engine.get_per_layer_logits(current_prompt, max_length=max_length)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to get per-layer logits: {e}. Falling back to final layer.")
            # 如果获取各层logits失败，回退到使用最终层
            inputs = engine.tokenizer(current_prompt, return_tensors="pt", truncation=True, max_length=max_length).to(engine.device)
            with torch.no_grad():
                outputs = engine.model(**inputs)
                final_logits = outputs.logits
            per_layer_logits = [final_logits]
        
        # 获取最后一个位置的logits（用于预测下一个token）
        # logits shape: [batch_size, seq_len, vocab_size]
        # 我们需要最后一个位置的logits
        num_layers = len(per_layer_logits)
        layer_entropies = []
        layer_token_ids = []
        layer_tokens = []
        
        # 计算每一层的熵和预测的token
        for layer_idx, logits in enumerate(per_layer_logits):
            # 获取最后一个位置的logits [batch_size, vocab_size]
            last_logits = logits[0, -1, :]  # [vocab_size]
            
            # 计算熵
            entropy = calculate_information_entropy(last_logits.unsqueeze(0)).item()  # [1] -> scalar
            layer_entropies.append(entropy)
            
            # 获取该层预测的token
            predicted_token_id = last_logits.argmax().item()
            predicted_token = engine.tokenizer.decode([predicted_token_id], skip_special_tokens=False)
            layer_token_ids.append(predicted_token_id)
            layer_tokens.append(predicted_token)
        
        # Conservative Dynamic策略：从末层向前查找熵最低的层
        # 自 L-1 到 0 依次检查，维护当前最佳 H_best 与层索引 j
        # 若 H_i < H_best 则更新 H_best <- H_i, j <- i
        # 否则对该样本早停
        selected_layer_idx = num_layers - 1  # 默认选择最后一层
        best_entropy = layer_entropies[selected_layer_idx]  # 初始化为最后一层的熵
        
        for layer_idx in range(num_layers - 2, -1, -1):  # 从倒数第二层向前
            if layer_entropies[layer_idx] < best_entropy:
                best_entropy = layer_entropies[layer_idx]
                selected_layer_idx = layer_idx
            else:
                # 一旦熵不再降低，立即停止查看（早停）
                break
        
        # 使用选中层的预测
        selected_token_id = layer_token_ids[selected_layer_idx]
        selected_token = layer_tokens[selected_layer_idx]
        
        if verbose:
            print(f"Token {len(generated_token_ids)}: Selected layer {selected_layer_idx}/{num_layers-1}, "
                  f"entropy={best_entropy:.4f}, token='{selected_token}'")
        
        # 记录信息
        generation_info["selected_layers"].append(selected_layer_idx)
        generation_info["layer_entropies"].append(layer_entropies.copy())
        generation_info["all_layer_tokens"].append(layer_tokens.copy())
        
        # 添加到结果中
        generated_token_ids.append(selected_token_id)
        current_prompt += selected_token
        
        # 如果遇到 EOS，则终止生成
        if selected_token_id == eos_token_id:
            break
    
    generated_text = engine.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    
    # 生成token文本列表
    generated_tokens = [engine.tokenizer.decode([token_id], skip_special_tokens=False) for token_id in generated_token_ids]
    
    if output_dict:
        generation_info["generated_text"] = generated_text
        generation_info["generated_token_ids"] = generated_token_ids
        generation_info["generated_tokens"] = generated_tokens
        return generated_text, generation_info
    else:
        return generated_text
