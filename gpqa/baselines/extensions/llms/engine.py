import torch
from dataclasses import asdict
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_available_gpus():
    """
    获取可用的GPU信息。
    
    :return: GPU信息字典列表
    """
    if not torch.cuda.is_available():
        return []
    
    gpus = []
    for i in range(torch.cuda.device_count()):
        gpu_info = {
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "memory_total": torch.cuda.get_device_properties(i).total_memory / 1024**3,  # GB
            "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
            "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3,  # GB
        }
        gpus.append(gpu_info)
    
    return gpus


def print_gpu_info():
    """
    打印可用的GPU信息。
    """
    gpus = get_available_gpus()
    if not gpus:
        print("没有可用的GPU")
        return
    
    print(f"可用GPU数量: {len(gpus)}")
    for gpu in gpus:
        print(f"  GPU {gpu['id']}: {gpu['name']}")
        print(f"    总内存: {gpu['memory_total']:.2f} GB")
        print(f"    已分配: {gpu['memory_allocated']:.2f} GB")
        print(f"    已保留: {gpu['memory_reserved']:.2f} GB")
        print(f"    可用: {gpu['memory_total'] - gpu['memory_reserved']:.2f} GB")


class Engine:
    def __init__(self, model_name: str = "gpt2", device: str = None, device_id: int = None, 
                 dtype=torch.bfloat16, device_map: str = None):
        """
        初始化LLM引擎。

        :param model_name: 模型名称或路径
        :param device: 使用的设备（'cuda', 'cpu', 'cuda:0', 'cuda:1' 等）。如果为None，自动选择
        :param device_id: GPU设备ID（0, 1, 2等）。如果指定，会覆盖device参数中的GPU ID
        :param dtype: 使用的 torch 数据类型
        :param device_map: 设备映射（用于多GPU，如'auto', 'balanced', 'balanced_low_0'等）
                          如果指定，会使用device_map而不是device参数
        """
        # 处理device_map（用于多GPU）
        if device_map is not None:
            self.device = "cuda"  # device_map模式下，device主要用于输入tensor
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left", trust_remote_code=True)
            self.dtype = dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=device_map
            )
            return
        
        # 处理device参数
        if device is None:
            # 自动选择设备
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # 如果指定了device_id，覆盖device中的GPU ID
        if device_id is not None:
            if not torch.cuda.is_available():
                raise ValueError(f"指定了device_id={device_id}，但CUDA不可用")
            if device_id < 0 or device_id >= torch.cuda.device_count():
                raise ValueError(f"device_id={device_id}超出范围，可用GPU数量: {torch.cuda.device_count()}")
            self.device = f"cuda:{device_id}"
        
        # 验证设备
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError("指定了CUDA设备，但CUDA不可用")
            # 提取GPU ID
            if ":" in self.device:
                gpu_id = int(self.device.split(":")[1])
                if gpu_id >= torch.cuda.device_count():
                    raise ValueError(f"GPU {gpu_id}不存在，可用GPU数量: {torch.cuda.device_count()}")
            # 设置当前GPU（如果指定了具体ID）
            if ":" in self.device:
                torch.cuda.set_device(int(self.device.split(":")[1]))
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left", trust_remote_code=True)
        self.dtype = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device, dtype=self.dtype)

    def generate(self, prompt: str, max_length: int = 1024, max_new_tokens: int = 50, skip_special_tokens: bool = True, extra_output_keys: list[str] = None, **kwargs) -> dict:
        """
        进行生成。

        :param extra_output_keys: 额外输出项
        :param max_length: 最大输入长度
        :param skip_special_tokens: 解码时是否跳过特殊token
        :param max_new_tokens: 最大生成token数目
        :param prompt: 输入提示文本
        :return: 生成的文本及其他信息
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        output = self.model.generate(**inputs, return_dict_in_generate=True, output_scores=True,
                                     max_new_tokens=max_new_tokens, **kwargs)

        sequence = output.sequences[0]
        scores = output.scores

        # 提取新生成的 token 和 ID
        generated_ids = sequence.tolist()[len(inputs["input_ids"][0]):]
        generated_tokens = [self.tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens) for token_id in
                            generated_ids]

        # 将输出转换为字典
        output_dict = asdict(output)

        generation_output = {
            "text": self.tokenizer.decode(sequence, skip_special_tokens=skip_special_tokens),
            "sequence": sequence,
            "scores": scores,
            "generated_tokens": generated_tokens,
            "generated_token_ids": generated_ids,
        }

        if extra_output_keys is not None:
            for key in extra_output_keys:
                if key in output_dict:
                    generation_output[key] = output_dict[key]

        return generation_output

    def get_logits(self, prompt: str, max_length: int = 1024) -> torch.Tensor:
        """
        获取模型的logits输出。

        :param max_length: 最大输入长度
        :param prompt: 输入提示文本
        :return: 模型输出的logits
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def get_per_layer_logits(self, prompt: str, max_length: int = 1024) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        获取每一层的logits输出（用于Conservative Dynamic策略）。
        
        通过手动遍历每一层，计算每层的hidden states，然后通过lm_head得到logits。

        :param max_length: 最大输入长度
        :param prompt: 输入提示文本
        :return: (各层logits列表, 最终logits)，各层logits的shape为 [batch_size, seq_len, vocab_size]
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        
        # 获取transformer模块
        if hasattr(self.model, "model"):
            transformer = self.model.model
        elif hasattr(self.model, "transformer"):
            transformer = self.model.transformer
        elif hasattr(self.model, "gpt_neox"):
            transformer = self.model.gpt_neox
        elif hasattr(self.model, "qwen2"):
            transformer = self.model.qwen2
        else:
            raise ValueError("无法识别transformer模块名称，请检查模型结构")
        
        # 获取层列表和embedding层
        if hasattr(transformer, "h"):
            layers = transformer.h
            if hasattr(transformer, "wte"):  # GPT-style
                embeddings = transformer.wte
            elif hasattr(transformer, "embed_tokens"):  # LLaMA/Qwen-style
                embeddings = transformer.embed_tokens
            else:
                raise ValueError("无法识别embedding模块")
        elif hasattr(transformer, "layers"):
            layers = transformer.layers
            if hasattr(transformer, "embed_tokens"):
                embeddings = transformer.embed_tokens
            else:
                raise ValueError("无法识别embedding模块")
        elif hasattr(transformer, "decoder") and hasattr(transformer.decoder, "layers"):
            layers = transformer.decoder.layers
            if hasattr(transformer.decoder, "embed_tokens"):
                embeddings = transformer.decoder.embed_tokens
            else:
                raise ValueError("无法识别embedding模块")
        else:
            raise ValueError("无法识别layers模块名称")
        
        # 存储各层的logits
        per_layer_logits = []
        
        with torch.no_grad():
            # 获取input embeddings
            input_ids = inputs["input_ids"]
            hidden_states = embeddings(input_ids)
            
            # 如果有position embeddings，添加它们
            if hasattr(transformer, "wpe"):  # GPT-style positional embeddings
                position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0)
                hidden_states = hidden_states + transformer.wpe(position_ids)
            elif hasattr(transformer, "embed_positions"):  # 某些模型
                position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0)
                hidden_states = hidden_states + transformer.embed_positions(position_ids)
            
            # 遍历每一层，计算每层的hidden states和logits
            for layer in layers:
                # 通过当前层
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=inputs.get("attention_mask", None),
                    use_cache=False,
                )
                
                # 获取输出（可能是tuple）
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
                
                # 通过lm_head获取当前层的logits
                logits = self.model.lm_head(hidden_states)
                per_layer_logits.append(logits)
            
            # 获取最终logits（与最后一层相同，但为了兼容性保留）
            final_logits = per_layer_logits[-1] if per_layer_logits else None
        
        return per_layer_logits, final_logits

