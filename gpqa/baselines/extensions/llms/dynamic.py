import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# 加载基础模型
model_id = "/workspace/ckpt/Qwen2.5-7B-Instruct"
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 获取 transformer 模块
if hasattr(base, "qwen2"):
    transformer = base.qwen2
elif hasattr(base, "model"):
    transformer = base.model
else:
    raise ValueError("无法识别 transformer 模块名称，请检查模型结构")

# 提取 decoder 最后 k 层
k = 3
all_layers = transformer.decoder.layers if hasattr(transformer, "decoder") else transformer.h
extra = [copy.deepcopy(layer) for layer in all_layers[-k:]]


class Qwen2ModifiedForCausalLM(PreTrainedModel):
    config_class = base.config.__class__

    def __init__(self, base_model, transformer, extra_modules):
        super().__init__(base_model.config)
        self.config = base_model.config
        self.base = base_model
        self.transformer = transformer
        self.lm_head = base_model.lm_head
        self.extra = nn.ModuleList(extra_modules)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs
        )
        hidden = out.last_hidden_state
        # 附加 k 层
        for layer in self.extra:
            hidden = layer(hidden)[0]
        logits = self.lm_head(hidden)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=out.past_key_values if use_cache else None,
        )


# 构造可 .generate() 模型
my_model = Qwen2ModifiedForCausalLM(base, transformer, extra).to(base.device).eval()

# 测试 generate
prompt = "请写一段关于自然语言处理的简短介绍："
inputs = tokenizer(prompt, return_tensors="pt").to(base.device)
with torch.no_grad():
    ids = my_model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
print(tokenizer.decode(ids[0], skip_special_tokens=True))
