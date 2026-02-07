"""
基于熵的层级选择工具模块（项目特有）

用于 Layer Contrast 研究中的动态多层解码：根据各层 hidden state 的 logits 信息熵，
选择「熵谷」或「熵+一致性」的最优层进行解码，支持早退、内层解码等策略。

主要函数：
- calculate_information_entropy: 计算 logits 分布的信息熵
- calculate_multi_layer_entropy_selection: 多层熵计算与熵谷选层
- _select_layers_from_entropies: 基于熵（及可选一致性因子）的层选择
"""
import torch
import torch.nn.functional as F
from typing import Optional, Callable


def calculate_cosine_similarity(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """
    计算两个hidden states之间的cosine similarity。
    
    Args:
        h1: 第一个hidden states，形状为 [batch_size, hidden_size]
        h2: 第二个hidden states，形状为 [batch_size, hidden_size]
    
    Returns:
        similarity: 形状为 [batch_size] 的张量，包含每个样本的cosine similarity
    """
    # 归一化到单位向量
    h1_norm = F.normalize(h1, p=2, dim=-1)
    h2_norm = F.normalize(h2, p=2, dim=-1)
    
    # 计算cosine similarity
    similarity = torch.sum(h1_norm * h2_norm, dim=-1)
    
    return similarity


def calculate_information_entropy(logits: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    # size of logits: [batch_size, vocab_size]

    # Step 0: 将 logits 数据类型转换为 float
    logits = logits.float()

    # Step 1: 计算 softmax 概率分布（注意使用 dim=-1 来保证维度正确）
    probs = F.softmax(logits, dim=-1)

    # Step 2: 计算信息熵（以 2 为底，单位是 bits）
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1)  # 加一个小量防止 log(0)

    # Step 3: 归一化（如果需要）
    if normalize:
        vocab_size = logits.shape[-1]
        max_entropy = torch.log2(torch.tensor(vocab_size, dtype=logits.dtype, device=logits.device))
        entropy = entropy / max_entropy

    return entropy  # shape: [batch_size]


def calculate_topk_information_entropy(logits: torch.Tensor, k: int, normalize: bool = True) -> torch.Tensor:
    # size of logits: [batch_size, vocab_size]

    # Step 0: 将 logits 数据类型转换为 float
    logits = logits.float()

    # Step 1: 获取 top-k 的 logit 及其对应的概率
    topk_logits, indices = torch.topk(logits, k=k, dim=-1)

    # Step 2: 对 top-k 的 logit 进行 softmax 归一化
    topk_probs = F.softmax(topk_logits, dim=-1)

    # Step 3: 在 top-k 概率上计算信息熵
    entropy = -torch.sum(topk_probs * torch.log2(topk_probs + 1e-12), dim=-1)

    # Step 4: 归一化（如果需要）
    if normalize:
        max_entropy = torch.log2(torch.tensor(k, dtype=logits.dtype, device=logits.device))
        entropy = entropy / max_entropy

    return entropy  # shape: [batch_size]


def _select_layers_from_entropies(entropies_per_layer: list[torch.Tensor], 
                                  layer_hidden_states: list[torch.Tensor] = None,
                                  consistency_factor: Optional[float] = None,
                                  *,
                                  use_embedding_cossim: bool = False,
                                  sampled_token_ids_per_layer: Optional[list[torch.Tensor]] = None,
                                  embedding_module: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                                  layer_decay_factor: Optional[float] = None) -> torch.Tensor:
    """
    基于每层的熵向量和可选的consistency factor，选择最优层。
    
    当consistency_factor不是None时，选择依据为：
    argmax_i(-Entropy(h_i) * (1 - consistency_factor) + consistency_factor * cos_similarity(h_i, h_L))
    
    其中：
    - 当consistency_factor接近1时，主要考虑相似性，熵值权重较小
    - 当consistency_factor接近0时，主要考虑熵值，相似性权重较小
    - 当consistency_factor = 0.5时，熵值和相似性权重相等
    
    当consistency_factor为None时，使用原来的熵值选择方法。

    Args:
        entropies_per_layer: 按层排列的熵向量列表，每个元素形状为 [B]，长度为 L（层数）。
        layer_hidden_states: 各层的hidden states列表，仅在consistency_factor不是None时需要。
        consistency_factor: 一致性因子，控制cosine similarity的权重。

    Returns:
        selected_idx: 形状 [B] 的长整型张量，给出每个样本最终选择的层索引（0..L-1）。
    """
    assert len(entropies_per_layer) >= 1, "entropies_per_layer must have at least one element"
    
    # 如果consistency_factor不是None，使用一致性增强选择方法
    if consistency_factor is not None:
        return _select_layers_with_consistency(
            entropies_per_layer,
            layer_hidden_states,
            consistency_factor,
            use_embedding_cossim=use_embedding_cossim,
            sampled_token_ids_per_layer=sampled_token_ids_per_layer,
            embedding_module=embedding_module,
            layer_decay_factor=layer_decay_factor,
        )
    
    # 否则使用原来的熵值选择方法
    return _select_layers_from_entropies_original(entropies_per_layer)


def _select_layers_from_entropies_original(entropies_per_layer: list[torch.Tensor]) -> torch.Tensor:
    """
    基于每层的熵向量，按样本逐个从末层向前选择熵更小的层；一旦某样本出现不再下降即停止继续向前看。

    Args:
        entropies_per_layer: 按层排列的熵向量列表，每个元素形状为 [B]，长度为 L（层数）。

    Returns:
        selected_idx: 形状 [B] 的长整型张量，给出每个样本最终选择的层索引（0..L-1）。
    """
    assert len(entropies_per_layer) >= 1, "entropies_per_layer must have at least one element"

    last_entropy = entropies_per_layer[-1]
    B = last_entropy.shape[0]
    device = last_entropy.device

    # 初始选择末层
    selected_idx = torch.full((B,), len(entropies_per_layer) - 1,
                              dtype=torch.long, device=device)
    selected_entropy = last_entropy  # [B]
    active = torch.ones(B, dtype=torch.bool, device=device)

    # 从倒数第二层往前
    for i in range(len(entropies_per_layer) - 2, -1, -1):
        curr = entropies_per_layer[i]  # [B]
        # 仅对仍active的样本评估是否更优
        better = (curr < selected_entropy) & active  # [B]
        if better.any():
            selected_idx = torch.where(better, torch.full_like(selected_idx, i), selected_idx)
            selected_entropy = torch.where(better, curr, selected_entropy)
        # 对未提升（>=）的样本，后续不再继续向前看
        stop_mask = (~(curr < selected_entropy)) & active
        if stop_mask.any():
            active = active & (~stop_mask)
        if not active.any():
            break

    return selected_idx  # [B]


def _select_layers_with_consistency(entropies_per_layer: list[torch.Tensor],
                                   layer_hidden_states: list[torch.Tensor],
                                   consistency_factor: float,
                                   *,
                                   use_embedding_cossim: bool = False,
                                   embedding_module: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                                   sampled_token_ids_per_layer: Optional[list[torch.Tensor]] = None,
                                   layer_decay_factor: Optional[float] = None) -> torch.Tensor:
    """
    基于熵值和与末层的一致性选择最优层。
    选择依据：argmax_i(-Entropy(h_i) * (1 - consistency_factor) + consistency_factor * cos_similarity(h_i, h_L))
    
    其中consistency_factor的含义：
    - 接近1：主要考虑相似性，熵值权重较小
    - 接近0：主要考虑熵值，相似性权重较小
    - 0.5：熵值和相似性权重相等

    Args:
        entropies_per_layer: 按层排列的熵向量列表，每个元素形状为 [B_sel]，长度为 L（层数）。
        layer_hidden_states: 各层的hidden states列表，每个元素形状为 [B_sel, H]。
        consistency_factor: 一致性因子，控制熵值和相似性的权重分配。值越大，相似性权重越高；值越小，熵值权重越高。
        use_embedding_cossim: 是否使用embedding计算cosine similarity。
        embedding_module: embedding模块，用于计算token向量。
        sampled_token_ids_per_layer: 每层采样的token ids列表。
        layer_decay_factor: 层衰减因子，用于计算加权和的末层数据。当不为None时，
                           末层数据将被替换为所有层的加权和，权重为末层=1，每往前一层
                           权重衰减至原来的layer_decay_factor倍。

    Returns:
        selected_idx: 形状 [B] 的长整型张量，给出每个样本最终选择的层索引（0..L-1）。
    """
    assert len(entropies_per_layer) == len(layer_hidden_states), "entropies_per_layer and layer_hidden_states must have the same length"
    assert len(entropies_per_layer) >= 1, "entropies_per_layer must have at least one element"
    
    L = len(entropies_per_layer)
    B_sel = entropies_per_layer[0].shape[0]  # 使用裁剪后的batch size
    device = entropies_per_layer[0].device
    
    # 计算每层的综合得分：-Entropy * (1 - consistency_factor) + consistency_factor * cos_similarity
    scores_per_layer = []
    if use_embedding_cossim:
        # 优先使用 embedding_module 来计算 token 向量
        assert embedding_module is not None and sampled_token_ids_per_layer is not None, "embedding_module and sampled_token_ids_per_layer must be not None"
        assert len(sampled_token_ids_per_layer) == L, "sampled_token_ids_per_layer must have the same length as entropies_per_layer"
        
        # 计算加权和的末层数据
        if layer_decay_factor is not None:
            # 计算所有层的加权和作为"末层"数据
            weighted_final_emb = torch.zeros_like(embedding_module(sampled_token_ids_per_layer[-1]))
            weight = 1.0
            for i in range(L-1, -1, -1):  # 从末层往前
                token_ids = sampled_token_ids_per_layer[i]  # [B_sel]
                emb = embedding_module(token_ids)  # [B_sel, H]
                weighted_final_emb += weight * emb
                weight *= layer_decay_factor
        else:
            # 使用原始末层数据
            final_ids = sampled_token_ids_per_layer[-1]  # [B_sel]
            weighted_final_emb = embedding_module(final_ids)  # [B_sel, H]
        
        for i in range(L):
            entropy = entropies_per_layer[i]  # [B_sel]
            token_ids = sampled_token_ids_per_layer[i]  # [B_sel]
            emb = embedding_module(token_ids)  # [B_sel, H]
            cos_sim = calculate_cosine_similarity(emb, weighted_final_emb)  # [B_sel]
            score = -entropy * (1 - consistency_factor) + consistency_factor * cos_sim
            scores_per_layer.append(score)
    else:
        # 使用 hidden state 余弦相似度
        if layer_decay_factor is not None:
            # 计算所有层的加权和作为"末层"数据
            weighted_final_hidden = torch.zeros_like(layer_hidden_states[-1])
            weight = 1.0
            for i in range(L-1, -1, -1):  # 从末层往前
                hidden = layer_hidden_states[i]  # [B_sel, H]
                weighted_final_hidden += weight * hidden
                weight *= layer_decay_factor
        else:
            # 使用原始末层数据
            weighted_final_hidden = layer_hidden_states[-1]  # [B_sel, H]
        
        for i in range(L):
            entropy = entropies_per_layer[i]  # [B_sel]
            hidden = layer_hidden_states[i]   # [B_sel, H]
            cos_sim = calculate_cosine_similarity(hidden, weighted_final_hidden)
            score = -entropy * (1 - consistency_factor) + consistency_factor * cos_sim
            scores_per_layer.append(score)
    
    # 找到每层得分的最大值，选择得分最高的层
    scores_tensor = torch.stack(scores_per_layer, dim=0)  # [L, B_sel]
    selected_idx = torch.argmax(scores_tensor, dim=0)     # [B_sel]
    
    return selected_idx


def calculate_multi_layer_entropy_selection(
    layer_hidden_states: list[torch.Tensor],
    lm_head: torch.nn.Module,
    *,
    consistency_factor: Optional[float] = None,
    use_embedding_cossim: bool = False,
    embedding_module: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    sampled_token_ids_per_layer: Optional[list[torch.Tensor]] = None,
    layer_decay_factor: Optional[float] = None,
    extra_selection_strategy: Optional[str] = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    多层熵值选择（适配 Transformers 的简化版）。

    与最初为 vLLM 设计的版本不同，这个实现假设已经拿到了所有层在同一 batch 上的 hidden states，
    并直接使用给定的 `lm_head` 计算每一层的 logits 与熵，从而选出最佳层。

    目前主要用于 decoder-only CausalLM（例如 gpt-oss）在 Transformers `generate` 流程中的按层熵选择解码。

    Args:
        layer_hidden_states:
            各层的 hidden states 列表，长度为 L；每个元素形状为 `[B, H]`，已经仅保留了当前 step 的最后一个位置。
        lm_head:
            语言模型头模块，用于将 hidden states 映射到词表 logits。
        consistency_factor, use_embedding_cossim, embedding_module, sampled_token_ids_per_layer, layer_decay_factor:
            与原始 vLLM 版本保持接口兼容的占位参数，目前在 Transformers 版中通常为默认值。
        extra_selection_strategy:
            额外的层选择策略，目前仅用于指示调用方所采用的策略名称（如 `"trough"`、`\"random_after\"` 等），
            实际策略逻辑可以在调用方中实现；本函数本身始终返回基于熵谷（最低熵）的层索引。

    Returns:
        selected_idx:
            形状 `[B]` 的长整型张量，每个样本选定的层索引（0..L-1），基于熵谷选择。
        entropies_per_layer:
            长度为 L 的列表，每个元素为该层的熵向量，形状 `[B]`。
        logits_per_layer:
            长度为 L 的列表，每个元素为该层对应的 logits，形状 `[B, V]`。
    """
    assert len(layer_hidden_states) >= 1, "layer_hidden_states must have at least one element"

    entropies_per_layer: list[torch.Tensor] = []
    logits_per_layer: list[torch.Tensor] = []

    for hs in layer_hidden_states:
        logits = lm_head(hs)  # [B, V]
        logits_per_layer.append(logits)
        entropy = calculate_information_entropy(logits)  # [B]
        entropies_per_layer.append(entropy)

    # 仅基于熵本身选出“熵谷”所在的层；之后的具体策略（如 random_after）由上层调用实现
    selected_idx = _select_layers_from_entropies(
        entropies_per_layer,
        layer_hidden_states=None,
        consistency_factor=None,
        use_embedding_cossim=False,
        sampled_token_ids_per_layer=None,
        embedding_module=None,
        layer_decay_factor=None,
    )

    return selected_idx, entropies_per_layer, logits_per_layer