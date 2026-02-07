import torch
import torch.nn.functional as F


def calculate_information_entropy(logits: torch.Tensor) -> torch.Tensor:
    # size of logits: [batch_size, vocab_size]

    # Step 0: 将 logits 数据类型转换为 float
    logits = logits.float()

    # Step 1: 计算 softmax 概率分布（注意使用 dim=-1 来保证维度正确）
    probs = F.softmax(logits, dim=-1)

    # Step 2: 计算信息熵（以 2 为底，单位是 bits）
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1)  # 加一个小量防止 log(0)

    return entropy  # shape: [batch_size]


def calculate_topk_information_entropy(logits: torch.Tensor, k: int) -> torch.Tensor:
    # size of logits: [batch_size, vocab_size]

    # Step 0: 将 logits 数据类型转换为 float
    logits = logits.float()

    # Step 1: 获取 top-k 的 logit 及其对应的概率
    topk_logits, indices = torch.topk(logits, k=k, dim=-1)

    # Step 2: 对 top-k 的 logit 进行 softmax 归一化
    topk_probs = F.softmax(topk_logits, dim=-1)

    # Step 3: 在 top-k 概率上计算信息熵
    entropy = -torch.sum(topk_probs * torch.log2(topk_probs + 1e-12), dim=-1)

    return entropy  # shape: [batch_size]
