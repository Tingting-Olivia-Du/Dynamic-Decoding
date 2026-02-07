import matplotlib.pyplot as plt


# generated_tokens: list of generated tokens (str)
# entropies: list of corresponding entropy values (float)

def plot_entropy(generated_tokens, entropies):
    token_labels = [f"'{t}'" for t in generated_tokens]  # 添加引号以增强可读性
    positions = range(1, len(generated_tokens) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(positions, entropies, marker='o', linestyle='-', color='b', label='Entropy per Token')

    # 标注每个 token 的具体值
    for i, label in enumerate(token_labels):
        plt.annotate(label, (positions[i], entropies[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title('Information Entropy per Generated Token', fontsize=14)
    plt.xlabel('Token Position', fontsize=12)
    plt.ylabel('Entropy (bits)', fontsize=12)
    plt.xticks(positions)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/entropy_per_token.png')  # 保存图像
