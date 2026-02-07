"""
Optional: collect divergence points and run path comparison using
transformers-dynamic + GptOssForCausalLM (entropy_decoding from generation/utils.py).

Run from repo root with PYTHONPATH including transformers-dynamic and gpqa/baselines:

  cd /path/to/Dynamic-Decoding
  PYTHONPATH="transformers-dynamic:gpqa/baselines:$PYTHONPATH" python gpqa/baselines/extensions/evaluation/run_evaluation_hf_gpt_oss.py \
    --model /path/to/gpt-oss-checkpoint \
    --data gpqa/baselines/dataset/gpqa_diamond.csv \
    --output evaluation_results_hf \
    --max_examples 5 \
    --max_new_tokens 128

Requires: transformers-dynamic package on path (GptOssForCausalLM, entropy_decoding),
          gpqa baselines on path (extensions.utils.gpqa_loader).
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure gpqa baselines and transformers-dynamic are on path when run from repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent.parent
_BASELINES = _REPO_ROOT / "gpqa" / "baselines"
_TRANSFORMERS_DYNAMIC = _REPO_ROOT / "transformers-dynamic"
for _p in [_BASELINES, _TRANSFORMERS_DYNAMIC]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        if _p == _TRANSFORMERS_DYNAMIC:
            sys.path.insert(0, str(_REPO_ROOT))

import torch

# After path setup
from extensions.utils.gpqa_loader import (
    load_gpqa_diamond,
    create_zero_shot_prompt,
    extract_answer_from_text,
)


def _import_transformers():
    """Import from transformers (dynamic); prefer explicit GptOss if available."""
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from transformers import GptOssForCausalLM
    except ImportError:
        GptOssForCausalLM = None
    return AutoTokenizer, AutoModelForCausalLM, GptOssForCausalLM, transformers


def collect_divergence_points_hf(
    model,
    tokenizer,
    dataset: List[Any],
    max_new_tokens: int = 512,
    device: str = "cuda",
    verbose: bool = False,
) -> List[Dict]:
    """
    使用 HuggingFace GptOssForCausalLM + entropy_decoding="trough" 收集分歧点。
    返回与 gpqa collect_divergence_points 相同结构的列表，便于复用 path 对比与扰动分析。
    """
    divergence_points = []
    generation_config = getattr(model, "generation_config", None) or {}
    if hasattr(generation_config, "copy"):
        gen_config = generation_config.copy()
    else:
        gen_config = dict(generation_config) if generation_config else {}

    gen_config["entropy_decoding"] = "trough"
    gen_config["entropy_record_tokens"] = True
    gen_config["return_dict_in_generate"] = True
    gen_config["max_new_tokens"] = max_new_tokens

    for question_id, example in enumerate(dataset):
        if verbose and question_id % 10 == 0:
            print(f"Processing question {question_id}/{len(dataset)}...")

        prompt = create_zero_shot_prompt(example)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    entropy_decoding="trough",
                    entropy_record_tokens=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                )
        except Exception as e:
            if verbose:
                print(f"Error question {question_id}: {e}")
            continue

        sequences = out.sequences
        entropy_per_layer_token_ids = getattr(out, "entropy_per_layer_token_ids", None)
        entropy_selected_layer_indices = getattr(out, "entropy_selected_layer_indices", None)

        if entropy_per_layer_token_ids is None or entropy_selected_layer_indices is None:
            continue

        input_len = inputs["input_ids"].shape[1]
        num_layers = entropy_per_layer_token_ids[0].shape[1] if len(entropy_per_layer_token_ids[0].shape) > 1 else 0
        final_layer_idx = num_layers - 1

        # Decode full sequence for context_before
        gen_ids = sequences[0][input_len:].tolist()
        generated_tokens = [tokenizer.decode([tid]) for tid in gen_ids]

        for token_pos, (per_layer_t, selected_idx_t) in enumerate(zip(entropy_per_layer_token_ids, entropy_selected_layer_indices)):
            if token_pos >= len(generated_tokens):
                break
            # per_layer_t: [batch, num_layers], selected_idx_t: [batch]
            selected_layer = int(selected_idx_t[0].item())
            per_layer = per_layer_t[0]
            selected_token_id = int(per_layer[selected_layer].item())
            final_token_id = int(per_layer[final_layer_idx].item())

            if selected_token_id == final_token_id:
                continue

            selected_token = tokenizer.decode([selected_token_id])
            final_token = tokenizer.decode([final_token_id])
            context_before = generated_tokens[:token_pos]

            intermediate_layers = []
            for layer_idx in range(selected_layer, final_layer_idx + 1):
                tid = int(per_layer[layer_idx].item())
                intermediate_layers.append({
                    "layer_idx": layer_idx,
                    "token": tokenizer.decode([tid]),
                    "entropy": None,
                })

            divergence_point = {
                "question_id": question_id,
                "token_position": token_pos,
                "selected_layer": selected_layer,
                "final_layer": final_layer_idx,
                "selected_token": selected_token,
                "final_token": final_token,
                "intermediate_layers": intermediate_layers,
                "entropies": [],
                "question": example.question,
                "correct_answer_index": example.correct_index,
                "context_before": context_before,
                "base_prompt": prompt,
            }
            divergence_points.append(divergence_point)
            if verbose:
                print(f"  Divergence at token {token_pos}: layer {selected_layer} vs {final_layer_idx}")

    return divergence_points


def compare_paths_hf(
    model,
    tokenizer,
    divergence_point: Dict,
    full_context_before: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> Dict:
    """从分歧点的两个 token 分别延续生成（标准 decode，无 entropy_decoding），对比答案正确性。"""
    selected_token = divergence_point["selected_token"]
    final_token = divergence_point["final_token"]
    correct_answer_index = divergence_point["correct_answer_index"]

    def run_path(prefix: str) -> tuple:
        inp = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = extract_answer_from_text(text)
        correct = answer is not None and ord(answer) - ord("A") == correct_answer_index
        return text, answer, correct

    path_a_text, path_a_answer, path_a_correct = run_path(full_context_before + selected_token)
    path_b_text, path_b_answer, path_b_correct = run_path(full_context_before + final_token)

    return {
        "path_a": {"text": path_a_text, "answer": path_a_answer, "is_correct": path_a_correct},
        "path_b": {"text": path_b_text, "answer": path_b_answer, "is_correct": path_b_correct},
        "accuracy_diff": 1.0 if path_a_correct and not path_b_correct else (-1.0 if not path_a_correct and path_b_correct else 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="HF GptOss entropy-decoding divergence evaluation")
    parser.add_argument("--model", type=str, required=True, help="GptOss checkpoint path or name")
    parser.add_argument("--data", type=str, required=True, help="GPQA diamond CSV path")
    parser.add_argument("--output", type=str, default="evaluation_results_hf", help="Output directory")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default=None, help="设备字符串，如 cuda, cuda:0, cpu；与 --device_id 二选一")
    parser.add_argument("--device_id", type=int, default=None, help="指定 GPU 编号（0, 1, 2, ...）；传入后使用 cuda:<device_id>，需在 GPU 上跑时推荐使用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_path_comparison", action="store_true", help="Run path comparison for each divergence point")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # 确定设备：优先 --device_id，其次 --device
    if args.device_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device_id 但 CUDA 不可用")
        device = f"cuda:{args.device_id}"
    elif args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f"Using device: {args.device}")

    AutoTokenizer, AutoModelForCausalLM, GptOssForCausalLM, _ = _import_transformers()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_gpqa_diamond(args.data, seed=args.seed)
    if args.max_examples:
        dataset = dataset[: args.max_examples]
    print(f"Loaded {len(dataset)} examples")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if GptOssForCausalLM is not None:
        model = GptOssForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(args.device)
    if getattr(model.config, "model_type", None) != "gpt_oss":
        print("Warning: model_type is not gpt_oss; entropy_decoding may not be used.")

    print("Collecting divergence points (entropy_decoding=trough)...")
    divergence_points = collect_divergence_points_hf(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_new_tokens=args.max_tokens,
        device=args.device,
        verbose=args.verbose,
    )
    print(f"Found {len(divergence_points)} divergence points")

    div_file = out_dir / "divergence_points_hf.json"
    with open(div_file, "w", encoding="utf-8") as f:
        json.dump(divergence_points, f, indent=2, ensure_ascii=False)
    print(f"Saved to {div_file}")

    if args.run_path_comparison and divergence_points:
        print("Running path comparison...")
        path_comparisons = []
        for i, div_point in enumerate(divergence_points):
            if args.verbose and i % 10 == 0:
                print(f"  Path comparison {i+1}/{len(divergence_points)}...")
            base_prompt = create_zero_shot_prompt(dataset[div_point["question_id"]])
            context_before = "".join(div_point.get("context_before", []))
            full_context = base_prompt + context_before
            try:
                comp = compare_paths_hf(
                    model=model,
                    tokenizer=tokenizer,
                    divergence_point=div_point,
                    full_context_before=full_context,
                    max_new_tokens=args.max_tokens,
                    device=args.device,
                )
                path_comparisons.append(comp)
            except Exception as e:
                if args.verbose:
                    print(f"  Error: {e}")
                path_comparisons.append({"path_a": {"is_correct": False}, "path_b": {"is_correct": False}, "accuracy_diff": 0.0})

        comp_file = out_dir / "path_comparisons_hf.json"
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(path_comparisons, f, indent=2, ensure_ascii=False)
        a_ok = sum(1 for c in path_comparisons if c.get("path_a", {}).get("is_correct", False))
        b_ok = sum(1 for c in path_comparisons if c.get("path_b", {}).get("is_correct", False))
        print(f"Path A accuracy: {a_ok}/{len(path_comparisons)}, Path B accuracy: {b_ok}/{len(path_comparisons)}")
        print(f"Saved to {comp_file}")


if __name__ == "__main__":
    main()
