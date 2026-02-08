"""
Optional: collect divergence points and run path comparison using
transformers-dynamic + GptOssForCausalLM (entropy_decoding from generation/utils.py).

Run from repo root with PYTHONPATH including transformers-dynamic and gpqa/baselines:

  cd /path/to/Dynamic-Decoding
  PYTHONPATH="transformers-dynamic:gpqa/baselines:$PYTHONPATH" python gpqa/baselines/extensions/evaluation/run_evaluation_hf_gpt_oss.py \
    --model /path/to/your/gpt-oss-20b-trough-checkpoint \
    --data gpqa/baselines/dataset/gpqa_diamond.csv \
    --output evaluation_results_hf \
    --device_id 0 \
    --run_path_comparison \
    --full_evaluation \
    --verbose

**重要**：验证「trough 后存在负面扰动」假设时，必须使用 `--run_path_comparison` 进行路径对比。
`--full_evaluation` 将额外运行扰动分析、报告生成与可视化。

Requires: transformers-dynamic package on path (GptOssForCausalLM, entropy_decoding),
          gpqa baselines on path (extensions.utils.gpqa_loader).
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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
from extensions.evaluation.perturbation_evaluation import (
    analyze_perturbation_start,
    evaluate_perturbation_impact,
)
from extensions.evaluation.visualization import create_all_visualizations


def _generate_text_report(
    evaluation_results: Dict,
    divergence_points: List[Dict],
    path_comparisons: List[Dict],
    perturbation_analyses: List[Dict],
    output_file: Path,
) -> None:
    """生成文本报告（与 run_evaluation 格式一致）。"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Conservative Dynamic Perturbation Analysis Report (HF GptOss)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Overall Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total divergence points: {evaluation_results['total_divergence_points']}\n")
        total_steps = evaluation_results.get("total_token_steps", 0)
        div_rate = evaluation_results.get("divergence_rate", 0.0)
        f.write(f"Total token steps (across examples): {total_steps}\n")
        f.write(f"Divergence rate: {div_rate:.2%}\n")
        f.write(f"Path A (selected layer) accuracy: {evaluation_results['path_a_accuracy']:.2%}\n")
        f.write(f"Path B (final layer) accuracy: {evaluation_results['path_b_accuracy']:.2%}\n")
        f.write(f"Accuracy improvement: {evaluation_results['accuracy_improvement']:.2%}\n")
        f.write(f"Average entropy increase: {evaluation_results['avg_entropy_increase']:.4f}\n")
        f.write(f"Average affected layers: {evaluation_results['avg_affected_layers']:.2f}\n\n")
        f.write("Perturbation Start Layer Distribution\n")
        f.write("-" * 80 + "\n")
        for layer, count in sorted(evaluation_results.get("perturbation_start_distribution", {}).items()):
            f.write(f"Layer {layer}: {count} occurrences\n")
        f.write("\n")
        neg_count = evaluation_results.get("negative_subset_count", 0)
        neg_dist = evaluation_results.get("negative_subset_perturbation_start_distribution", {})
        f.write("Negative Subset (Path A correct, Path B wrong) Perturbation Start Distribution\n")
        f.write("-" * 80 + "\n")
        f.write(f"Negative subset size: {neg_count}\n")
        for layer, count in sorted(neg_dist.items()):
            f.write(f"Layer {layer}: {count} occurrences\n")
        f.write("\n")
        by_depth = evaluation_results.get("by_trough_depth", {})
        f.write("By Trough Depth (final_layer - selected_layer)\n")
        f.write("-" * 80 + "\n")
        for label in ["shallow", "mid", "deep"]:
            if label not in by_depth:
                continue
            d = by_depth[label]
            f.write(f"  {label}: count={d['count']}, Path A acc={d['path_a_accuracy']:.2%}, Path B acc={d['path_b_accuracy']:.2%}\n")
        f.write("\n")
        f.write("Layer-Entropy Correlation\n")
        f.write("-" * 80 + "\n")
        for layer_idx in sorted(evaluation_results.get("layer_entropy_correlation", {}).keys()):
            corr = evaluation_results["layer_entropy_correlation"][layer_idx]
            f.write(f"Layer {layer_idx}: mean={corr['mean_entropy']:.4f}, std={corr['std_entropy']:.4f}, count={corr['count']}\n")
        f.write("\nSample Cases\n")
        f.write("-" * 80 + "\n")
        for i in range(min(5, len(divergence_points))):
            div_point = divergence_points[i]
            comparison = path_comparisons[i] if i < len(path_comparisons) else {}
            analysis = perturbation_analyses[i] if i < len(perturbation_analyses) else {}
            f.write(f"\nCase {i+1}:\n")
            f.write(f"  Question ID: {div_point['question_id']}\n")
            f.write(f"  Token Position: {div_point['token_position']}\n")
            f.write(f"  Selected Layer: {div_point['selected_layer']}, Token: '{div_point['selected_token']}'\n")
            f.write(f"  Final Layer: {div_point['final_layer']}, Token: '{div_point['final_token']}'\n")
            f.write(f"  Path A Correct: {comparison.get('path_a', {}).get('is_correct', 'N/A')}\n")
            f.write(f"  Path B Correct: {comparison.get('path_b', {}).get('is_correct', 'N/A')}\n")
            f.write(f"  Perturbation Start Layer: {analysis.get('perturbation_start_layer', 'N/A')}\n")


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
    return_stats: bool = False,
) -> List[Dict] | Tuple[List[Dict], Dict]:
    """
    使用 HuggingFace GptOssForCausalLM + entropy_decoding="trough" 收集分歧点。
    返回与 gpqa collect_divergence_points 相同结构的列表，便于复用 path 对比与扰动分析。
    """
    divergence_points = []
    total_token_steps = 0
    num_examples_processed = 0
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
        gen_ids = sequences[0][input_len:].tolist()
        total_token_steps += len(gen_ids)
        num_examples_processed += 1
        num_layers = entropy_per_layer_token_ids[0].shape[1] if len(entropy_per_layer_token_ids[0].shape) > 1 else 0
        final_layer_idx = num_layers - 1

        # Decode full sequence for context_before
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

    if return_stats:
        stats = {
            "total_token_steps": total_token_steps,
            "num_examples_processed": num_examples_processed,
        }
        return divergence_points, stats
    return divergence_points


def compare_paths_hf(
    model,
    tokenizer,
    divergence_point: Dict,
    full_context_before: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
    do_sample: bool = False,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> Dict:
    """从分歧点的两个 token 分别延续生成（标准 decode，无 entropy_decoding），对比答案正确性。"""
    selected_token = divergence_point["selected_token"]
    final_token = divergence_point["final_token"]
    correct_answer_index = divergence_point["correct_answer_index"]

    def run_path(prefix: str, step_seed: Optional[int] = None) -> tuple:
        if step_seed is not None:
            torch.manual_seed(step_seed)
            if hasattr(random, "seed"):
                random.seed(step_seed)
        inp = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = extract_answer_from_text(text)
        correct = answer is not None and ord(answer) - ord("A") == correct_answer_index
        return text, answer, correct

    path_a_text, path_a_answer, path_a_correct = run_path(
        full_context_before + selected_token, seed if do_sample else None
    )
    path_b_text, path_b_answer, path_b_correct = run_path(
        full_context_before + final_token, (seed + 1) if (do_sample and seed is not None) else None
    )

    return {
        "path_a": {"text": path_a_text, "answer": path_a_answer, "is_correct": path_a_correct},
        "path_b": {"text": path_b_text, "answer": path_b_answer, "is_correct": path_b_correct},
        "accuracy_diff": 1.0 if path_a_correct and not path_b_correct else (-1.0 if not path_a_correct and path_b_correct else 0.0),
    }


def compare_paths_with_sampling_hf(
    model,
    tokenizer,
    divergence_point: Dict,
    full_context_before: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
    k: int = 5,
    temperature: float = 0.7,
    seed: Optional[int] = None,
) -> Dict:
    """从分歧点的两个 token 分别延续采样 k 次，对比两条路径的正确率分布。"""
    selected_token = divergence_point["selected_token"]
    final_token = divergence_point["final_token"]
    correct_answer_index = divergence_point["correct_answer_index"]

    path_a_correct_count = 0
    path_b_correct_count = 0
    path_a_results = []
    path_b_results = []

    for i in range(k):
        step_seed = (seed + i) if seed is not None else None
        pa_text, pa_answer = _run_path_once(
            model, tokenizer, full_context_before + selected_token,
            max_new_tokens, device, True, temperature, step_seed,
        )
        pb_text, pb_answer = _run_path_once(
            model, tokenizer, full_context_before + final_token,
            max_new_tokens, device, True, temperature,
            (seed + i + 100) if seed is not None else None,
        )
        pa_correct = pa_answer is not None and ord(pa_answer) - ord("A") == correct_answer_index
        pb_correct = pb_answer is not None and ord(pb_answer) - ord("A") == correct_answer_index
        if pa_correct:
            path_a_correct_count += 1
        if pb_correct:
            path_b_correct_count += 1
        path_a_results.append({"answer": pa_answer, "is_correct": pa_correct})
        path_b_results.append({"answer": pb_answer, "is_correct": pb_correct})

    return {
        "k": k,
        "temperature": temperature,
        "path_a_correct_count": path_a_correct_count,
        "path_b_correct_count": path_b_correct_count,
        "path_a_accuracy": path_a_correct_count / k if k > 0 else 0.0,
        "path_b_accuracy": path_b_correct_count / k if k > 0 else 0.0,
        "path_a_better": path_a_correct_count > path_b_correct_count,
        "path_a_results": path_a_results,
        "path_b_results": path_b_results,
    }


def _run_path_once(
    model, tokenizer, prefix: str, max_new_tokens: int, device: str,
    do_sample: bool, temperature: float, seed: Optional[int],
) -> Tuple[str, Optional[str]]:
    """单次路径生成，返回 (text, answer)。"""
    if seed is not None:
        torch.manual_seed(seed)
        if hasattr(random, "seed"):
            random.seed(seed)
    inp = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = extract_answer_from_text(text)
    return text, answer


def main():
    parser = argparse.ArgumentParser(
        description="HF GptOss entropy-decoding divergence evaluation. "
        "**重要**：验证「trough 后存在负面扰动」假设时，必须使用 --run_path_comparison。"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="GptOss checkpoint 路径或 HuggingFace 模型 ID，例如 /path/to/your/gpt-oss-20b-trough-checkpoint 或 username/model-name"
    )
    parser.add_argument("--data", type=str, required=True, help="GPQA diamond CSV path")
    parser.add_argument("--output", type=str, default="evaluation_results_hf", help="Output directory")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default=None, help="设备字符串，如 cuda, cuda:0, cpu；与 --device_id 二选一")
    parser.add_argument("--device_id", type=int, default=None, help="指定 GPU 编号（0, 1, 2, ...）；传入后使用 cuda:<device_id>，需在 GPU 上跑时推荐使用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run_path_comparison", action="store_true",
        help="【必需验证假设时使用】对每个分歧点运行路径对比（trough token vs 末层 token），对比答案正确性"
    )
    parser.add_argument(
        "--full_evaluation", action="store_true",
        help="完整评估：路径对比 + 扰动分析 + 报告 + 可视化。隐含 --run_path_comparison"
    )
    parser.add_argument("--sampling_k", type=int, default=0, help="若 >0，对前 N 个分歧点做 k 次采样路径对比")
    parser.add_argument("--sampling_max_divergence_points", type=int, default=20, help="参与多采样的最大分歧点数量")
    parser.add_argument("--sampling_temperature", type=float, default=0.7, help="多采样时的温度")
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

    run_path_comparison = args.run_path_comparison or args.full_evaluation

    print("Collecting divergence points (entropy_decoding=trough)...")
    divergence_points, collect_stats = collect_divergence_points_hf(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_new_tokens=args.max_tokens,
        device=args.device,
        verbose=args.verbose,
        return_stats=True,
    )
    print(f"Found {len(divergence_points)} divergence points")

    total_token_steps = collect_stats.get("total_token_steps", 0)
    divergence_rate = (len(divergence_points) / total_token_steps) if total_token_steps > 0 else 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    div_file = out_dir / f"divergence_points_hf_{timestamp}.json"
    with open(div_file, "w", encoding="utf-8") as f:
        json.dump(divergence_points, f, indent=2, ensure_ascii=False)
    print(f"Saved to {div_file}")

    path_comparisons: List[Dict] = []
    if run_path_comparison and divergence_points:
        print("Running path comparison (greedy)...")
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
                    seed=args.seed,
                )
                path_comparisons.append(comp)
            except Exception as e:
                if args.verbose:
                    print(f"  Error: {e}")
                path_comparisons.append({"path_a": {"is_correct": False}, "path_b": {"is_correct": False}, "accuracy_diff": 0.0})

        comp_file = out_dir / f"path_comparisons_hf_{timestamp}.json"
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(path_comparisons, f, indent=2, ensure_ascii=False)
        a_ok = sum(1 for c in path_comparisons if c.get("path_a", {}).get("is_correct", False))
        b_ok = sum(1 for c in path_comparisons if c.get("path_b", {}).get("is_correct", False))
        print(f"Path A accuracy: {a_ok}/{len(path_comparisons)}, Path B accuracy: {b_ok}/{len(path_comparisons)}")
        print(f"Saved to {comp_file}")

    # 多采样路径对比（可选）
    if args.sampling_k > 0 and divergence_points:
        n_sampling = min(args.sampling_max_divergence_points, len(divergence_points))
        print(f"Running multi-sample path comparison (k={args.sampling_k}) on {n_sampling} divergence points...")
        sampling_results = []
        for i in range(n_sampling):
            if args.verbose and i % 5 == 0:
                print(f"  Sampling divergence point {i+1}/{n_sampling}...")
            div_point = divergence_points[i]
            base_prompt = create_zero_shot_prompt(dataset[div_point["question_id"]])
            context_before = "".join(div_point.get("context_before", []))
            full_context = base_prompt + context_before
            try:
                res = compare_paths_with_sampling_hf(
                    model=model,
                    tokenizer=tokenizer,
                    divergence_point=div_point,
                    full_context_before=full_context,
                    max_new_tokens=args.max_tokens,
                    device=args.device,
                    k=args.sampling_k,
                    temperature=args.sampling_temperature,
                    seed=args.seed,
                )
                sampling_results.append(res)
            except Exception as e:
                if args.verbose:
                    print(f"  Error: {e}")
        if sampling_results:
            path_a_better = sum(1 for r in sampling_results if r.get("path_a_better", False))
            print(f"  Path A better in {path_a_better}/{len(sampling_results)} divergence points")
            samp_file = out_dir / f"sampling_comparisons_hf_{timestamp}.json"
            with open(samp_file, "w", encoding="utf-8") as f:
                json.dump(sampling_results, f, indent=2, ensure_ascii=False)
            print(f"Saved to {samp_file}")

    # 完整评估：扰动分析、报告、可视化
    if args.full_evaluation and divergence_points and path_comparisons:
        print("Running perturbation analysis...")
        perturbation_analyses = [analyze_perturbation_start(dp) for dp in divergence_points]
        pert_file = out_dir / f"perturbation_analyses_hf_{timestamp}.json"
        with open(pert_file, "w", encoding="utf-8") as f:
            json.dump(perturbation_analyses, f, indent=2, ensure_ascii=False)
        print(f"Saved to {pert_file}")

        evaluation_results = evaluate_perturbation_impact(
            divergence_points=divergence_points,
            path_comparisons=path_comparisons,
        )
        evaluation_results["total_token_steps"] = total_token_steps
        evaluation_results["divergence_rate"] = divergence_rate
        evaluation_results["num_examples_processed"] = collect_stats.get("num_examples_processed", 0)

        eval_file = out_dir / f"evaluation_results_hf_{timestamp}.json"
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"Saved to {eval_file}")

        report_file = out_dir / f"report_hf_{timestamp}.txt"
        _generate_text_report(
            evaluation_results,
            divergence_points,
            path_comparisons,
            perturbation_analyses,
            report_file,
        )
        print(f"Saved report to {report_file}")

        print("Creating visualizations...")
        try:
            create_all_visualizations(
                divergence_points=divergence_points,
                evaluation_results=evaluation_results,
                output_dir=out_dir,
            )
        except Exception as e:
            print(f"Warning: Failed to create visualizations: {e}")

        print("=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        print(f"Total divergence points: {evaluation_results['total_divergence_points']}")
        print(f"Divergence rate: {evaluation_results.get('divergence_rate', 0):.2%}")
        print(f"Path A (trough layer) accuracy: {evaluation_results['path_a_accuracy']:.2%}")
        print(f"Path B (final layer) accuracy: {evaluation_results['path_b_accuracy']:.2%}")
        print(f"Accuracy improvement: {evaluation_results['accuracy_improvement']:.2%}")
        print("=" * 80)


if __name__ == "__main__":
    main()
