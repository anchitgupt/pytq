"""Benchmark: perplexity on WikiText-2 with TurboQuant KV cache."""
import argparse
import json
import math
import os
from typing import Callable, Optional

import torch
from tqdm import tqdm

from pytq.kv_cache import TurboQuantKVCache


def compute_perplexity(
    model,
    tokenizer,
    dataset_text: str,
    cache_factory=None,  # callable returning a fresh TurboQuantKVCache, or None for baseline
    max_length: int = 2048,
    stride: int = 512,
    device: str = "cpu",
) -> float:
    """Compute perplexity using a sliding window approach.

    Args:
        model: A HuggingFace CausalLM model.
        tokenizer: Matching tokenizer.
        dataset_text: Raw text to evaluate on.
        cache_factory: Optional callable returning a fresh TurboQuantKVCache per
            window. When None, the model runs without a custom KV cache (baseline).
        max_length: Window size in tokens.
        stride: Step size between consecutive windows.
        device: Torch device string.

    Returns:
        Perplexity as a float.
    """
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)
    nlls = []
    prev_end = 0
    max_windows = 50

    # Build list of (begin_loc, end_loc, target_begin) tuples
    windows = []
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end
        windows.append((begin_loc, end_loc, trg_len))
        prev_end = end_loc
        if end_loc == seq_len:
            break

    windows = windows[:max_windows]

    model.eval()
    with torch.no_grad():
        for begin_loc, end_loc, trg_len in tqdm(windows, desc="Perplexity windows"):
            input_chunk = input_ids[:, begin_loc:end_loc]

            target_ids = input_chunk.clone()
            # Mask out the context tokens that are not part of the target
            target_ids[:, :-trg_len] = -100

            kwargs = {}
            if cache_factory is not None:
                kwargs["past_key_values"] = cache_factory()

            outputs = model(
                input_chunk,
                labels=target_ids,
                **kwargs,
            )
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)

    total_nll = torch.stack(nlls).sum()
    total_tokens = sum(trg_len for _, _, trg_len in windows)
    ppl = math.exp(total_nll.item() / total_tokens)
    return ppl


def run_quality_benchmark(
    model_name: str,
    bits_list: list,
    device: str = "cpu",
    output_dir: str = ".",
):
    """Run the quality benchmark: baseline + one run per requested bit-width.

    Args:
        model_name: HuggingFace model identifier.
        bits_list: List of integer bit-widths to evaluate.
        device: Torch device string.
        output_dir: Directory where the JSON results file is written.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    model = model.to(device)
    model.eval()

    print("Loading WikiText-2 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = "\n\n".join(dataset["text"])

    # Detect head_dim from model config
    config = model.config
    if hasattr(config, "head_dim"):
        head_dim = config.head_dim
    elif hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
        head_dim = config.hidden_size // config.num_attention_heads
    else:
        raise ValueError(
            "Cannot determine head_dim from model config. "
            "Expected config.head_dim or (config.hidden_size, config.num_attention_heads)."
        )

    print(f"Detected head_dim={head_dim}")

    results = {}

    # Baseline (no quantization)
    print("\n--- Baseline (no KV cache quantization) ---")
    baseline_ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_text=dataset_text,
        cache_factory=None,
        device=device,
    )
    results["baseline"] = {"perplexity": baseline_ppl}
    print(f"Baseline perplexity: {baseline_ppl:.4f}")

    # Per bit-width runs
    for bits in bits_list:
        print(f"\n--- TurboQuant KV cache: {bits}-bit ---")
        factory = lambda b=bits, hd=head_dim, d=device: TurboQuantKVCache(
            bits=b, head_dim=hd, device=d
        )
        quant_ppl = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_text=dataset_text,
            cache_factory=factory,
            device=device,
        )
        delta = quant_ppl - baseline_ppl
        pct_delta = (delta / baseline_ppl) * 100.0
        results[f"{bits}bit"] = {
            "perplexity": quant_ppl,
            "delta": delta,
            "pct_delta": pct_delta,
        }
        print(
            f"{bits}-bit perplexity: {quant_ppl:.4f}  "
            f"delta: {delta:+.4f}  ({pct_delta:+.2f}%)"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "quality_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark perplexity on WikiText-2 with TurboQuant KV cache."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model identifier (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Bit-widths to benchmark (default: 1 2 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save JSON results (default: current directory)",
    )
    args = parser.parse_args()

    run_quality_benchmark(
        model_name=args.model,
        bits_list=args.bits,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
