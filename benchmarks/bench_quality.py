"""Benchmark: perplexity on WikiText-2 with TurboQuant KV cache."""
import argparse
import json
import math
import os
from typing import Callable, Optional

import torch
from tqdm import tqdm

from pytq.kv_cache import TurboQuantKVCache


def _quantize_key_tensor(key: torch.Tensor, bits: int, head_dim: int) -> torch.Tensor:
    """Quantize and immediately dequantize a key tensor to simulate TurboQuant compression.

    This measures the quality impact of quantization without needing to integrate
    with HuggingFace's internal cache mechanics.

    Args:
        key: (batch, n_heads, seq_len, head_dim)
        bits: Bit-width for quantization.
        head_dim: Dimension per attention head.

    Returns:
        Reconstructed key tensor with same shape.
    """
    from pytq.quantize_mse import TurboQuantMSE

    batch, n_heads, seq_len, hd = key.shape
    result = torch.zeros_like(key)
    for h in range(n_heads):
        q = TurboQuantMSE(dim=hd, bits=bits, seed=h)
        k_head = key[:, h, :, :]  # (batch, seq_len, head_dim)
        qt = q.quantize(k_head)
        result[:, h, :, :] = q.dequantize(qt)
    return result


def compute_perplexity(
    model,
    tokenizer,
    dataset_text: str,
    quantize_bits: int = 0,
    head_dim: int = 64,
    max_length: int = 2048,
    stride: int = 512,
    device: str = "cpu",
) -> float:
    """Compute perplexity using a sliding window approach.

    When quantize_bits > 0, installs forward hooks on attention layers to
    quantize/dequantize key tensors, simulating TurboQuant KV cache compression.

    Args:
        model: A HuggingFace CausalLM model.
        tokenizer: Matching tokenizer.
        dataset_text: Raw text to evaluate on.
        quantize_bits: If > 0, quantize keys at this bit-width. 0 = baseline.
        head_dim: Dimension per attention head.
        max_length: Window size in tokens.
        stride: Step size between consecutive windows.
        device: Torch device string.

    Returns:
        Perplexity as a float.
    """
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)
    prev_end = 0
    max_windows = 50

    windows = []
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end
        windows.append((begin_loc, end_loc, trg_len))
        prev_end = end_loc
        if end_loc == seq_len:
            break
    windows = windows[:max_windows]

    # Install hooks to quantize keys if needed
    hooks = []
    if quantize_bits > 0:
        for module in model.modules():
            # Find attention modules — they compute key projections
            module_name = type(module).__name__.lower()
            if "attention" in module_name and hasattr(module, "k_proj"):
                orig_k_proj = module.k_proj

                class QuantizeKeyWrapper(torch.nn.Module):
                    def __init__(self, original, bits, hd):
                        super().__init__()
                        self.original = original
                        self.bits = bits
                        self.hd = hd

                    def forward(self, x):
                        k = self.original(x)
                        # k shape: (batch, seq_len, n_heads * head_dim) or (batch, seq_len, n_kv_heads * head_dim)
                        batch, seq_len, total_dim = k.shape
                        n_heads = total_dim // self.hd
                        k_reshaped = k.view(batch, seq_len, n_heads, self.hd).transpose(1, 2)
                        k_quantized = _quantize_key_tensor(k_reshaped, self.bits, self.hd)
                        return k_quantized.transpose(1, 2).reshape(batch, seq_len, total_dim)

                wrapper = QuantizeKeyWrapper(orig_k_proj, quantize_bits, head_dim)
                module.k_proj = wrapper
                hooks.append((module, "k_proj", orig_k_proj))

    model.eval()
    nlls = []
    with torch.no_grad():
        for begin_loc, end_loc, trg_len in tqdm(windows, desc="Perplexity windows"):
            input_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_chunk.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)

    # Remove hooks
    for module, attr_name, orig in hooks:
        setattr(module, attr_name, orig)

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

    # Detect model max position length
    max_length = 2048
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        if hasattr(config, attr):
            max_length = min(getattr(config, attr), 2048)
            break
    stride = min(512, max_length // 4)
    print(f"Using max_length={max_length}, stride={stride}")

    results = {}

    # Baseline (no quantization)
    print("\n--- Baseline (no KV cache quantization) ---")
    baseline_ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_text=dataset_text,
        quantize_bits=0,
        head_dim=head_dim,
        max_length=max_length,
        stride=stride,
        device=device,
    )
    results["baseline"] = {"perplexity": baseline_ppl}
    print(f"Baseline perplexity: {baseline_ppl:.4f}")

    # Per bit-width runs
    for bits in bits_list:
        print(f"\n--- TurboQuant KV cache: {bits}-bit ---")
        quant_ppl = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_text=dataset_text,
            quantize_bits=bits,
            head_dim=head_dim,
            max_length=max_length,
            stride=stride,
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
