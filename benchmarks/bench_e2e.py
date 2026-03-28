"""Benchmark: End-to-end — Will it fit on a 16 GB MacBook Air?

Estimates whether TurboQuant KV cache fits within a 16 GB memory budget for
common models at various bit-widths, and computes compression ratios vs fp16.
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional

from benchmarks.bench_memory import (
    MODEL_CONFIGS,
    estimate_kv_memory_bytes,
    find_max_context,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_RAM_GB = 16.0          # MacBook Air unified memory
PRACTICAL_CTX = 8192          # "practical" context length cap
DEFAULT_BITS = [2, 3, 4, 16]

# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_e2e_benchmark(
    models: Optional[List] = None,
    bits: Optional[List] = None,
    output_dir: str = "benchmark_results/e2e",
) -> dict:
    """Run end-to-end benchmark: fit check on 16 GB MacBook Air.

    For each (model, bit-width) pair:
      - Finds the maximum context length that fits within 16 GB.
      - Estimates memory at a practical context = min(max_ctx, 8192).
      - Computes compression ratio vs fp16.
      - Reports OK / EXCEEDS 16 GB status.

    Args:
        models:     List of model names to evaluate (defaults to all in MODEL_CONFIGS).
        bits:       List of bit-widths to evaluate (default: [2, 3, 4, 16]).
        output_dir: Directory where results.json and summary.md are saved.

    Returns:
        Nested dict: results[model_name][bits] = {max_ctx, mem_gb, ratio, fits}.
    """
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    if bits is None:
        bits = DEFAULT_BITS

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    budget_bytes = int(TARGET_RAM_GB * (1024 ** 3))
    results: dict = {}

    print("=" * 70)
    print(f"  End-to-End Benchmark — Will it fit on {TARGET_RAM_GB:.0f} GB MacBook Air?")
    print(f"  Models : {models}")
    print(f"  Bits   : {bits}")
    print("=" * 70)

    for model_name in models:
        cfg = MODEL_CONFIGS.get(model_name)
        if cfg is None:
            print(f"  [WARN] Unknown model '{model_name}' — skipping.")
            continue

        results[model_name] = {}
        n_layers   = cfg["n_layers"]
        n_kv_heads = cfg["n_kv_heads"]
        head_dim   = cfg["head_dim"]
        weight_gb  = cfg.get("weight_gb", 0.0)

        print(f"\n  Model: {model_name}  (weights ~{weight_gb:.1f} GB)")
        print(f"  {'Bits':>5}  {'MaxCtx':>10}  {'CtxUsed':>10}  {'KV Mem':>10}  {'Ratio':>8}  Status")
        print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  ------")

        # Pre-compute fp16 memory at practical context for ratio reference
        fp16_practical_mem: Optional[int] = None

        for b in bits:
            max_ctx = find_max_context(
                n_layers, n_kv_heads, head_dim, b,
                available_ram_gb=TARGET_RAM_GB,
            )
            practical_ctx = min(max_ctx, PRACTICAL_CTX) if max_ctx > 0 else 0
            mem_bytes = (
                estimate_kv_memory_bytes(n_layers, n_kv_heads, practical_ctx, head_dim, b)
                if practical_ctx > 0
                else 0
            )
            mem_gb = mem_bytes / (1024 ** 3)
            fits = mem_bytes <= budget_bytes

            # Record fp16 reference
            if b == 16:
                fp16_practical_mem = mem_bytes if mem_bytes > 0 else None

            # Compression ratio vs fp16 (computed after fp16 entry or lazily below)
            ratio = "—"  # type: ignore[assignment]  # str or float

            results[model_name][b] = {
                "max_ctx": max_ctx,
                "practical_ctx": practical_ctx,
                "mem_bytes": mem_bytes,
                "mem_gb": round(mem_gb, 4),
                "fits_16gb": fits,
                "compression_ratio": None,  # filled below
            }

        # Fill compression ratios now that fp16 entry exists
        fp16_entry = results[model_name].get(16)
        fp16_mem = fp16_entry["mem_bytes"] if fp16_entry else None

        for b in bits:
            entry = results[model_name][b]
            if fp16_mem and fp16_mem > 0 and b != 16:
                ratio_val = fp16_mem / entry["mem_bytes"] if entry["mem_bytes"] > 0 else 0.0
                entry["compression_ratio"] = round(ratio_val, 2)
                ratio_str = f"{ratio_val:.2f}x"
            elif b == 16:
                entry["compression_ratio"] = 1.0
                ratio_str = "1.00x"
            else:
                ratio_str = "  —  "

            status = "OK" if entry["fits_16gb"] else "EXCEEDS 16GB"
            bits_label = "fp16" if b == 16 else f"{b}-bit"
            print(
                f"  {bits_label:>5}  {entry['max_ctx']:>10,}  "
                f"{entry['practical_ctx']:>10,}  "
                f"{entry['mem_gb']:>9.3f}G  "
                f"{ratio_str:>8}  {status}"
            )

    # Save JSON
    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    # Save markdown summary
    md_path = out_path / "summary.md"
    _write_markdown_summary(results, bits, md_path)
    print(f"  Summary  saved to {md_path}")

    return results


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _write_markdown_summary(results: dict, bits: list, md_path: Path) -> None:
    """Write a markdown table summarising the e2e benchmark results."""
    lines = [
        "# End-to-End Benchmark — 16 GB MacBook Air",
        "",
        f"Memory budget: **{TARGET_RAM_GB:.0f} GB**  |  "
        f"Practical context cap: **{PRACTICAL_CTX:,} tokens**",
        "",
    ]

    for model_name, model_data in results.items():
        lines += [f"## {model_name}", ""]
        # Header
        header_bits = [str(b) if b != 16 else "fp16" for b in bits]
        lines.append(
            "| Metric | " + " | ".join(header_bits) + " |"
        )
        lines.append(
            "| --- | " + " | ".join(["---"] * len(bits)) + " |"
        )

        # Max context row
        max_ctx_vals = [
            f"{model_data[b]['max_ctx']:,}" if b in model_data else "—"
            for b in bits
        ]
        lines.append("| Max context (tokens) | " + " | ".join(max_ctx_vals) + " |")

        # KV memory row
        mem_vals = [
            f"{model_data[b]['mem_gb']:.3f} GB" if b in model_data else "—"
            for b in bits
        ]
        lines.append(f"| KV mem @ min(max_ctx,{PRACTICAL_CTX:,}) | " + " | ".join(mem_vals) + " |")

        # Compression ratio row
        ratio_vals = []
        for b in bits:
            if b not in model_data:
                ratio_vals.append("—")
            elif b == 16:
                ratio_vals.append("1.00×")
            else:
                r = model_data[b].get("compression_ratio")
                ratio_vals.append(f"{r:.2f}×" if r else "—")
        lines.append("| Compression vs fp16 | " + " | ".join(ratio_vals) + " |")

        # Fits row
        fits_vals = [
            ("**OK**" if model_data[b]["fits_16gb"] else "**EXCEEDS**")
            if b in model_data else "—"
            for b in bits
        ]
        lines.append("| Fits 16 GB? | " + " | ".join(fits_vals) + " |")

        lines.append("")

    md_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark: Will TurboQuant KV cache fit on a 16 GB MacBook Air?"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to evaluate (default: all configured models).",
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        type=float,
        default=DEFAULT_BITS,
        help=f"Bit-widths to evaluate (default: {DEFAULT_BITS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/e2e",
        help="Directory to save results.json and summary.md (default: benchmark_results/e2e).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_e2e_benchmark(
        models=args.models,
        bits=args.bits,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
