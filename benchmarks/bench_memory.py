"""Benchmark: KV cache memory profiling across models and bit-widths.

Estimates KV cache memory analytically (no model loading required).
For quantized keys, uses the same packed-index + fp32-norm formula as
TurboQuantKVCache.key_memory_bytes(); values are always stored in fp16.
"""
import argparse
import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "llama-3.2-1b": {"n_layers": 16, "n_kv_heads": 8, "head_dim": 64,  "weight_gb": 2.4},
    "phi-3-mini":   {"n_layers": 32, "n_kv_heads": 8, "head_dim": 96,  "weight_gb": 7.6},
    "mistral-7b":   {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128, "weight_gb": 14.4},
    "llama-3.1-8b": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128, "weight_gb": 16.0},
}

BITS_LIST     = [1, 2, 2.5, 3, 3.5, 4, 16]
SEQ_LENGTHS   = [512, 1024, 2048, 4096, 8192, 16384, 32768]
AVAILABLE_RAM_GB = 10.0

# ---------------------------------------------------------------------------
# Core estimation functions
# ---------------------------------------------------------------------------

def estimate_kv_memory_bytes(
    n_layers: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
    bits: float,
) -> int:
    """Return estimated KV cache memory in bytes for the given configuration.

    For fp16 (bits == 16):
        Both keys and values are stored as fp16 (2 bytes per element).
        memory = n_layers * n_kv_heads * seq_len * head_dim * 2   (keys)
               + n_layers * n_kv_heads * seq_len * head_dim * 2   (values)

    For quantized keys (bits < 16):
        Keys  : packed indices  ceil(bits * head_dim / 8) bytes/token
              + fp32 norm       4 bytes/token
        Values: fp16            head_dim * 2 bytes/token
    """
    if bits == 16:
        # Keys + values both fp16
        per_token = head_dim * 2 + head_dim * 2  # keys + values
        return n_layers * n_kv_heads * seq_len * per_token
    else:
        key_bytes_per_token   = math.ceil(bits * head_dim / 8) + 4  # packed + norm
        value_bytes_per_token = head_dim * 2                         # fp16
        per_token = key_bytes_per_token + value_bytes_per_token
        return n_layers * n_kv_heads * seq_len * per_token


def find_max_context(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bits: float,
    available_ram_gb: float,
    lo: int = 1,
    hi: int = 1_000_000,
) -> int:
    """Binary search for the maximum context length that fits within RAM budget.

    Returns the largest seq_len such that estimate_kv_memory_bytes <= available_ram_gb.
    Returns 0 if even seq_len=lo exceeds the budget.
    """
    budget_bytes = int(available_ram_gb * (1024 ** 3))

    if estimate_kv_memory_bytes(n_layers, n_kv_heads, lo, head_dim, bits) > budget_bytes:
        return 0

    while lo < hi:
        mid = (lo + hi + 1) // 2
        mem = estimate_kv_memory_bytes(n_layers, n_kv_heads, mid, head_dim, bits)
        if mem <= budget_bytes:
            lo = mid
        else:
            hi = mid - 1

    return lo


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_memory_benchmark() -> dict:
    """Iterate over all (model, bits, seq_len) combinations and collect results.

    Returns a nested dict:
        results[model_name][bits][seq_len] = memory_bytes
    Also computes max_context per (model, bits) and stores in:
        results[model_name]["max_context"][bits] = max_seq_len
    """
    results: dict = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        results[model_name] = {"max_context": {}}
        n_layers   = cfg["n_layers"]
        n_kv_heads = cfg["n_kv_heads"]
        head_dim   = cfg["head_dim"]

        for bits in BITS_LIST:
            results[model_name][bits] = {}

            for seq_len in SEQ_LENGTHS:
                mem = estimate_kv_memory_bytes(n_layers, n_kv_heads, seq_len, head_dim, bits)
                results[model_name][bits][seq_len] = mem

            max_ctx = find_max_context(
                n_layers, n_kv_heads, head_dim, bits, AVAILABLE_RAM_GB
            )
            results[model_name]["max_context"][bits] = max_ctx

            mem_at_4k = results[model_name][bits].get(4096, None)
            mem_str = f"{mem_at_4k / 1e9:.3f} GB" if mem_at_4k is not None else "n/a"
            print(
                f"  {model_name:>16s}  bits={bits:>4.1f}  "
                f"mem@4096={mem_str:>12s}  max_ctx={max_ctx:>8,d}"
            )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_memory_vs_seqlen(results: dict, output_dir: Path) -> None:
    """One figure per model: memory (GB) vs sequence length for each bit-width."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_CONFIGS:
        fig, ax = plt.subplots(figsize=(8, 5))

        for bits in BITS_LIST:
            x = SEQ_LENGTHS
            y = [results[model_name][bits][s] / 1e9 for s in x]
            label = "fp16" if bits == 16 else f"{bits}-bit"
            linestyle = "--" if bits == 16 else "-"
            ax.plot(x, y, marker="o", markersize=4, label=label, linestyle=linestyle)

        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_ylabel("KV Cache Memory (GB)")
        ax.set_title(f"KV Cache Memory — {model_name}")
        ax.legend(title="Precision", loc="upper left")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        fname = output_dir / f"memory_vs_seqlen_{model_name.replace(' ', '_')}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


def _plot_max_context(results: dict, output_dir: Path) -> None:
    """Bar chart: max context length (within AVAILABLE_RAM_GB) per model and bit-width."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed — skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(MODEL_CONFIGS.keys())
    n_models    = len(model_names)
    n_bits      = len(BITS_LIST)
    x           = np.arange(n_models)
    width       = 0.8 / n_bits

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_bits))  # type: ignore[attr-defined]

    for i, bits in enumerate(BITS_LIST):
        heights = [results[m]["max_context"][bits] for m in model_names]
        offset  = (i - n_bits / 2 + 0.5) * width
        label   = "fp16" if bits == 16 else f"{bits}-bit"
        ax.bar(x + offset, heights, width=width * 0.9, label=label, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Max Context Length (tokens)")
    ax.set_title(
        f"Max Context Length within {AVAILABLE_RAM_GB:.0f} GB RAM Budget"
    )
    ax.legend(title="Precision", ncol=2, loc="upper right")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}")  # type: ignore[attr-defined]
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    fname = output_dir / "max_context_bar.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile KV cache memory across models and bit-widths."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/memory"),
        help="Directory to save plots (default: benchmark_results/memory).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 65)
    print(" KV Cache Memory Benchmark")
    print(f" RAM budget : {AVAILABLE_RAM_GB:.1f} GB")
    print(f" Models     : {', '.join(MODEL_CONFIGS)}")
    print(f" Bits       : {BITS_LIST}")
    print(f" Seq lengths: {SEQ_LENGTHS}")
    print("=" * 65)

    results = run_memory_benchmark()

    print("\nGenerating plots …")
    _plot_memory_vs_seqlen(results, args.output_dir)
    _plot_max_context(results, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
