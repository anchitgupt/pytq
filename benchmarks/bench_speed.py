"""Benchmark: TurboQuant speed — quantize/dequantize latency and KV cache throughput."""
import argparse
import json
import os
import time

import torch

from pytq.quantize_mse import TurboQuantMSE
from pytq.kv_cache import TurboQuantKVCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync(device: str) -> None:
    """Synchronize the device so timing is accurate."""
    if device == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass  # torch version does not expose mps.synchronize
    elif device == "cuda":
        torch.cuda.synchronize()


def _now(device: str) -> float:
    _sync(device)
    return time.perf_counter()


# ---------------------------------------------------------------------------
# 1. Quantize / dequantize latency
# ---------------------------------------------------------------------------

def bench_quantize_latency(
    device: str = "cpu",
    seq_len: int = 1024,
    dims: tuple = (64, 96, 128),
    bits_list: tuple = (2, 3, 4),
    warmup: int = 5,
    repeats: int = 20,
) -> list[dict]:
    """Measure quantize + dequantize round-trip latency.

    Returns a list of result dicts with keys:
        dim, bits, quantize_ms, dequantize_ms, roundtrip_ms
    """
    results = []

    for dim in dims:
        for bits in bits_list:
            q = TurboQuantMSE(dim=dim, bits=bits, seed=42, device=device)
            x = torch.randn(seq_len, dim, device=device)

            # --- warmup ---
            for _ in range(warmup):
                qt = q.quantize(x)
                _ = q.dequantize(qt)
            _sync(device)

            # --- timed quantize ---
            t0 = _now(device)
            for _ in range(repeats):
                qt = q.quantize(x)
            t1 = _now(device)
            quantize_ms = (t1 - t0) / repeats * 1_000

            # --- timed dequantize ---
            t0 = _now(device)
            for _ in range(repeats):
                _ = q.dequantize(qt)
            t1 = _now(device)
            dequantize_ms = (t1 - t0) / repeats * 1_000

            roundtrip_ms = quantize_ms + dequantize_ms

            results.append(
                {
                    "dim": dim,
                    "bits": bits,
                    "seq_len": seq_len,
                    "quantize_ms": round(quantize_ms, 4),
                    "dequantize_ms": round(dequantize_ms, 4),
                    "roundtrip_ms": round(roundtrip_ms, 4),
                }
            )
            print(
                f"  dim={dim:3d}  bits={bits}  "
                f"quant={quantize_ms:.3f}ms  "
                f"dequant={dequantize_ms:.3f}ms  "
                f"roundtrip={roundtrip_ms:.3f}ms"
            )

    return results


# ---------------------------------------------------------------------------
# 2. KV cache throughput
# ---------------------------------------------------------------------------

def bench_kv_cache_throughput(
    device: str = "cpu",
    head_dim: int = 128,
    n_heads: int = 8,
    bits_list: tuple = (2, 3, 4),
    seq_lengths: tuple = (128, 512, 2048),
    layer_idx: int = 0,
    warmup: int = 3,
    repeats: int = 10,
) -> list[dict]:
    """Measure KV cache update + retrieval throughput.

    Returns a list of result dicts with keys:
        bits, seq_len, update_ms, get_ms, tokens_per_sec
    """
    results = []
    batch = 1

    for bits in bits_list:
        for seq_len in seq_lengths:
            key = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
            value = torch.randn(batch, n_heads, seq_len, head_dim, device=device)

            # --- warmup ---
            for _ in range(warmup):
                cache = TurboQuantKVCache(bits=bits, head_dim=head_dim, device=device)
                cache.update(key, value, layer_idx=layer_idx)
                cache.get(layer_idx)
            _sync(device)

            # --- timed update ---
            t0 = _now(device)
            for _ in range(repeats):
                cache = TurboQuantKVCache(bits=bits, head_dim=head_dim, device=device)
                cache.update(key, value, layer_idx=layer_idx)
            t1 = _now(device)
            update_ms = (t1 - t0) / repeats * 1_000

            # Reuse the last cache for get timing
            t0 = _now(device)
            for _ in range(repeats):
                cache.get(layer_idx)
            t1 = _now(device)
            get_ms = (t1 - t0) / repeats * 1_000

            total_tokens = batch * n_heads * seq_len
            tokens_per_sec = total_tokens / ((update_ms + get_ms) / 1_000)

            results.append(
                {
                    "bits": bits,
                    "seq_len": seq_len,
                    "n_heads": n_heads,
                    "head_dim": head_dim,
                    "update_ms": round(update_ms, 4),
                    "get_ms": round(get_ms, 4),
                    "tokens_per_sec": round(tokens_per_sec, 1),
                }
            )
            print(
                f"  bits={bits}  seq_len={seq_len:5d}  "
                f"update={update_ms:.3f}ms  "
                f"get={get_ms:.3f}ms  "
                f"throughput={tokens_per_sec:.0f} tok/s"
            )

    return results


# ---------------------------------------------------------------------------
# 3. Plotting helpers
# ---------------------------------------------------------------------------

def _plot_latency(results: list[dict], output_dir: str) -> None:
    """Bar chart of quantize+dequantize roundtrip latency by dim and bits."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [skip] matplotlib not available — skipping latency plot")
        return

    dims = sorted({r["dim"] for r in results})
    bits_vals = sorted({r["bits"] for r in results})

    x = np.arange(len(dims))
    width = 0.25
    offsets = np.linspace(-(len(bits_vals) - 1) / 2, (len(bits_vals) - 1) / 2, len(bits_vals)) * width

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, bits in enumerate(bits_vals):
        latencies = [
            next(r["roundtrip_ms"] for r in results if r["dim"] == d and r["bits"] == bits)
            for d in dims
        ]
        ax.bar(x + offsets[i], latencies, width, label=f"{bits}-bit")

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Roundtrip latency (ms)")
    ax.set_title("TurboQuant quantize+dequantize latency")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.legend(title="Bits")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    path = os.path.join(output_dir, "latency_by_dim_bits.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved latency plot → {path}")


def _plot_throughput(results: list[dict], output_dir: str) -> None:
    """Line chart of KV cache throughput by seq_len and bits."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [skip] matplotlib not available — skipping throughput plot")
        return

    bits_vals = sorted({r["bits"] for r in results})
    seq_lens = sorted({r["seq_len"] for r in results})

    fig, ax = plt.subplots(figsize=(8, 5))
    for bits in bits_vals:
        tps = [
            next(r["tokens_per_sec"] for r in results if r["bits"] == bits and r["seq_len"] == s)
            for s in seq_lens
        ]
        ax.plot(seq_lens, tps, marker="o", label=f"{bits}-bit")

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Throughput (tokens / s)")
    ax.set_title("TurboQuant KV cache throughput")
    ax.legend(title="Bits")
    ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()

    path = os.path.join(output_dir, "kvcache_throughput.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved throughput plot → {path}")


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------

def run_speed_benchmark(device: str = "cpu", output_dir: str = "results") -> None:
    """Run both benchmarks, save JSON results and plots."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== TurboQuant speed benchmark  (device={device}) ===\n")

    # -- latency --
    print("--- Quantize / dequantize latency ---")
    latency_results = bench_quantize_latency(device=device)

    # -- throughput --
    print("\n--- KV cache update + retrieval throughput ---")
    throughput_results = bench_kv_cache_throughput(device=device)

    # -- save JSON --
    payload = {
        "device": device,
        "latency": latency_results,
        "throughput": throughput_results,
    }
    json_path = os.path.join(output_dir, "speed_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved JSON results → {json_path}")

    # -- save plots --
    print("\nGenerating plots …")
    _plot_latency(latency_results, output_dir)
    _plot_throughput(throughput_results, output_dir)

    print("\n=== Done ===\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure TurboQuant quantize/dequantize latency and KV cache throughput."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to run on: cpu | cuda | mps  (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to write JSON and PNG outputs (default: results/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Validate / fall back for MPS
    device = args.device
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("WARNING: MPS requested but not available — falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available — falling back to CPU")
        device = "cpu"

    run_speed_benchmark(device=device, output_dir=args.output_dir)
