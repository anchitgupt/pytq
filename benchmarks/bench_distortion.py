"""Benchmark: Theoretical Distortion Validation for TurboQuant.

Reproduces Figures 1-3 from the TurboQuant paper by comparing measured
MSE distortion and IP error variance against theoretical bounds.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch

from pytq.quantize_mse import TurboQuantMSE
from pytq.quantize_prod import TurboQuantProd
from pytq.utils import mse_distortion, ip_distortion, mse_upper_bound, mse_lower_bound, ip_upper_bound


DIMS = [64, 96, 128, 768, 1536, 3072]
BITS = [1, 2, 3, 4]


def _generate_unit_vectors(n: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Generate n random unit-norm vectors of dimension dim."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn(n, dim, generator=gen)
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return x / norms


def run_distortion_benchmark(
    n_vectors: int = 500,
    n_queries: int = 200,
    output_dir: str = "results/distortion",
    seed: int = 42,
) -> dict:
    """Run the distortion validation benchmark.

    Measures MSE distortion and IP error variance for TurboQuantMSE and
    TurboQuantProd across multiple dimensions and bit widths, then compares
    against theoretical upper/lower bounds from the paper.

    Args:
        n_vectors: Number of database vectors to quantize.
        n_queries:  Number of query vectors for IP distortion.
        output_dir: Directory to save JSON results and PNG plots.
        seed:       Random seed for reproducibility.

    Returns:
        Dictionary containing all measured values and theoretical bounds.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "n_vectors": n_vectors,
            "n_queries": n_queries,
            "dims": DIMS,
            "bits": BITS,
        },
        "mse": {},
        "ip": {},
        "bounds": {},
    }

    # Pre-compute theoretical bounds (bit-level, independent of dim for MSE)
    for b in BITS:
        results["bounds"][str(b)] = {
            "mse_upper": mse_upper_bound(b),
            "mse_lower": mse_lower_bound(b),
        }

    print(f"Running distortion benchmark: {n_vectors} vectors, {n_queries} queries")
    print(f"Dimensions: {DIMS}")
    print(f"Bits: {BITS}")
    print()

    for dim in DIMS:
        print(f"  dim={dim}")
        x = _generate_unit_vectors(n_vectors, dim, seed=seed)
        q = _generate_unit_vectors(n_queries, dim, seed=seed + 1)

        results["mse"][str(dim)] = {}
        results["ip"][str(dim)] = {}

        # Store per-dim IP bounds (they depend on dim)
        for b in BITS:
            results["bounds"][str(b)][f"ip_upper_dim{dim}"] = ip_upper_bound(b, dim)

        for b in BITS:
            entry_mse: dict = {"bits": b, "dim": dim}
            entry_ip: dict = {"bits": b, "dim": dim}

            # --- TurboQuantMSE ---
            try:
                quantizer_mse = TurboQuantMSE(dim=dim, bits=b, seed=seed)
                qt_mse = quantizer_mse.quantize(x)
                x_hat_mse = quantizer_mse.dequantize(qt_mse)

                mse_val = mse_distortion(x, x_hat_mse).item()
                entry_mse["mse_value"] = mse_val

                # IP distortion: broadcast query against all vectors
                # x_hat_mse: (n_vectors, dim), q: (n_queries, dim)
                # expand to (n_queries, n_vectors, dim)
                x_exp = x.unsqueeze(0).expand(n_queries, -1, -1)
                x_hat_exp = x_hat_mse.unsqueeze(0).expand(n_queries, -1, -1)
                q_exp = q.unsqueeze(1).expand(-1, n_vectors, -1)

                var_mse, bias_mse = ip_distortion(x_exp, x_hat_exp, q_exp)
                entry_ip["variance_mse"] = var_mse.item()
                entry_ip["bias_mse"] = bias_mse.item()

            except Exception as exc:
                entry_mse["mse_value"] = None
                entry_mse["error_mse"] = str(exc)
                entry_ip["variance_mse"] = None
                entry_ip["bias_mse"] = None

            # --- TurboQuantProd (requires >= 2 bits) ---
            if b < 2:
                entry_mse["mse_prod"] = "N/A"
                entry_ip["variance_prod"] = "N/A"
                entry_ip["bias_prod"] = "N/A"
            else:
                try:
                    quantizer_prod = TurboQuantProd(dim=dim, bits=b, seed=seed)
                    qt_prod = quantizer_prod.quantize(x)
                    x_hat_prod = quantizer_prod.dequantize(qt_prod)

                    mse_val_prod = mse_distortion(x, x_hat_prod).item()
                    entry_mse["mse_prod"] = mse_val_prod

                    x_hat_prod_exp = x_hat_prod.unsqueeze(0).expand(n_queries, -1, -1)
                    var_prod, bias_prod = ip_distortion(x_exp, x_hat_prod_exp, q_exp)
                    entry_ip["variance_prod"] = var_prod.item()
                    entry_ip["bias_prod"] = bias_prod.item()

                except Exception as exc:
                    entry_mse["mse_prod"] = None
                    entry_mse["error_prod"] = str(exc)
                    entry_ip["variance_prod"] = None
                    entry_ip["bias_prod"] = None

            # Theoretical bounds for easy lookup
            entry_mse["mse_upper"] = mse_upper_bound(b)
            entry_mse["mse_lower"] = mse_lower_bound(b)
            entry_ip["ip_upper"] = ip_upper_bound(b, dim)

            results["mse"][str(dim)][str(b)] = entry_mse
            results["ip"][str(dim)][str(b)] = entry_ip

            print(
                f"    b={b}: MSE_mse={entry_mse.get('mse_value', 'err'):.4e}  "
                f"MSE_prod={entry_mse.get('mse_prod', 'err')}  "
                f"upper={entry_mse['mse_upper']:.4e}  lower={entry_mse['mse_lower']:.4e}"
            )

    # Save JSON
    json_path = out_path / "distortion_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        _plot_mse_vs_bits(results, out_path)
        _plot_ip_error_vs_bits(results, out_path)
    except ImportError:
        print("matplotlib not available — skipping plots")

    return results


def _plot_mse_vs_bits(results: dict, out_path: Path) -> None:
    """Plot MSE distortion vs bits for each dimension against theoretical bounds."""
    import matplotlib.pyplot as plt

    bits_arr = BITS
    upper = [mse_upper_bound(b) for b in bits_arr]
    lower = [mse_lower_bound(b) for b in bits_arr]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax_idx, dim in enumerate(DIMS):
        ax = axes[ax_idx]
        dim_data = results["mse"].get(str(dim), {})

        mse_vals: list = []
        prod_vals: list = []
        valid_bits: list = []

        for b in bits_arr:
            entry = dim_data.get(str(b), {})
            mse_val = entry.get("mse_value")
            prod_val = entry.get("mse_prod")
            if mse_val is not None:
                mse_vals.append((b, mse_val))
            if prod_val not in (None, "N/A") and isinstance(prod_val, float):
                prod_vals.append((b, prod_val))

        ax.plot(bits_arr, upper, "r--", label="Upper bound", linewidth=1.5)
        ax.plot(bits_arr, lower, "g--", label="Lower bound", linewidth=1.5)

        if mse_vals:
            bx, vy = zip(*mse_vals)
            ax.plot(bx, vy, "b-o", label="TurboQuant_mse", markersize=5)

        if prod_vals:
            bx, vy = zip(*prod_vals)
            ax.plot(bx, vy, "m-s", label="TurboQuant_prod", markersize=5)

        ax.set_yscale("log")
        ax.set_xlabel("Bits per dimension")
        ax.set_ylabel("MSE distortion")
        ax.set_title(f"dim={dim}")
        ax.set_xticks(bits_arr)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("MSE Distortion vs Bits — TurboQuant vs Theoretical Bounds", fontsize=13)
    fig.tight_layout()

    plot_path = out_path / "mse_vs_bits.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"MSE plot saved to {plot_path}")


def _plot_ip_error_vs_bits(results: dict, out_path: Path) -> None:
    """Plot IP error variance vs bits for each dimension against theoretical bounds."""
    import matplotlib.pyplot as plt

    bits_arr = BITS

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax_idx, dim in enumerate(DIMS):
        ax = axes[ax_idx]
        dim_data = results["ip"].get(str(dim), {})

        ip_upper = [ip_upper_bound(b, dim) for b in bits_arr]

        mse_vars: list = []
        prod_vars: list = []

        for b in bits_arr:
            entry = dim_data.get(str(b), {})
            var_mse = entry.get("variance_mse")
            var_prod = entry.get("variance_prod")
            if var_mse is not None:
                mse_vars.append((b, var_mse))
            if var_prod not in (None, "N/A") and isinstance(var_prod, float):
                prod_vars.append((b, var_prod))

        ax.plot(bits_arr, ip_upper, "r--", label="IP upper bound", linewidth=1.5)

        if mse_vars:
            bx, vy = zip(*mse_vars)
            ax.plot(bx, vy, "b-o", label="TurboQuant_mse", markersize=5)

        if prod_vars:
            bx, vy = zip(*prod_vars)
            ax.plot(bx, vy, "m-s", label="TurboQuant_prod", markersize=5)

        ax.set_yscale("log")
        ax.set_xlabel("Bits per dimension")
        ax.set_ylabel("IP error variance")
        ax.set_title(f"dim={dim}")
        ax.set_xticks(bits_arr)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("IP Error Variance vs Bits — TurboQuant vs Theoretical Bounds", fontsize=13)
    fig.tight_layout()

    plot_path = out_path / "ip_error_vs_bits.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"IP error plot saved to {plot_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TurboQuant distortion against theoretical bounds."
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=500,
        help="Number of database vectors to quantize (default: 500)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=200,
        help="Number of query vectors for IP distortion (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/distortion",
        help="Directory to save results (default: results/distortion)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_distortion_benchmark(
        n_vectors=args.n_vectors,
        n_queries=args.n_queries,
        output_dir=args.output_dir,
        seed=args.seed,
    )
