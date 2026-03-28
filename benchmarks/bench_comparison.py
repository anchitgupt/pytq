"""Benchmark: TurboQuant vs Naive/Uniform quantization comparison.

Shows WHY the Lloyd-Max rotation approach matters by comparing three methods:
  1. NaiveQuantizer       - uniform scalar quantization on [-1, 1], no rotation
  2. UniformWithRotation  - same uniform bins but with random rotation first
  3. TurboQuant           - rotation + Lloyd-Max codebook optimised for the
                            marginal distribution of rotated unit-norm vectors
"""
import argparse
import json
import os

import torch

from pytq.quantize_mse import TurboQuantMSE
from pytq.rotation import generate_rotation_matrix


class NaiveQuantizer:
    """Uniform scalar quantizer on [-1, 1] with no rotation or normalisation."""

    def __init__(self, bits):
        n = 2 ** bits
        self.centroids = torch.linspace(-1 + 1 / n, 1 - 1 / n, n)

    def quantize_dequantize(self, x):
        x_clamped = x.clamp(-1.0, 1.0)
        dists = (x_clamped.unsqueeze(-1) - self.centroids.unsqueeze(0).unsqueeze(0)).abs()
        indices = dists.argmin(dim=-1)
        return self.centroids[indices]


class UniformWithRotation:
    """Uniform scalar quantizer applied AFTER random rotation + normalization."""

    def __init__(self, bits, dim, seed=0):
        self.dim = dim
        n = 2 ** bits
        self.centroids = torch.linspace(-1 + 1 / n, 1 - 1 / n, n)
        self.rotation = generate_rotation_matrix(dim, seed)

    def quantize_dequantize(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)

        norms = x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x_flat / norms
        y = x_unit @ self.rotation.T

        y_clamped = y.clamp(-1.0, 1.0)
        dists = (y_clamped.unsqueeze(-1) - self.centroids.unsqueeze(0).unsqueeze(0)).abs()
        indices = dists.argmin(dim=-1)
        y_hat = self.centroids[indices]

        x_hat = (y_hat @ self.rotation) * norms
        return x_hat.reshape(orig_shape)


def run_comparison(dim=128, n_vectors=10000, bits_list=None, output_dir=".", seed=42):
    if bits_list is None:
        bits_list = [1, 2, 3, 4]

    print(f"Quantization Method Comparison (dim={dim}, {n_vectors} vectors)")
    print("=" * 70)

    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn(n_vectors, dim, generator=gen)
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    results = {"config": {"dim": dim, "n_vectors": n_vectors, "bits": bits_list}, "rows": []}

    print(f"\n{'Bits':<6}{'Naive':<12}{'Uniform+Rot':<14}{'TurboQuant':<14}{'Improvement'}")
    print(f"{'----':<6}{'--------':<12}{'-----------':<14}{'----------':<14}{'-----------'}")

    for bits in bits_list:
        # Naive
        naive = NaiveQuantizer(bits=bits)
        mse_naive = ((x - naive.quantize_dequantize(x)) ** 2).mean().item()

        # Uniform + Rotation
        uni_rot = UniformWithRotation(bits=bits, dim=dim, seed=seed)
        mse_ur = ((x - uni_rot.quantize_dequantize(x)) ** 2).mean().item()

        # TurboQuant
        tq = TurboQuantMSE(dim=dim, bits=bits, seed=seed)
        qt = tq.quantize(x)
        x_hat = tq.dequantize(qt)
        mse_tq = ((x - x_hat) ** 2).mean().item()

        improvement = mse_naive / mse_tq if mse_tq > 0 else float("inf")

        results["rows"].append({
            "bits": bits,
            "mse_naive": mse_naive,
            "mse_uniform_rot": mse_ur,
            "mse_turboquant": mse_tq,
            "improvement_vs_naive": improvement,
        })

        print(f"{bits:<6}{mse_naive:<12.6f}{mse_ur:<14.6f}{mse_tq:<14.6f}{improvement:.1f}x vs naive")

    print()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare TurboQuant vs naive quantization")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-vectors", type=int, default=10000)
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--output-dir", type=str, default="results/comparison")
    args = parser.parse_args()
    run_comparison(dim=args.dim, n_vectors=args.n_vectors, bits_list=args.bits, output_dir=args.output_dir)
