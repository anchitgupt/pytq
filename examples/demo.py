#!/usr/bin/env python3
"""pytq demo — TurboQuant in action.

Run: python examples/demo.py
"""
import torch
import time

from pytq import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache, OutlierConfig, OutlierQuantizer


def banner(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def section(text):
    print(f"\n--- {text} ---\n")


def main():
    banner("pytq Demo — TurboQuant for PyTorch")
    torch.manual_seed(42)

    # ─── 1. Basic MSE Quantization ───────────────────────────────
    section("1. MSE-Optimal Quantization (TurboQuantMSE)")

    dim = 128
    n_vectors = 1000
    x = torch.randn(n_vectors, dim)

    print(f"  Input: {n_vectors} vectors, dim={dim}")
    print(f"  {'Bits':>6}  {'MSE':>10}  {'Compression':>12}  {'Latency':>10}")
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 12}  {'─' * 10}")

    for bits in [1, 2, 3, 4]:
        q = TurboQuantMSE(dim=dim, bits=bits, seed=42)

        t0 = time.perf_counter()
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        elapsed = (time.perf_counter() - t0) * 1000

        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        norm_sq = (x ** 2).sum(dim=-1).mean().item()
        mse_norm = mse / norm_sq

        fp16_bytes = n_vectors * dim * 2
        quant_bytes = n_vectors * (dim * 1 + 4)  # uint8 indices + fp32 norm
        ratio = fp16_bytes / quant_bytes

        print(f"  {bits:>4}-bit  {mse_norm:>10.4f}  {ratio:>10.1f}x  {elapsed:>8.1f}ms")

    # ─── 2. Unbiased Inner Products ──────────────────────────────
    section("2. Unbiased Inner Products (TurboQuantProd)")

    y = torch.randn(n_vectors, dim)
    ip_true = (x * y).sum(dim=-1)

    print(f"  {'Method':>20}  {'Mean Error':>12}  {'Bias':>10}")
    print(f"  {'─' * 20}  {'─' * 12}  {'─' * 10}")

    for bits in [2, 3, 4]:
        # MSE quantizer (biased)
        q_mse = TurboQuantMSE(dim=dim, bits=bits, seed=42)
        qt_mse = q_mse.quantize(x)
        x_hat_mse = q_mse.dequantize(qt_mse)
        ip_mse = (x_hat_mse * y).sum(dim=-1)
        bias_mse = (ip_mse - ip_true).mean().item()

        # Prod quantizer (unbiased)
        q_prod = TurboQuantProd(dim=dim, bits=bits, seed=42)
        qt_prod = q_prod.quantize(x)
        x_hat_prod = q_prod.dequantize(qt_prod)
        ip_prod = (x_hat_prod * y).sum(dim=-1)
        bias_prod = (ip_prod - ip_true).mean().item()

        print(f"  MSE  {bits}-bit           {abs(bias_mse):>12.4f}  {'biased':>10}")
        print(f"  Prod {bits}-bit           {abs(bias_prod):>12.4f}  {'unbiased':>10}")

    # ─── 3. Outlier Channel Handling ─────────────────────────────
    section("3. Outlier Channels (fractional bit-widths)")

    cfg = OutlierConfig(outlier_fraction=0.25, outlier_bits=3, normal_bits=2)
    print(f"  Config: 25% outliers at 3-bit + 75% normal at 2-bit")
    print(f"  Effective bit-width: {cfg.effective_bits:.2f}")

    oq = OutlierQuantizer(dim=dim, config=cfg, seed=42)
    qt_out = oq.quantize(x)
    x_hat_out = oq.dequantize(qt_out)
    mse_out = ((x - x_hat_out) ** 2).sum(dim=-1).mean().item()

    # Compare with pure 2-bit
    q2 = TurboQuantMSE(dim=dim, bits=2, seed=42)
    qt2 = q2.quantize(x)
    x_hat2 = q2.dequantize(qt2)
    mse_2bit = ((x - x_hat2) ** 2).sum(dim=-1).mean().item()

    norm_sq = (x ** 2).sum(dim=-1).mean().item()
    print(f"\n  {'Method':>20}  {'MSE (normalized)':>16}")
    print(f"  {'─' * 20}  {'─' * 16}")
    print(f"  {'Pure 2-bit':>20}  {mse_2bit / norm_sq:>16.4f}")
    print(f"  {'2.25-bit (outlier)':>20}  {mse_out / norm_sq:>16.4f}")
    improvement = mse_2bit / max(mse_out, 1e-10)
    print(f"\n  Outlier handling gives {improvement:.1f}x lower distortion than pure 2-bit")

    # ─── 4. KV Cache Demo ────────────────────────────────────────
    section("4. KV Cache Compression (TurboQuantKVCache)")

    batch, n_heads, seq_len, head_dim = 1, 8, 512, 128
    keys = torch.randn(batch, n_heads, seq_len, head_dim)
    values = torch.randn(batch, n_heads, seq_len, head_dim)

    fp16_key_bytes = keys.nelement() * 2
    print(f"  Simulated attention: batch={batch}, heads={n_heads}, seq={seq_len}, head_dim={head_dim}")
    print(f"  FP16 key size: {fp16_key_bytes / 1024:.1f} KB")

    print(f"\n  {'Bits':>6}  {'Compressed':>12}  {'Savings':>10}  {'Cos Sim':>10}")
    print(f"  {'─' * 6}  {'─' * 12}  {'─' * 10}  {'─' * 10}")

    for bits in [2, 3, 4]:
        cache = TurboQuantKVCache(bits=bits, head_dim=head_dim)
        cache.update(keys, values, layer_idx=0)
        k_out, v_out = cache.get(layer_idx=0)

        compressed_bytes = cache.key_memory_bytes(layer_idx=0)
        savings = fp16_key_bytes / compressed_bytes

        cos_sim = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, head_dim), k_out.reshape(-1, head_dim), dim=-1
        ).mean().item()

        print(f"  {bits:>4}-bit  {compressed_bytes / 1024:>10.1f} KB  {savings:>8.1f}x  {cos_sim:>10.4f}")

    # ─── Summary ─────────────────────────────────────────────────
    banner("Summary")
    print("  TurboQuant provides:")
    print("    - Near-optimal MSE distortion (within 2.72x of info-theoretic bound)")
    print("    - Unbiased inner product estimation (TurboQuantProd)")
    print("    - Fractional bit-widths via outlier channel handling")
    print("    - Drop-in KV cache compression for HuggingFace models")
    print("    - Zero training, zero calibration — works instantly on any vector")
    print()
    print("  Paper: https://arxiv.org/abs/2504.19874")
    print("  GitHub: https://github.com/anchitgupta/pytq")
    print()


if __name__ == "__main__":
    main()
