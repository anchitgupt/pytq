# pytq: TurboQuant Implementation & Benchmarking Suite

## Overview

A Python library implementing the TurboQuant algorithm (Zandieh et al., ICLR 2026) for data-oblivious vector quantization, with extensive benchmarking focused on KV cache compression for LLM inference on a 16GB MacBook Air.

**Paper:** "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv: 2504.19874)

**Goal:** Answer the question: "At X bits with TurboQuant, can model Y fit in my 16GB MacBook Air while maintaining acceptable quality and speed?"

## Target Hardware

- MacBook Air, 16GB RAM (~10-12GB available after system overhead)
- Apple Silicon (MPS backend for PyTorch)
- CPU fallback for unsupported MPS operations

## Project Structure

```
pytq/
├── pytq/                     # Core package
│   ├── __init__.py
│   ├── codebook.py           # Lloyd-Max codebook construction for Beta distribution
│   ├── rotation.py           # Random orthogonal rotation via QR decomposition
│   ├── quantize_mse.py       # TurboQuant_mse: rotate -> scalar quantize -> dequantize
│   ├── quantize_prod.py      # TurboQuant_prod: MSE quantize + QJL residual correction
│   ├── outlier.py            # Outlier channel detection + split bit allocation
│   ├── kv_cache.py           # HuggingFace KV cache wrapper
│   └── utils.py              # Distortion metrics, memory measurement helpers
├── benchmarks/
│   ├── bench_distortion.py   # Theoretical validation: MSE & IP error vs bounds
│   ├── bench_memory.py       # Peak RAM, KV cache size at various seq lengths
│   ├── bench_quality.py      # Perplexity + downstream task accuracy
│   ├── bench_speed.py        # Tokens/sec, attention latency on MPS/CPU
│   ├── bench_e2e.py          # End-to-end: max context on 16GB at each bit-width
│   └── run_all.py            # Run full benchmark suite
├── tests/
│   ├── test_codebook.py
│   ├── test_quantize.py
│   └── test_kv_cache.py
├── results/                  # Benchmark output (JSON + PNG + markdown)
├── pyproject.toml
└── README.md
```

## Core Algorithm

### TurboQuant_mse (MSE-Optimal)

1. **Codebook construction** (offline, once per dimension `d` and bit-width `b`):
   - The rotated vector coordinates follow a Beta distribution: `f_X(x) = [Gamma(d/2)] / [sqrt(pi) * Gamma((d-1)/2)] * (1 - x^2)^{(d-3)/2}` for `x in [-1, 1]`
   - Run Lloyd-Max algorithm on a dense grid (~50K points) for ~300 iterations to find optimal `2^b` centroids
   - Store codebooks as precomputed lookup tables

2. **Normalize**:
   - Compute and store the L2 norm: `norm = ||x||`
   - Normalize to unit sphere: `x_hat = x / norm`

3. **Quantize**:
   - Generate random orthogonal matrix `Pi` via QR decomposition of a Gaussian matrix (shared per layer, seeded deterministically)
   - Rotate: `y = Pi * x_hat` (each coordinate now follows the known Beta distribution)
   - For each coordinate `j`: store `idx_j = argmin_k |y_j - c_k|` as a `b`-bit integer

4. **Dequantize**:
   - Reconstruct: `y_tilde_j = c_{idx_j}` for each `j`
   - Inverse rotate: `x_hat_tilde = Pi^T * y_tilde`
   - Rescale: `x_tilde = norm * x_hat_tilde`

**Theoretical guarantee (Theorem 1):** `D_mse <= (sqrt(3)*pi/2) * (1/4^b)` (~2.72x the information-theoretic lower bound of `1/4^b`)

### TurboQuant_prod (Inner-Product-Optimal)

Extends TurboQuant_mse with a residual correction for unbiased inner products:

5. Apply TurboQuant_mse with bit-width `(b-1)` to get MSE-optimal quantization
6. Compute residual: `r = x - Q_mse^{-1}(Q_mse(x))`
7. Store residual norm: `||r||` (fp32 scalar, needed for QJL dequantization)
8. Apply QJL to the residual:
   - `Q_qjl(r) = sign(S * r)` where `S` is a `d x d` random Gaussian matrix (shared per layer, seeded deterministically — same seed reproduces `S` without storing it)
   - `Q_qjl^{-1}(z) = (sqrt(pi/2) / d) * ||r|| * S^T * z`
9. Final inner product estimate = MSE estimate + residual correction

**Theoretical guarantee (Theorem 2):**
- Unbiased: `E[<y, x_tilde>] = <y, x>`
- Distortion: `D_prod <= (sqrt(3)*pi^2 * ||y||^2 / d) * (1/4^b)`

### Outlier Channel Handling

Channels are split into outlier and non-outlier sets based on magnitude. Higher bit precision is allocated to outlier channels:

- **2.5-bit config** (head_dim=128): 32 outlier channels at 3 bits + 96 channels at 2 bits = `(32*3 + 96*2)/128 = 2.5`
- **3.5-bit config** (head_dim=128): analogous split yielding 3.5 effective bits
- For models with different head dimensions (e.g., 64 or 96), the outlier/non-outlier split ratios are scaled proportionally to maintain the same effective bit-width (e.g., for head_dim=64: 16 outliers at 3 bits + 48 at 2 bits = 2.5)

## KV Cache Integration

### TurboQuantKVCache

A custom class wrapping HuggingFace's `DynamicCache`:

- **On `update(key, value, layer_idx)`**: Quantize incoming key tensor using TurboQuant. Values remain in fp16 by default (configurable).
- **Compressed storage format per token per head**:
  - `indices`: uint8 tensor of quantization indices (b bits packed, ceil(b*head_dim/8) bytes)
  - `norm`: fp32 scalar (L2 norm of original vector, for rescaling after dequant)
  - `residual_norm`: fp32 scalar (only for TurboQuant_prod — `||r||` for QJL dequantization)
  - Rotation matrix `Pi`: not stored per-token; shared per layer, generated deterministically from a stored seed
  - QJL matrix `S`: not stored; regenerated from a per-layer seed
- **On key retrieval**: Dequantize on-the-fly for attention computation.
- **Design choice — keys-only by default**: This is a project design decision (not paper-prescribed). The paper's KV cache experiments focus on key quantization because `Q @ K^T` is the memory bottleneck during autoregressive decoding. Value quantization can be added as a configuration option in future work.

### Bit-width Configurations

Benchmarked at: 1-bit, 2-bit, 2.5-bit, 3-bit, 3.5-bit, 4-bit, plus fp16 baseline.

### Target Models

| Model | Params | fp16 Size | Fits in 16GB at fp16? |
|-------|--------|-----------|----------------------|
| Llama-3.2-1B | 1.2B | ~2.4GB | Yes, comfortably |
| Phi-3-mini | 3.8B | ~7.6GB | Yes, tight with long context |
| Mistral-7B-v0.3 | 7.2B | ~14.4GB | Barely |
| Llama-3.1-8B | 8B | ~16GB | No, needs compression |

For 7B/8B models: load with 4-bit weight quantization via `torchao` (Apple Silicon compatible) as baseline, then layer TurboQuant KV cache compression on top.

## Benchmark Suite

### Benchmark 1: Theoretical Distortion Validation (`bench_distortion.py`)

- Generate/load 100K random vectors in dimensions 64, 96, 128, 768, 1536, 3072 (64/96/128 match target model head dims)
- Quantize with TurboQuant_mse and TurboQuant_prod at 1, 2, 3, 4 bits
- Measure: MSE, inner-product error (mean + variance), bias
- Plot against theoretical upper bound (`sqrt(3)*pi/2 * 1/4^b`) and lower bound (`1/4^b`)
- Verify TurboQuant_prod is unbiased; show TurboQuant_mse bias shrinks with higher bits
- Output: matplotlib charts reproducing Figures 1-3 from the paper

### Benchmark 2: Memory Profiling (`bench_memory.py`)

- For each model x bit-width combination:
  - Measure KV cache memory at sequence lengths: 512, 1K, 2K, 4K, 8K, 16K, 32K
  - Measure peak process RSS via `psutil`
  - Calculate max context length fitting in ~10GB free RAM
- Output: table + line chart (memory vs sequence length per config)

### Benchmark 3: Quality (`bench_quality.py`)

- Perplexity on WikiText-2 (standard, fast to compute)
- Subset of MMLU (5-shot, ~1K questions across 5 categories) for downstream accuracy
- For each model x bit-width: report perplexity delta and accuracy delta vs fp16 baseline
- Output: comparison table

### Benchmark 4: Speed (`bench_speed.py`)

- Prefill speed: tokens/sec for prompt lengths 128, 512, 2048
- Decode speed: tokens/sec for generation (100 generated tokens)
- Attention latency: single attention forward pass, quantized vs fp16 KV cache
- Test on both CPU and MPS backends
- Output: bar charts comparing speed across bit-widths and backends

### Benchmark 5: End-to-End "Will It Fit?" (`bench_e2e.py`)

- For each model, find max context length at each bit-width on 16GB
- Run a generation task at that max length, report quality and speed
- Summary table:

| Model | Bit-width | Max Context | Peak RAM | Tok/s | Perplexity |
|-------|-----------|-------------|----------|-------|------------|
| Llama-3.2-1B | fp16 | 32K | 8.2GB | 45 | 8.1 |
| Llama-3.2-1B | 2.5-bit | 96K+ | 7.1GB | 52 | 8.3 |
| Llama-3.1-8B | 4-bit w + fp16 KV | 4K | 12GB | 12 | 7.2 |
| Llama-3.1-8B | 4-bit w + 2.5-bit KV | 16K | 10GB | 15 | 7.4 |

*(numbers are illustrative)*

### Benchmark Entry Points

- Individual: `python -m benchmarks.bench_e2e --model llama-3.2-1b --bits 2.5`
- Full suite: `python -m benchmarks.run_all`
- Progress bars via `tqdm`

## Testing

### test_codebook.py
- Verify Lloyd-Max centroids match expected values from the paper (e.g., b=1: `+/- sqrt(2/(pi*d))`)
- Codebook is sorted, distortion decreases with more bits

### test_quantize.py
- Round-trip: quantize -> dequantize produces vectors with MSE within theoretical bounds
- TurboQuant_prod inner products are unbiased (mean error ~ 0 over 10K trials)
- Bit-width storage is correct (b bits per coordinate)

### test_kv_cache.py
- KV cache wrapper stores/retrieves correctly
- Memory footprint is reduced vs fp16
- Attention output with quantized keys is close to fp16 baseline (cosine similarity > 0.95)

## Dependencies

- `torch >= 2.1` (MPS support)
- `transformers` (model loading, tokenizer, generation)
- `torchao` (4-bit weight quantization for 7B/8B models — Apple Silicon compatible, unlike bitsandbytes which requires CUDA)
- `datasets` (WikiText-2, MMLU)
- `psutil` (memory profiling)
- `matplotlib` (benchmark charts)
- `scipy` (Beta distribution for codebook construction)
- `numpy`
- `pytest` (testing)
- `tqdm` (progress bars)

## Primary API Surface

```python
# Core quantization
from pytq import TurboQuantMSE, TurboQuantProd

quantizer = TurboQuantMSE(dim=128, bits=2, device="mps")
q = quantizer.quantize(x)        # x: Tensor[..., dim] -> QuantizedTensor
x_hat = quantizer.dequantize(q)  # QuantizedTensor -> Tensor[..., dim]

quantizer_prod = TurboQuantProd(dim=128, bits=3, device="mps")
q = quantizer_prod.quantize(x)
ip = quantizer_prod.inner_product(q, y)  # unbiased <y, x> estimate

# KV cache integration
from pytq import TurboQuantKVCache

cache = TurboQuantKVCache(bits=2.5, quantize_keys=True, quantize_values=False)
model.generate(inputs, past_key_values=cache)

# Outlier handling
from pytq import OutlierConfig
outlier_cfg = OutlierConfig(outlier_fraction=0.25, outlier_bits=3, normal_bits=2)
cache = TurboQuantKVCache(outlier_config=outlier_cfg)
```

## Out of Scope

- No custom Metal/Triton kernels -- pure PyTorch only
- No weight quantization implementation -- use existing torchao for that; TurboQuant handles KV cache only
- No training or fine-tuning -- TurboQuant is data-oblivious by design
- No vector search benchmarks -- focused on LLM inference for MacBook Air use case
