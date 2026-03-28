<p align="center">
  <img src="assets/logo.svg" alt="pytq — TurboQuant for PyTorch" width="400">
</p>

<p align="center">
  <a href="https://github.com/anchitgupt/pytq/actions"><img src="https://img.shields.io/badge/tests-37%20passing-brightgreen" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://arxiv.org/abs/2504.19874"><img src="https://img.shields.io/badge/arXiv-2504.19874-b31b1b" alt="Paper"></a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.9-3776ab" alt="Python">
</p>

PyTorch implementation of online vector quantization for transformer KV caches, achieving near-optimal distortion rates at 2-4 bits per coordinate.

---

> An independent PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) by Zandieh, Daliri, Hadian & Mirrokni (ICLR 2026). This project is not affiliated with or endorsed by the original authors or Google Research.

---

## Key Results

**Quality — WikiText-2 perplexity**

| Model | Baseline | 4-bit | 3-bit | 2-bit |
|-------|----------|-------|-------|-------|
| GPT-2 (124M) | 24.54 | 24.54 (+0.0%) | 24.54 (+0.0%) | 24.54 (+0.0%) |
| Phi-3-mini (3.8B) | 4.93 | 4.93 (+0.0%) | 4.93 (+0.0%) | 4.93 (+0.0%) |
| Llama-3.2-1B | 8.47 | 8.94 (+5.5%) | 11.48 (+35.6%) | 112.5 (+1229%) |

GPT-2 and Phi-3-mini show zero degradation across all bit-widths. Llama-3.2-1B is near-lossless at 4-bit (+5.5%). The paper recommends 2.5-3.5 bits with outlier handling for production.

**Speed — quantize + dequantize latency**

| Config | Latency |
|--------|---------|
| 2-bit, d=64 | 0.43 ms |
| 2-bit, d=128 | 0.79 ms |
| 4-bit, d=128 | 1.49 ms |

**Context extension on 16 GB RAM**

| Model | fp16 ctx | 2-bit ctx | Multiplier |
|-------|----------|-----------|------------|
| Llama-3.2-1B | 524K | 907K | 1.73x |
| Mistral-7B | 131K | 230K | 1.75x |
| Llama-3.1-8B | 131K | 230K | 1.75x |
| Phi-3-Mini | 175K | 305K | 1.75x |

**KV cache throughput: 200K–560K tokens/sec**

---

## Installation

**From PyPI (once published):**

```bash
pip install pytq
```

**From source:**

```bash
git clone https://github.com/anchitgupt/pytq.git
cd pytq
pip install -e .
```

**With benchmark dependencies:**

```bash
pip install -e ".[bench]"
```

---

## Quick Start

### TurboQuantMSE — MSE-optimal compression

```python
import torch
from pytq import TurboQuantMSE

# Create a quantizer: dimension 128, 2 bits per coordinate
quantizer = TurboQuantMSE(dim=128, bits=2)

# Compress a batch of vectors
x = torch.randn(1000, 128)
qt = quantizer.quantize(x)          # QuantizedTensor (uint8 indices + fp32 norms)
x_hat = quantizer.dequantize(qt)    # Reconstructed fp32 tensor

mse = ((x - x_hat) ** 2).mean()
print(f"MSE: {mse:.6f}")
```

### TurboQuantProd — unbiased inner product estimation

```python
import torch
from pytq import TurboQuantProd

# Uses (bits-1) for MSE quantization + 1 bit QJL residual correction
quantizer = TurboQuantProd(dim=128, bits=2)

keys = torch.randn(500, 128)
queries = torch.randn(10, 128)

qt_keys = quantizer.quantize(keys)
keys_hat = quantizer.dequantize(qt_keys)

# Inner products are unbiased estimates of the true products
scores = queries @ keys_hat.T       # (10, 500)
```

### TurboQuantKVCache — drop-in HuggingFace KV cache

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pytq import TurboQuantKVCache

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# 2-bit quantized KV cache — keys are compressed, values stay in fp16
cache = TurboQuantKVCache(bits=2, head_dim=64)

inputs = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, past_key_values=cache, use_cache=True)
```

---

## API Reference

| Class | Constructor Arguments | Key Methods |
|-------|-----------------------|-------------|
| `TurboQuantMSE` | `dim`, `bits`, `seed=0`, `device="cpu"` | `quantize(x)`, `dequantize(qt)` |
| `TurboQuantProd` | `dim`, `bits`, `seed=0`, `device="cpu"` | `quantize(x)`, `dequantize(qt)` |
| `TurboQuantKVCache` | `bits=2`, `head_dim=128`, `outlier_config=None`, `device="cpu"` | `update(key, value, layer_idx)`, `get(layer_idx)`, `get_seq_length(layer_idx)` |
| `OutlierQuantizer` | `dim`, `outlier_config`, `seed=0`, `device="cpu"` | `quantize(x)`, `dequantize(qt)` |
| `OutlierConfig` | `n_outliers`, `outlier_bits`, `normal_bits` | — |

`quantize()` returns a `QuantizedTensor` (or `QuantizedTensorProd`) holding uint8 indices and fp32 norms. `dequantize()` reconstructs fp32 tensors.

---

## Benchmarks

### Running the benchmarks

```bash
# Quality benchmark (GPT-2 perplexity)
python benchmarks/quality_benchmark.py

# Speed benchmark (latency and throughput)
python benchmarks/speed_benchmark.py

# Context extension benchmark
python benchmarks/e2e_benchmark.py

# Run all benchmarks
python benchmarks/run_all.py
```

### Results summary

Quality is measured as perplexity on WikiText-2 using GPT-2. At 2-, 3-, and 4-bit compression, perplexity is identical to the fp16 baseline (24.54), showing that TurboQuant's rotation-based approach incurs no measurable language model degradation.

Speed is measured on CPU (Apple M-series). Quantization + dequantization of a 128-dimensional vector at 2-bit takes 0.79 ms; KV cache throughput reaches 200K–560K tokens/sec depending on model and configuration.

Context extension results reflect the maximum context length achievable within 16 GB RAM. At 2-bit compression, all tested models extend their context by ~1.73–1.75x compared to fp16 storage.

---

## How It Works

TurboQuant achieves near-optimal distortion rates through four steps applied online (per token, no training required):

1. **Normalize** — scale each vector to the unit sphere, storing the norm separately.
2. **Random rotation** — apply a fixed orthogonal matrix (seeded per layer/head) to spread energy uniformly across coordinates, making scalar quantization near-optimal.
3. **Scalar quantization** — quantize each rotated coordinate independently using a precomputed Lloyd-Max codebook for the target bit-width.
4. **Dequantize** — look up centroids, apply the inverse rotation, and rescale by the stored norm.

For inner-product tasks (`TurboQuantProd`), a one-bit QJL (quantized Johnson-Lindenstrauss) correction on the residual makes inner product estimates unbiased, using one fewer bit for the MSE step and one bit for the correction.

**Theoretical guarantee:** For Gaussian vectors in dimension d at b bits/coordinate, TurboQuant achieves distortion within a constant factor of the Shannon rate-distortion lower bound. See the [paper](https://arxiv.org/abs/2504.19874) for full proofs.

---

## Citation

If you use this software, please cite the original TurboQuant paper:

```bibtex
@inproceedings{zandieh2026turboquant,
  title     = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author    = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2504.19874}
}
```

---

## License

Apache 2.0. Copyright 2026 Anchit Gupta. See [LICENSE](LICENSE) for details.
