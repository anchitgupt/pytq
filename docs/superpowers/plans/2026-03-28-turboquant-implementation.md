# TurboQuant (pytq) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the TurboQuant vector quantization algorithm in PyTorch with HuggingFace KV cache integration, and benchmark memory/quality/speed tradeoffs for LLM inference on a 16GB MacBook Air.

**Architecture:** Core quantization library (`pytq/`) with three layers: (1) primitives (codebook, rotation), (2) quantizers (MSE, prod), (3) HuggingFace integration (KV cache wrapper). Benchmark harness (`benchmarks/`) measures distortion, memory, quality, speed, and end-to-end fitness.

**Tech Stack:** Python 3.10+, PyTorch >= 2.1 (MPS), HuggingFace Transformers, scipy, torchao, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-03-28-turboquant-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package config, dependencies |
| `pytq/__init__.py` | Public API exports |
| `pytq/codebook.py` | Lloyd-Max codebook construction for Beta distribution |
| `pytq/rotation.py` | Deterministic random orthogonal matrix generation (QR) |
| `pytq/quantize_mse.py` | TurboQuant_mse: normalize, rotate, scalar quantize, dequantize |
| `pytq/quantize_prod.py` | TurboQuant_prod: MSE + QJL residual correction for unbiased IP |
| `pytq/outlier.py` | Outlier channel detection and split bit-width allocation |
| `pytq/kv_cache.py` | HuggingFace DynamicCache wrapper with TurboQuant compression |
| `pytq/utils.py` | Distortion metrics, memory helpers |
| `tests/test_codebook.py` | Codebook correctness tests |
| `tests/test_quantize.py` | MSE + prod quantizer round-trip and property tests |
| `tests/test_kv_cache.py` | KV cache integration tests |
| `benchmarks/bench_distortion.py` | Theoretical validation: error vs bounds |
| `benchmarks/bench_memory.py` | RAM profiling per model x bit-width x seq-length |
| `benchmarks/bench_quality.py` | Perplexity + MMLU accuracy |
| `benchmarks/bench_speed.py` | Tokens/sec, attention latency |
| `benchmarks/bench_e2e.py` | End-to-end "will it fit on 16GB?" |
| `benchmarks/run_all.py` | Full suite runner |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `pytq/__init__.py`
- Create: `pytq/utils.py`
- Create: `tests/__init__.py`
- Create: `benchmarks/__init__.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/anchitgupta/Documents/Github/pytq
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pytq"
version = "0.1.0"
description = "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "numpy",
    "scipy",
]

[project.optional-dependencies]
bench = [
    "transformers",
    "torchao",
    "datasets",
    "psutil",
    "matplotlib",
    "tqdm",
]
dev = [
    "pytest",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create pytq/__init__.py**

```python
# Empty initially — populated in Task 8 after all modules exist.
```

- [ ] **Step 4: Create pytq/utils.py**

```python
import torch
from dataclasses import dataclass
import math


@dataclass
class QuantizedTensor:
    """Compressed representation of a quantized vector."""
    indices: torch.Tensor   # uint8, shape [..., dim]
    norm: torch.Tensor      # fp32, shape [...]
    bits: int
    dim: int


def mse_distortion(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute MSE distortion: E[||x - x_hat||^2] normalized by ||x||^2."""
    diff = original - reconstructed
    mse = (diff * diff).sum(dim=-1)
    norm_sq = (original * original).sum(dim=-1)
    return (mse / norm_sq).mean()


def ip_distortion(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute inner product distortion: E[|<y, x> - <y, x_hat>|^2] and bias.

    Returns (variance, bias) of the inner product error.
    """
    ip_true = torch.einsum("...d,...d->...", query, original)
    ip_approx = torch.einsum("...d,...d->...", query, reconstructed)
    error = ip_approx - ip_true
    bias = error.mean()
    variance = (error * error).mean()
    return variance, bias


def mse_upper_bound(bits: int) -> float:
    """Theoretical upper bound: sqrt(3)*pi/2 * (1/4^b)."""
    return (math.sqrt(3) * math.pi / 2) * (1.0 / (4 ** bits))


def mse_lower_bound(bits: int) -> float:
    """Information-theoretic lower bound: 1/4^b."""
    return 1.0 / (4 ** bits)


def ip_upper_bound(bits: int, dim: int) -> float:
    """Theoretical upper bound for IP distortion: sqrt(3)*pi^2/d * (1/4^b)."""
    return (math.sqrt(3) * math.pi**2 / dim) * (1.0 / (4 ** bits))


def get_memory_bytes(tensor: torch.Tensor) -> int:
    """Return memory usage of a tensor in bytes."""
    return tensor.nelement() * tensor.element_size()
```

- [ ] **Step 5: Create empty test and benchmark __init__.py files**

Create `tests/__init__.py` and `benchmarks/__init__.py` as empty files.

- [ ] **Step 6: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
results/
.pytest_cache/
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml pytq/__init__.py pytq/utils.py tests/__init__.py benchmarks/__init__.py .gitignore docs/
git commit -m "feat: project scaffolding with pyproject.toml, utils, and design docs"
```

---

## Task 2: Codebook Construction (Lloyd-Max)

**Files:**
- Create: `pytq/codebook.py`
- Create: `tests/test_codebook.py`

- [ ] **Step 1: Write failing tests for codebook**

```python
# tests/test_codebook.py
import torch
import math
import pytest
from pytq.codebook import build_codebook


class TestBuildCodebook:
    def test_1bit_centroid_count(self):
        """1-bit codebook should have 2 centroids."""
        centroids = build_codebook(dim=128, bits=1)
        assert centroids.shape == (2,)

    def test_2bit_centroid_count(self):
        """2-bit codebook should have 4 centroids."""
        centroids = build_codebook(dim=128, bits=2)
        assert centroids.shape == (4,)

    def test_3bit_centroid_count(self):
        centroids = build_codebook(dim=128, bits=3)
        assert centroids.shape == (8,)

    def test_4bit_centroid_count(self):
        centroids = build_codebook(dim=128, bits=4)
        assert centroids.shape == (16,)

    def test_centroids_sorted(self):
        """Centroids should be sorted in ascending order."""
        for bits in [1, 2, 3, 4]:
            centroids = build_codebook(dim=128, bits=bits)
            assert torch.all(centroids[1:] > centroids[:-1]), f"Not sorted for bits={bits}"

    def test_centroids_symmetric(self):
        """Centroids should be symmetric around 0 (distribution is symmetric)."""
        for bits in [1, 2, 3, 4]:
            centroids = build_codebook(dim=128, bits=bits)
            assert torch.allclose(centroids, -centroids.flip(0), atol=1e-4), (
                f"Not symmetric for bits={bits}"
            )

    def test_1bit_centroid_values_high_dim(self):
        """For b=1, high-dim centroids should approximate +/- sqrt(2/(pi*d))."""
        dim = 1024
        centroids = build_codebook(dim=dim, bits=1)
        expected = math.sqrt(2.0 / (math.pi * dim))
        assert abs(abs(centroids[1].item()) - expected) < 0.01 * expected

    def test_distortion_decreases_with_bits(self):
        """More bits should give lower quantization distortion."""
        dim = 128
        distortions = []
        for bits in [1, 2, 3, 4]:
            centroids = build_codebook(dim=dim, bits=bits)
            # Distortion is computed internally by build_codebook
            # We test indirectly: more centroids => smaller cell sizes
            max_gap = (centroids[1:] - centroids[:-1]).max().item()
            distortions.append(max_gap)
        for i in range(len(distortions) - 1):
            assert distortions[i] > distortions[i + 1]

    def test_different_dims_give_different_codebooks(self):
        """Codebooks for different dimensions should differ."""
        c64 = build_codebook(dim=64, bits=2)
        c256 = build_codebook(dim=256, bits=2)
        assert not torch.allclose(c64, c256)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/anchitgupta/Documents/Github/pytq && python -m pytest tests/test_codebook.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pytq.codebook'`

- [ ] **Step 3: Implement codebook.py**

```python
# pytq/codebook.py
"""Lloyd-Max codebook construction for the Beta distribution on the unit sphere.

After random rotation, each coordinate of a unit-norm vector follows:
  f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2}
for x in [-1, 1]. This module builds optimal scalar quantizers for this distribution.
"""
import torch
import math
from scipy.special import gammaln
import numpy as np
from functools import lru_cache


def _beta_pdf(x: np.ndarray, dim: int) -> np.ndarray:
    """PDF of a single coordinate of a uniformly random point on S^{d-1}.

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2}
    """
    log_norm = gammaln(dim / 2) - 0.5 * math.log(math.pi) - gammaln((dim - 1) / 2)
    exponent = (dim - 3) / 2.0
    log_pdf = log_norm + exponent * np.log(np.maximum(1 - x * x, 1e-300))
    return np.exp(log_pdf)


def _lloyd_max(pdf_values: np.ndarray, grid: np.ndarray, n_centroids: int,
               max_iter: int = 300) -> np.ndarray:
    """Run Lloyd-Max algorithm on a discrete grid with given PDF values.

    Args:
        pdf_values: PDF evaluated at grid points (unnormalized is fine).
        grid: 1D array of grid points.
        n_centroids: Number of centroids (2^b).
        max_iter: Maximum iterations.

    Returns:
        Sorted array of optimal centroids.
    """
    # Normalize PDF to sum to 1 (discrete approximation)
    weights = pdf_values / pdf_values.sum()

    # Initialize centroids uniformly across the grid range
    centroids = np.linspace(grid[0], grid[-1], n_centroids)

    for _ in range(max_iter):
        # Assignment: each grid point to nearest centroid
        # distances[i, j] = |grid[i] - centroids[j]|
        dists = np.abs(grid[:, None] - centroids[None, :])
        assignments = np.argmin(dists, axis=1)

        # Update: weighted mean of assigned points
        new_centroids = np.zeros_like(centroids)
        for k in range(n_centroids):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = np.average(grid[mask], weights=weights[mask])
            else:
                new_centroids[k] = centroids[k]

        if np.allclose(centroids, new_centroids, atol=1e-10):
            break
        centroids = new_centroids

    centroids.sort()
    return centroids


@lru_cache(maxsize=64)
def build_codebook(dim: int, bits: int, grid_size: int = 50000) -> torch.Tensor:
    """Build Lloyd-Max optimal codebook for the sphere coordinate distribution.

    Args:
        dim: Dimension of the vectors (d).
        bits: Number of bits per coordinate (b). Codebook has 2^b entries.
        grid_size: Number of grid points for the discrete approximation.

    Returns:
        Sorted tensor of centroids, shape (2^b,).
    """
    n_centroids = 2 ** bits

    # Build a dense grid over the support [-1, 1], focusing on the high-density region
    # For high dim, the distribution concentrates around 0 with std ~ 1/sqrt(d)
    sigma = 1.0 / math.sqrt(dim)
    bound = min(1.0, 6 * sigma)  # 6-sigma covers essentially all mass
    grid = np.linspace(-bound, bound, grid_size)

    pdf_vals = _beta_pdf(grid, dim)

    centroids_np = _lloyd_max(pdf_vals, grid, n_centroids)
    centroids = torch.tensor(centroids_np, dtype=torch.float32)
    centroids.requires_grad_(False)
    return centroids.clone()  # clone to prevent mutation of cached tensor
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/anchitgupta/Documents/Github/pytq && python -m pytest tests/test_codebook.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pytq/codebook.py tests/test_codebook.py
git commit -m "feat: Lloyd-Max codebook construction for sphere coordinate distribution"
```

---

## Task 3: Random Rotation

**Files:**
- Create: `pytq/rotation.py`
- Create: `tests/test_rotation.py`

- [ ] **Step 1: Write failing tests for rotation**

Create `tests/test_rotation.py` with the following content:

```python
# tests/test_rotation.py
import torch
import pytest
from pytq.rotation import generate_rotation_matrix


class TestRotationMatrix:
    def test_orthogonal(self):
        """Rotation matrix should be orthogonal: R^T @ R = I."""
        R = generate_rotation_matrix(dim=128, seed=42)
        eye = torch.eye(128)
        assert torch.allclose(R.T @ R, eye, atol=1e-5)

    def test_determinant_one(self):
        """Orthogonal matrix from QR has |det| = 1."""
        R = generate_rotation_matrix(dim=64, seed=42)
        det = torch.linalg.det(R)
        assert abs(abs(det.item()) - 1.0) < 1e-4

    def test_deterministic_with_seed(self):
        """Same seed should produce same rotation matrix."""
        R1 = generate_rotation_matrix(dim=128, seed=42)
        R2 = generate_rotation_matrix(dim=128, seed=42)
        assert torch.allclose(R1, R2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different matrices."""
        R1 = generate_rotation_matrix(dim=128, seed=42)
        R2 = generate_rotation_matrix(dim=128, seed=99)
        assert not torch.allclose(R1, R2)

    def test_shape(self):
        R = generate_rotation_matrix(dim=64, seed=0)
        assert R.shape == (64, 64)

    def test_preserves_norm(self):
        """Rotation should preserve vector norm."""
        R = generate_rotation_matrix(dim=128, seed=42)
        x = torch.randn(128)
        y = R @ x
        assert torch.allclose(x.norm(), y.norm(), atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rotation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement rotation.py**

```python
# pytq/rotation.py
"""Deterministic random orthogonal matrix generation via QR decomposition."""
import torch


def generate_rotation_matrix(
    dim: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition of a Gaussian matrix.

    The matrix is deterministic for a given (dim, seed) pair.

    Args:
        dim: Matrix dimension (d x d).
        seed: RNG seed for reproducibility.
        device: Target device.

    Returns:
        Orthogonal matrix Q, shape (dim, dim).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    # Generate random Gaussian matrix on CPU (for determinism), then move
    G = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)

    # Make QR decomposition unique by ensuring positive diagonal in R
    # This prevents sign ambiguity in Q
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs.unsqueeze(0)

    return Q.to(device)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rotation.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pytq/rotation.py tests/test_rotation.py
git commit -m "feat: deterministic random orthogonal rotation via QR decomposition"
```

---

## Task 4: TurboQuant_mse Quantizer

**Files:**
- Create: `pytq/quantize_mse.py`
- Create: `tests/test_quantize.py`

- [ ] **Step 1: Write failing tests for TurboQuant_mse**

```python
# tests/test_quantize.py
import torch
import math
import pytest
from pytq.quantize_mse import TurboQuantMSE
from pytq.utils import mse_distortion, mse_upper_bound


class TestTurboQuantMSE:
    def test_quantize_returns_quantized_tensor(self):
        q = TurboQuantMSE(dim=128, bits=2, seed=42)
        x = torch.randn(10, 128)
        result = q.quantize(x)
        assert result.indices.shape == (10, 128)
        assert result.indices.dtype == torch.uint8
        assert result.norm.shape == (10,)
        assert result.bits == 2
        assert result.dim == 128

    def test_dequantize_shape(self):
        q = TurboQuantMSE(dim=128, bits=2, seed=42)
        x = torch.randn(10, 128)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == x.shape

    def test_roundtrip_mse_within_bound(self):
        """MSE should be within theoretical upper bound for unit-norm vectors."""
        dim = 128
        for bits in [1, 2, 3, 4]:
            q = TurboQuantMSE(dim=dim, bits=bits, seed=42)
            x = torch.randn(1000, dim)
            x = x / x.norm(dim=-1, keepdim=True)  # unit norm
            qt = q.quantize(x)
            x_hat = q.dequantize(qt)
            mse = mse_distortion(x, x_hat).item()
            bound = mse_upper_bound(bits)
            assert mse < bound * 1.5, (
                f"bits={bits}: MSE {mse:.4f} exceeds 1.5x bound {bound:.4f}"
            )

    def test_mse_decreases_with_bits(self):
        """Higher bit-width should give lower MSE."""
        dim = 128
        x = torch.randn(500, dim)
        mses = []
        for bits in [1, 2, 3, 4]:
            q = TurboQuantMSE(dim=dim, bits=bits, seed=42)
            qt = q.quantize(x)
            x_hat = q.dequantize(qt)
            mse = mse_distortion(x, x_hat).item()
            mses.append(mse)
        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], f"MSE did not decrease: {mses}"

    def test_preserves_norm(self):
        """Dequantized vector should approximately preserve input norm."""
        q = TurboQuantMSE(dim=128, bits=4, seed=42)
        x = torch.randn(100, 128) * 5.0  # non-unit norm
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        norm_orig = x.norm(dim=-1)
        norm_recon = x_hat.norm(dim=-1)
        ratio = norm_recon / norm_orig
        assert (ratio.mean() - 1.0).abs() < 0.2

    def test_deterministic(self):
        """Same seed should give same quantization."""
        q = TurboQuantMSE(dim=64, bits=2, seed=42)
        x = torch.randn(10, 64)
        qt1 = q.quantize(x)
        qt2 = q.quantize(x)
        assert torch.equal(qt1.indices, qt2.indices)

    def test_batch_dims(self):
        """Should work with arbitrary leading batch dimensions."""
        q = TurboQuantMSE(dim=64, bits=2, seed=42)
        x = torch.randn(4, 8, 64)
        qt = q.quantize(x)
        assert qt.indices.shape == (4, 8, 64)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == (4, 8, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_quantize.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement quantize_mse.py**

```python
# pytq/quantize_mse.py
"""TurboQuant_mse: MSE-optimal vector quantization via rotation + scalar quantization."""
import torch
from pytq.codebook import build_codebook
from pytq.rotation import generate_rotation_matrix
from pytq.utils import QuantizedTensor


class TurboQuantMSE:
    """MSE-optimal quantizer using random rotation + Lloyd-Max scalar quantization.

    Algorithm:
        1. Normalize input to unit sphere, store norm
        2. Rotate with random orthogonal matrix (deterministic from seed)
        3. Scalar quantize each coordinate using precomputed Lloyd-Max codebook
        4. Dequantize: lookup centroids, inverse rotate, rescale by norm
    """

    def __init__(self, dim: int, bits: int, seed: int = 0, device: str = "cpu"):
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = device

        self.codebook = build_codebook(dim, bits).to(device)
        self.rotation = generate_rotation_matrix(dim, seed, device=device)

    def quantize(self, x: torch.Tensor) -> QuantizedTensor:
        """Quantize input vectors.

        Args:
            x: Input tensor, shape [..., dim].

        Returns:
            QuantizedTensor with indices and norms.
        """
        orig_shape = x.shape
        assert orig_shape[-1] == self.dim
        x_flat = x.reshape(-1, self.dim).to(self.device)

        # Step 1: Normalize
        norms = x_flat.norm(dim=-1)
        safe_norms = norms.clamp(min=1e-8)
        x_hat = x_flat / safe_norms.unsqueeze(-1)

        # Step 2: Rotate
        y = x_hat @ self.rotation.T  # (N, dim) @ (dim, dim)^T = (N, dim)

        # Step 3: Scalar quantize — find nearest centroid per coordinate
        # y: (N, dim), codebook: (2^b,)
        dists = (y.unsqueeze(-1) - self.codebook.unsqueeze(0).unsqueeze(0)).abs()
        indices = dists.argmin(dim=-1).to(torch.uint8)

        batch_shape = orig_shape[:-1]
        return QuantizedTensor(
            indices=indices.reshape(*batch_shape, self.dim),
            norm=norms.reshape(*batch_shape),
            bits=self.bits,
            dim=self.dim,
        )

    def dequantize(self, qt: QuantizedTensor) -> torch.Tensor:
        """Dequantize back to full vectors.

        Args:
            qt: QuantizedTensor from quantize().

        Returns:
            Reconstructed tensor, shape [..., dim].
        """
        batch_shape = qt.indices.shape[:-1]
        indices_flat = qt.indices.reshape(-1, self.dim).long()
        norms_flat = qt.norm.reshape(-1)

        # Step 1: Lookup centroids
        y_tilde = self.codebook[indices_flat]  # (N, dim)

        # Step 2: Inverse rotate
        x_hat_tilde = y_tilde @ self.rotation  # (N, dim) @ (dim, dim) = (N, dim)

        # Step 3: Rescale
        x_tilde = x_hat_tilde * norms_flat.unsqueeze(-1)

        return x_tilde.reshape(*batch_shape, self.dim)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_quantize.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pytq/quantize_mse.py tests/test_quantize.py
git commit -m "feat: TurboQuant_mse quantizer with normalize-rotate-quantize pipeline"
```

---

## Task 5: TurboQuant_prod Quantizer

**Files:**
- Create: `pytq/quantize_prod.py`
- Extend: `tests/test_quantize.py`

- [ ] **Step 1: Write failing tests for TurboQuant_prod**

Add to `tests/test_quantize.py`: insert the two new imports at the top of the file (after existing imports), then add the `TestTurboQuantProd` class at the bottom of the file.

```python
# Add these imports at the TOP of tests/test_quantize.py, after existing imports:
from pytq.quantize_prod import TurboQuantProd
from pytq.utils import ip_distortion

# Add this class at the BOTTOM of tests/test_quantize.py:


class TestTurboQuantProd:
    def test_quantize_returns_result(self):
        q = TurboQuantProd(dim=128, bits=2, seed=42)
        x = torch.randn(10, 128)
        result = q.quantize(x)
        assert result.indices.shape == (10, 128)

    def test_inner_product_unbiased(self):
        """Inner product estimation should be unbiased (mean error ~ 0)."""
        dim = 128
        q = TurboQuantProd(dim=dim, bits=2, seed=42)
        x = torch.randn(5000, dim)
        y = torch.randn(5000, dim)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        ip_true = (y * x).sum(dim=-1)
        ip_approx = (y * x_hat).sum(dim=-1)
        bias = (ip_approx - ip_true).mean().item()
        assert abs(bias) < 0.5, f"Bias too large: {bias}"

    def test_ip_variance_decreases_with_bits(self):
        """Higher bit-width should give lower IP variance."""
        dim = 128
        x = torch.randn(1000, dim)
        y = torch.randn(1000, dim)
        variances = []
        for bits in [2, 3, 4]:
            q = TurboQuantProd(dim=dim, bits=bits, seed=42)
            qt = q.quantize(x)
            x_hat = q.dequantize(qt)
            var, _ = ip_distortion(x, x_hat, y)
            variances.append(var.item())
        for i in range(len(variances) - 1):
            assert variances[i] > variances[i + 1], f"Variance did not decrease: {variances}"

    def test_dequantize_shape(self):
        q = TurboQuantProd(dim=64, bits=3, seed=42)
        x = torch.randn(10, 64)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == (10, 64)

    def test_batch_dims(self):
        q = TurboQuantProd(dim=64, bits=2, seed=42)
        x = torch.randn(4, 8, 64)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == (4, 8, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_quantize.py::TestTurboQuantProd -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement quantize_prod.py**

```python
# pytq/quantize_prod.py
"""TurboQuant_prod: Inner-product-optimal quantization with QJL residual correction."""
import torch
import math
from dataclasses import dataclass
from pytq.quantize_mse import TurboQuantMSE
from pytq.utils import QuantizedTensor


@dataclass
class QuantizedTensorProd:
    """Compressed representation for TurboQuant_prod."""
    mse_qt: QuantizedTensor          # MSE quantization at (b-1) bits
    qjl_signs: torch.Tensor          # sign bits from QJL, shape [..., dim]
    residual_norm: torch.Tensor      # ||r||, shape [...]
    bits: int
    dim: int

    # Proxy attributes so it can be used like a QuantizedTensor
    @property
    def indices(self) -> torch.Tensor:
        return self.mse_qt.indices

    @property
    def norm(self) -> torch.Tensor:
        return self.mse_qt.norm


class TurboQuantProd:
    """Inner-product-optimal quantizer: MSE quantization + QJL residual correction.

    Uses (b-1) bits for MSE quantization and 1 bit for QJL on the residual.
    This gives unbiased inner product estimates.
    """

    def __init__(self, dim: int, bits: int, seed: int = 0, device: str = "cpu"):
        assert bits >= 2, "TurboQuant_prod requires at least 2 bits (1 for MSE + 1 for QJL)"
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = device

        # MSE quantizer at (b-1) bits
        self.mse_quantizer = TurboQuantMSE(dim, bits - 1, seed=seed, device=device)

        # QJL random matrix — generated from seed, shared across all tokens
        self._qjl_seed = seed + 1_000_000  # offset to avoid collision with rotation seed

    def _get_qjl_matrix(self, device: str = "cpu") -> torch.Tensor:
        """Generate the QJL projection matrix S deterministically."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self._qjl_seed)
        S = torch.randn(self.dim, self.dim, generator=gen, dtype=torch.float32)
        return S.to(device)

    def quantize(self, x: torch.Tensor) -> QuantizedTensorProd:
        """Quantize with MSE + QJL residual correction."""
        # Step 1: MSE quantize at (b-1) bits
        mse_qt = self.mse_quantizer.quantize(x)
        x_mse = self.mse_quantizer.dequantize(mse_qt)

        # Step 2: Compute residual
        residual = x - x_mse
        batch_shape = residual.shape[:-1]
        residual_flat = residual.reshape(-1, self.dim).to(self.device)

        # Step 3: Residual norm
        residual_norm = residual_flat.norm(dim=-1)

        # Step 4: QJL sign bits
        S = self._get_qjl_matrix(self.device)
        projected = residual_flat @ S.T  # (N, dim)
        signs = (projected >= 0).to(torch.uint8)  # 1 bit per coordinate

        return QuantizedTensorProd(
            mse_qt=mse_qt,
            qjl_signs=signs.reshape(*batch_shape, self.dim),
            residual_norm=residual_norm.reshape(*batch_shape),
            bits=self.bits,
            dim=self.dim,
        )

    def dequantize(self, qt: QuantizedTensorProd) -> torch.Tensor:
        """Dequantize with QJL correction for unbiased inner products."""
        # Step 1: MSE dequantize
        x_mse = self.mse_quantizer.dequantize(qt.mse_qt)

        # Step 2: QJL dequantize the residual
        batch_shape = qt.qjl_signs.shape[:-1]
        signs_flat = qt.qjl_signs.reshape(-1, self.dim).float()
        r_norm_flat = qt.residual_norm.reshape(-1)

        # Convert 0/1 signs to -1/+1
        z = 2.0 * signs_flat - 1.0

        S = self._get_qjl_matrix(self.device)
        # Q_qjl^{-1}(z) = sqrt(pi/2) / d * ||r|| * S^T @ z
        scale = math.sqrt(math.pi / 2) / self.dim
        residual_hat = scale * r_norm_flat.unsqueeze(-1) * (z @ S)  # (N, dim)

        x_hat = x_mse + residual_hat.reshape(*batch_shape, self.dim)
        return x_hat
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_quantize.py -v`
Expected: All 12 tests PASS (7 MSE + 5 prod)

- [ ] **Step 5: Commit**

```bash
git add pytq/quantize_prod.py tests/test_quantize.py
git commit -m "feat: TurboQuant_prod quantizer with QJL residual for unbiased inner products"
```

---

## Task 6: Outlier Channel Handling

**Files:**
- Create: `pytq/outlier.py`
- Extend: `tests/test_quantize.py`

- [ ] **Step 1: Write failing tests for outlier handling**

Add to `tests/test_quantize.py`: insert the new import at the top of the file (after existing imports), then add the `TestOutlierQuantizer` class at the bottom of the file.

```python
# Add this import at the TOP of tests/test_quantize.py, after existing imports:
from pytq.outlier import OutlierConfig, OutlierQuantizer

# Add this class at the BOTTOM of tests/test_quantize.py:


class TestOutlierQuantizer:
    def test_effective_bits(self):
        cfg = OutlierConfig(outlier_fraction=0.25, outlier_bits=3, normal_bits=2)
        assert cfg.effective_bits == 2.25  # (0.25*3 + 0.75*2) = 2.25
        # For head_dim=128: 32*3 + 96*2 = 288 / 128 = 2.25

    def test_quantize_shape(self):
        cfg = OutlierConfig(outlier_fraction=0.25, outlier_bits=3, normal_bits=2)
        oq = OutlierQuantizer(dim=128, config=cfg, seed=42)
        x = torch.randn(10, 128)
        result = oq.quantize(x)
        assert result.dim == 128

    def test_dequantize_roundtrip(self):
        cfg = OutlierConfig(outlier_fraction=0.25, outlier_bits=3, normal_bits=2)
        oq = OutlierQuantizer(dim=128, config=cfg, seed=42)
        x = torch.randn(100, 128)
        qt = oq.quantize(x)
        x_hat = oq.dequantize(qt)
        assert x_hat.shape == x.shape
        # Should be better than pure 2-bit (due to outlier channels getting 3 bits)
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean()
        assert mse < ((x ** 2).sum(dim=-1).mean() * 0.5)  # not garbage

    def test_different_head_dims(self):
        """Should work with head_dim != 128."""
        for dim in [64, 96, 128]:
            cfg = OutlierConfig(outlier_fraction=0.25, outlier_bits=3, normal_bits=2)
            oq = OutlierQuantizer(dim=dim, config=cfg, seed=42)
            x = torch.randn(10, dim)
            qt = oq.quantize(x)
            x_hat = oq.dequantize(qt)
            assert x_hat.shape == x.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_quantize.py::TestOutlierQuantizer -v`
Expected: FAIL

- [ ] **Step 3: Implement outlier.py**

```python
# pytq/outlier.py
"""Outlier channel detection and split bit-width allocation.

Channels with highest magnitude get more bits; remaining channels get fewer bits.
E.g., 2.5-bit = 25% outliers at 3 bits + 75% at 2 bits.
"""
import torch
from dataclasses import dataclass
from pytq.quantize_mse import TurboQuantMSE
from pytq.utils import QuantizedTensor


@dataclass
class OutlierConfig:
    outlier_fraction: float = 0.25  # fraction of channels treated as outliers
    outlier_bits: int = 3
    normal_bits: int = 2

    @property
    def effective_bits(self) -> float:
        return self.outlier_fraction * self.outlier_bits + (1 - self.outlier_fraction) * self.normal_bits


@dataclass
class OutlierQuantizedTensor:
    """Stores separate quantizations for outlier and normal channels."""
    outlier_qt: QuantizedTensor
    normal_qt: QuantizedTensor
    outlier_indices_map: torch.Tensor  # which channels are outliers
    normal_indices_map: torch.Tensor
    dim: int
    norm: torch.Tensor  # original full-vector norm

    @property
    def bits(self) -> float:
        return (self.outlier_qt.bits * len(self.outlier_indices_map)
                + self.normal_qt.bits * len(self.normal_indices_map)) / self.dim

    @property
    def indices(self) -> torch.Tensor:
        """Not meaningful for outlier quantizer, but provided for compatibility."""
        return self.normal_qt.indices


class OutlierQuantizer:
    """Quantizer that allocates more bits to outlier channels.

    Approach: Apply the full-vector random rotation first (mapping the full vector
    onto the sphere coordinate distribution), then split the rotated coordinates
    into outlier and non-outlier sets based on magnitude. Each set is scalar-quantized
    independently using its own Lloyd-Max codebook (built for the full dim, since
    the rotation was applied to the full vector). This ensures the codebook matches
    the actual data distribution.
    """

    def __init__(self, dim: int, config: OutlierConfig, seed: int = 0, device: str = "cpu"):
        self.dim = dim
        self.config = config
        self.device = device
        self.n_outlier = max(1, int(dim * config.outlier_fraction))
        self.n_normal = dim - self.n_outlier

        # Shared rotation for the full vector
        from pytq.rotation import generate_rotation_matrix
        from pytq.codebook import build_codebook
        self.rotation = generate_rotation_matrix(dim, seed, device=device)
        self.codebook_outlier = build_codebook(dim, config.outlier_bits).to(device)
        self.codebook_normal = build_codebook(dim, config.normal_bits).to(device)

    def _select_outlier_channels(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select outlier channels based on average magnitude of rotated coordinates."""
        avg_magnitude = y.abs().mean(dim=0)  # (dim,)
        _, sorted_idx = avg_magnitude.sort(descending=True)
        outlier_idx = sorted_idx[:self.n_outlier].sort().values
        normal_idx = sorted_idx[self.n_outlier:].sort().values
        return outlier_idx, normal_idx

    def quantize(self, x: torch.Tensor) -> OutlierQuantizedTensor:
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.dim).to(self.device)

        # Normalize
        norms = x_flat.norm(dim=-1)
        safe_norms = norms.clamp(min=1e-8)
        x_hat = x_flat / safe_norms.unsqueeze(-1)

        # Rotate full vector
        y = x_hat @ self.rotation.T  # (N, dim)

        # Split channels by magnitude
        outlier_idx, normal_idx = self._select_outlier_channels(y)

        # Scalar quantize each group with its own codebook
        y_outlier = y[:, outlier_idx]
        y_normal = y[:, normal_idx]

        dists_o = (y_outlier.unsqueeze(-1) - self.codebook_outlier.unsqueeze(0).unsqueeze(0)).abs()
        idx_o = dists_o.argmin(dim=-1).to(torch.uint8)

        dists_n = (y_normal.unsqueeze(-1) - self.codebook_normal.unsqueeze(0).unsqueeze(0)).abs()
        idx_n = dists_n.argmin(dim=-1).to(torch.uint8)

        qt_outlier = QuantizedTensor(indices=idx_o, norm=norms, bits=self.config.outlier_bits, dim=self.n_outlier)
        qt_normal = QuantizedTensor(indices=idx_n, norm=norms, bits=self.config.normal_bits, dim=self.n_normal)

        return OutlierQuantizedTensor(
            outlier_qt=qt_outlier,
            normal_qt=qt_normal,
            outlier_indices_map=outlier_idx,
            normal_indices_map=normal_idx,
            dim=self.dim,
            norm=norms.reshape(*batch_shape),
        )

    def dequantize(self, qt: OutlierQuantizedTensor) -> torch.Tensor:
        batch_shape = qt.norm.shape
        norms_flat = qt.norm.reshape(-1)

        # Lookup centroids
        y_outlier = self.codebook_outlier[qt.outlier_qt.indices.long()]
        y_normal = self.codebook_normal[qt.normal_qt.indices.long()]

        N = y_outlier.shape[0]
        y_full = torch.zeros(N, self.dim, device=y_outlier.device)
        y_full[:, qt.outlier_indices_map] = y_outlier
        y_full[:, qt.normal_indices_map] = y_normal

        # Inverse rotate and rescale
        x_hat_tilde = y_full @ self.rotation
        x_tilde = x_hat_tilde * norms_flat.unsqueeze(-1)

        return x_tilde.reshape(*batch_shape, self.dim)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_quantize.py -v`
Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pytq/outlier.py tests/test_quantize.py
git commit -m "feat: outlier channel detection with split bit-width allocation"
```

---

## Task 7: KV Cache Integration

**Files:**
- Create: `pytq/kv_cache.py`
- Create: `tests/test_kv_cache.py`

- [ ] **Step 1: Write failing tests for KV cache**

```python
# tests/test_kv_cache.py
import torch
import pytest
from pytq.kv_cache import TurboQuantKVCache


class TestTurboQuantKVCache:
    def test_update_and_retrieve(self):
        """Store and retrieve keys/values."""
        cache = TurboQuantKVCache(bits=2, head_dim=64)
        # Simulate: batch=1, n_heads=4, seq_len=8, head_dim=64
        key = torch.randn(1, 4, 8, 64)
        value = torch.randn(1, 4, 8, 64)
        cache.update(key, value, layer_idx=0)
        k_out, v_out = cache.get(layer_idx=0)
        assert k_out.shape == key.shape
        assert v_out.shape == value.shape

    def test_values_stored_exactly(self):
        """Values should be stored in fp16, not quantized."""
        cache = TurboQuantKVCache(bits=2, head_dim=64)
        value = torch.randn(1, 4, 8, 64)
        key = torch.randn(1, 4, 8, 64)
        cache.update(key, value, layer_idx=0)
        _, v_out = cache.get(layer_idx=0)
        assert torch.allclose(v_out, value, atol=1e-3)

    def test_keys_approximately_preserved(self):
        """Dequantized keys should be close to originals."""
        cache = TurboQuantKVCache(bits=4, head_dim=64)
        key = torch.randn(1, 4, 8, 64)
        value = torch.randn(1, 4, 8, 64)
        cache.update(key, value, layer_idx=0)
        k_out, _ = cache.get(layer_idx=0)
        cos_sim = torch.nn.functional.cosine_similarity(
            key.reshape(-1, 64), k_out.reshape(-1, 64), dim=-1
        )
        assert cos_sim.mean() > 0.90

    def test_sequential_updates(self):
        """Cache should grow as tokens are added."""
        cache = TurboQuantKVCache(bits=2, head_dim=64)
        for t in range(5):
            key = torch.randn(1, 4, 1, 64)
            value = torch.randn(1, 4, 1, 64)
            cache.update(key, value, layer_idx=0)
        k_out, v_out = cache.get(layer_idx=0)
        assert k_out.shape == (1, 4, 5, 64)
        assert v_out.shape == (1, 4, 5, 64)

    def test_multiple_layers(self):
        """Should handle multiple layers independently."""
        cache = TurboQuantKVCache(bits=2, head_dim=64)
        for layer in range(3):
            key = torch.randn(1, 4, 8, 64)
            value = torch.randn(1, 4, 8, 64)
            cache.update(key, value, layer_idx=layer)
        for layer in range(3):
            k, v = cache.get(layer_idx=layer)
            assert k.shape == (1, 4, 8, 64)

    def test_memory_smaller_than_fp16(self):
        """Quantized cache should use less memory than fp16."""
        cache = TurboQuantKVCache(bits=2, head_dim=128)
        key = torch.randn(1, 32, 1024, 128)
        value = torch.randn(1, 32, 1024, 128)
        fp16_bytes = key.nelement() * 2  # fp16 = 2 bytes
        cache.update(key, value, layer_idx=0)
        compressed_bytes = cache.key_memory_bytes(layer_idx=0)
        assert compressed_bytes < fp16_bytes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_kv_cache.py -v`
Expected: FAIL

- [ ] **Step 3: Implement kv_cache.py**

```python
# pytq/kv_cache.py
"""HuggingFace-compatible KV cache with TurboQuant key compression.

Implements the DynamicCache interface so it can be passed as past_key_values
to HuggingFace model.generate().
"""
import torch
import math
from transformers.cache_utils import DynamicCache
from pytq.quantize_mse import TurboQuantMSE
from pytq.outlier import OutlierConfig, OutlierQuantizer


class TurboQuantKVCache(DynamicCache):
    """KV cache that quantizes keys using TurboQuant while keeping values in fp16.

    Subclasses HuggingFace DynamicCache for compatibility with generate().
    """

    def __init__(
        self,
        bits: int | float = 2,
        head_dim: int = 128,
        outlier_config: OutlierConfig | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.bits = bits
        self.head_dim = head_dim
        self.outlier_config = outlier_config
        self.device = device
        self._quantizers: dict[tuple[int, int], TurboQuantMSE | OutlierQuantizer] = {}

        # Parallel storage for quantized keys (indexed by layer)
        self._quantized_keys: dict[int, list] = {}  # layer_idx -> list of qt per head

    def _get_quantizer(self, layer_idx: int, head_idx: int):
        key = (layer_idx, head_idx)
        if key not in self._quantizers:
            seed = layer_idx * 10000 + head_idx
            if self.outlier_config is not None:
                self._quantizers[key] = OutlierQuantizer(
                    self.head_dim, self.outlier_config, seed=seed, device=self.device,
                )
            else:
                self._quantizers[key] = TurboQuantMSE(
                    self.head_dim, int(self.bits), seed=seed, device=self.device,
                )
        return self._quantizers[key]

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add new key-value pairs. Returns (full_keys, full_values) for this layer.

        Matches the DynamicCache.update() signature expected by HuggingFace.
        """
        batch, n_heads, seq_len, head_dim = key.shape

        if layer_idx not in self._quantized_keys:
            self._quantized_keys[layer_idx] = []

        # Quantize keys per head
        qt_keys_per_head = []
        for h in range(n_heads):
            quantizer = self._get_quantizer(layer_idx, h)
            k_head = key[:, h, :, :]  # (batch, seq_len, head_dim)
            qt = quantizer.quantize(k_head)
            qt_keys_per_head.append(qt)
        self._quantized_keys[layer_idx].append(qt_keys_per_head)

        # Store values via parent DynamicCache (handles concat)
        # We store dummy keys in parent (zeros) and override retrieval
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(torch.zeros_like(key))
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], torch.zeros_like(key)], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=2)

        # Return dequantized keys + values for attention
        keys_deq, values = self.get(layer_idx)
        return keys_deq, values

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve dequantized keys and values."""
        all_keys = []
        for qt_keys_per_head in self._quantized_keys[layer_idx]:
            keys_this_step = []
            for h, qt in enumerate(qt_keys_per_head):
                quantizer = self._get_quantizer(layer_idx, h)
                k_head = quantizer.dequantize(qt)
                keys_this_step.append(k_head)
            keys_this_step = torch.stack(keys_this_step, dim=1)
            all_keys.append(keys_this_step)

        keys = torch.cat(all_keys, dim=2)
        values = self.value_cache[layer_idx]
        return keys, values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.value_cache):
            return 0
        return self.value_cache[layer_idx].shape[2]

    def key_memory_bytes(self, layer_idx: int) -> int:
        """Estimate compressed key memory in bytes for a layer."""
        total = 0
        for qt_keys_per_head in self._quantized_keys.get(layer_idx, []):
            for qt in qt_keys_per_head:
                # Theoretical: ceil(bits * dim / 8) bytes per token
                n_tokens = qt.indices.shape[0] * (qt.indices.shape[1] if qt.indices.dim() > 2 else 1)
                total += math.ceil(self.bits * self.head_dim / 8) * n_tokens
                total += qt.norm.nelement() * 4  # fp32 norms
        return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_kv_cache.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pytq/kv_cache.py tests/test_kv_cache.py
git commit -m "feat: TurboQuant KV cache wrapper for HuggingFace Transformers"
```

---

## Task 8: Update __init__.py Exports

**Files:**
- Modify: `pytq/__init__.py`

- [ ] **Step 1: Update __init__.py with all exports**

```python
from pytq.quantize_mse import TurboQuantMSE
from pytq.quantize_prod import TurboQuantProd
from pytq.outlier import OutlierConfig, OutlierQuantizer
from pytq.kv_cache import TurboQuantKVCache

__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "OutlierConfig",
    "OutlierQuantizer",
    "TurboQuantKVCache",
]
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add pytq/__init__.py
git commit -m "feat: update package exports with all public API"
```

---

## Task 9: Benchmark — Theoretical Distortion Validation

**Files:**
- Create: `benchmarks/bench_distortion.py`

- [ ] **Step 1: Implement bench_distortion.py**

```python
# benchmarks/bench_distortion.py
"""Benchmark 1: Validate TurboQuant distortion against theoretical bounds.

Reproduces Figures 1-3 from the paper:
- Error distribution histograms for MSE and prod quantizers
- Variance of IP error vs average inner product
- MSE and IP error vs theoretical bounds across bit-widths
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from pytq.quantize_mse import TurboQuantMSE
from pytq.quantize_prod import TurboQuantProd
from pytq.utils import mse_distortion, ip_distortion, mse_upper_bound, mse_lower_bound, ip_upper_bound


def run_distortion_benchmark(
    dims: list[int] = [64, 96, 128, 768, 1536, 3072],
    bits_list: list[int] = [1, 2, 3, 4],
    n_vectors: int = 10_000,
    n_queries: int = 1_000,
    output_dir: str = "results/distortion",
):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for dim in dims:
        print(f"\n=== Dimension {dim} ===")
        results[dim] = {}

        x = torch.randn(n_vectors, dim)
        x = x / x.norm(dim=-1, keepdim=True)  # unit norm
        y = torch.randn(n_queries, dim)
        y = y / y.norm(dim=-1, keepdim=True)

        for bits in bits_list:
            print(f"  bits={bits}...")
            entry = {}

            # MSE quantizer
            q_mse = TurboQuantMSE(dim=dim, bits=bits, seed=42)
            qt_mse = q_mse.quantize(x)
            x_hat_mse = q_mse.dequantize(qt_mse)
            mse_val = mse_distortion(x, x_hat_mse).item()
            mse_var, mse_bias = ip_distortion(x[:n_queries], x_hat_mse[:n_queries], y)

            entry["mse"] = {
                "mse_distortion": mse_val,
                "ip_variance": mse_var.item(),
                "ip_bias": mse_bias.item(),
                "mse_upper_bound": mse_upper_bound(bits),
                "mse_lower_bound": mse_lower_bound(bits),
            }

            # Prod quantizer (requires bits >= 2)
            if bits >= 2:
                q_prod = TurboQuantProd(dim=dim, bits=bits, seed=42)
                qt_prod = q_prod.quantize(x)
                x_hat_prod = q_prod.dequantize(qt_prod)
                prod_mse = mse_distortion(x, x_hat_prod).item()
                prod_var, prod_bias = ip_distortion(x[:n_queries], x_hat_prod[:n_queries], y)

                entry["prod"] = {
                    "mse_distortion": prod_mse,
                    "ip_variance": prod_var.item(),
                    "ip_bias": prod_bias.item(),
                    "ip_upper_bound": ip_upper_bound(bits, dim),
                }
            else:
                entry["prod"] = "N/A (TurboQuant_prod requires bits >= 2)"

            results[dim][bits] = entry
            print(f"    MSE: {mse_val:.6f} (bound: {mse_upper_bound(bits):.6f})")
            if bits >= 2:
                print(f"    Prod IP bias: {prod_bias.item():.6f}, variance: {prod_var.item():.6f}")
            else:
                print(f"    Prod: N/A (requires bits >= 2)")

    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: MSE vs bits with bounds
    _plot_mse_vs_bits(results, bits_list, dims, output_dir)
    _plot_ip_error_vs_bits(results, bits_list, dims, output_dir)
    print(f"\nResults saved to {output_dir}/")


def _plot_mse_vs_bits(results, bits_list, dims, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for dim in dims:
        mses = [results[dim][b]["mse"]["mse_distortion"] for b in bits_list]
        ax.plot(bits_list, mses, "o-", label=f"d={dim}")
    # Bounds
    upper = [mse_upper_bound(b) for b in bits_list]
    lower = [mse_lower_bound(b) for b in bits_list]
    ax.plot(bits_list, upper, "k--", label="Upper bound", linewidth=2)
    ax.plot(bits_list, lower, "k:", label="Lower bound", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Bit-width")
    ax.set_ylabel("MSE Distortion")
    ax.set_title("TurboQuant MSE vs Theoretical Bounds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "mse_vs_bits.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_ip_error_vs_bits(results, bits_list, dims, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for dim in dims:
        prod_bits = [b for b in bits_list if b >= 2]
        variances = [results[dim][b]["prod"]["ip_variance"] for b in prod_bits]
        ax.plot(prod_bits, variances, "o-", label=f"d={dim}")
    ax.set_yscale("log")
    ax.set_xlabel("Bit-width")
    ax.set_ylabel("IP Error Variance")
    ax.set_title("TurboQuant_prod Inner Product Error Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "ip_error_vs_bits.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vectors", type=int, default=10_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--output-dir", type=str, default="results/distortion")
    args = parser.parse_args()
    run_distortion_benchmark(n_vectors=args.n_vectors, n_queries=args.n_queries, output_dir=args.output_dir)
```

- [ ] **Step 2: Smoke test**

Run: `python -m benchmarks.bench_distortion --n-vectors 100 --n-queries 50`
Expected: Completes without error, produces `results/distortion/results.json` and PNG plots.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_distortion.py
git commit -m "feat: benchmark for theoretical distortion validation"
```

---

## Task 10: Benchmark — Memory Profiling

**Files:**
- Create: `benchmarks/bench_memory.py`

- [ ] **Step 1: Implement bench_memory.py**

```python
# benchmarks/bench_memory.py
"""Benchmark 2: Memory profiling of TurboQuant KV cache across models and bit-widths.

Measures: KV cache memory, peak process RSS, max context length for 10GB budget.
"""
import torch
import json
import os
import argparse
import psutil
import matplotlib.pyplot as plt
from pytq.kv_cache import TurboQuantKVCache


# Model configs: (name, n_layers, n_heads, head_dim, weight_size_gb)
MODEL_CONFIGS = {
    "llama-3.2-1b": {"n_layers": 16, "n_kv_heads": 8, "head_dim": 64, "weight_gb": 2.4},
    "phi-3-mini": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 96, "weight_gb": 7.6},
    "mistral-7b": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128, "weight_gb": 14.4},
    "llama-3.1-8b": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128, "weight_gb": 16.0},
}

BITS_LIST = [1, 2, 2.5, 3, 3.5, 4, 16]  # 16 = fp16 baseline; fractional via outlier split
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
AVAILABLE_RAM_GB = 10.0  # 16GB - ~6GB system


def estimate_kv_memory_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    bits: int | float,
) -> int:
    """Estimate KV cache memory in bytes.

    Keys: bits per coordinate * head_dim * seq_len * n_heads * n_layers + norms
    Values: fp16 = 2 bytes per element
    """
    if bits == 16:
        key_bytes = n_layers * n_kv_heads * seq_len * head_dim * 2  # fp16
    else:
        # Quantized: ceil(bits * head_dim / 8) bytes for packed indices + 4 bytes norm per token per head
        import math as _math
        index_bytes_per_token = _math.ceil(bits * head_dim / 8)
        key_bytes = n_layers * n_kv_heads * seq_len * (index_bytes_per_token + 4)

    value_bytes = n_layers * n_kv_heads * seq_len * head_dim * 2  # fp16
    return key_bytes + value_bytes


def find_max_context(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bits: int | float,
    weight_gb: float,
) -> int:
    """Find maximum context length that fits in available RAM."""
    available_bytes = (AVAILABLE_RAM_GB - weight_gb) * 1e9
    if available_bytes <= 0:
        return 0
    # Binary search
    lo, hi = 0, 1_000_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        mem = estimate_kv_memory_bytes(n_layers, n_kv_heads, head_dim, mid, bits)
        if mem <= available_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def run_memory_benchmark(output_dir: str = "results/memory"):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        print(f"\n=== {model_name} ===")
        results[model_name] = {}

        for bits in BITS_LIST:
            results[model_name][bits] = {}
            for seq_len in SEQ_LENGTHS:
                mem_bytes = estimate_kv_memory_bytes(
                    cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"], seq_len, bits
                )
                mem_gb = mem_bytes / 1e9
                results[model_name][bits][seq_len] = {
                    "kv_cache_gb": round(mem_gb, 4),
                    "total_gb": round(mem_gb + cfg["weight_gb"], 4),
                    "fits_16gb": (mem_gb + cfg["weight_gb"]) <= 16.0,
                    "fits_10gb_free": mem_gb <= (AVAILABLE_RAM_GB - cfg["weight_gb"]),
                }

            max_ctx = find_max_context(
                cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"], bits, cfg["weight_gb"]
            )
            results[model_name][bits]["max_context"] = max_ctx
            print(f"  {bits}-bit: max context = {max_ctx:,} tokens")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_memory_vs_seqlen(results, output_dir)
    _plot_max_context(results, output_dir)
    print(f"\nResults saved to {output_dir}/")


def _plot_memory_vs_seqlen(results, output_dir):
    for model_name in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        for bits in BITS_LIST:
            mem_values = [results[model_name][bits][s]["total_gb"]
                         for s in SEQ_LENGTHS]
            label = f"{bits}-bit" if bits < 16 else "fp16"
            ax.plot(SEQ_LENGTHS, mem_values, "o-", label=label)
        ax.axhline(y=16.0, color="red", linestyle="--", label="16GB limit")
        ax.axhline(y=AVAILABLE_RAM_GB, color="orange", linestyle="--", label=f"{AVAILABLE_RAM_GB}GB free")
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Total Memory (GB)")
        ax.set_title(f"{model_name}: Memory vs Sequence Length")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, f"{model_name}_memory.png"), dpi=150, bbox_inches="tight")
        plt.close()


def _plot_max_context(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(results.keys())
    x = range(len(model_names))
    width = 0.2
    for i, bits in enumerate(BITS_LIST):
        max_ctxs = [results[m][bits]["max_context"] for m in model_names]
        label = f"{bits}-bit" if bits < 16 else "fp16"
        ax.bar([xi + i * width for xi in x], max_ctxs, width, label=label)
    ax.set_xticks([xi + width * 1.5 for xi in x])
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel("Max Context Length (tokens)")
    ax.set_title("Maximum Context Length on 16GB MacBook Air")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(output_dir, "max_context.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/memory")
    args = parser.parse_args()
    run_memory_benchmark(output_dir=args.output_dir)
```

- [ ] **Step 2: Smoke test**

Run: `python -m benchmarks.bench_memory`
Expected: Produces `results/memory/results.json` and PNG plots.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_memory.py
git commit -m "feat: memory profiling benchmark for KV cache across models"
```

---

## Task 11: Benchmark — Quality (Perplexity)

**Files:**
- Create: `benchmarks/bench_quality.py`

- [ ] **Step 1: Implement bench_quality.py**

```python
# benchmarks/bench_quality.py
"""Benchmark 3: Quality evaluation — perplexity on WikiText-2 with TurboQuant KV cache.

Measures perplexity delta vs fp16 baseline for each model x bit-width.
MMLU evaluation is optional and can be enabled with --mmlu flag.
"""
import torch
import json
import os
import argparse
import math
from tqdm import tqdm


def compute_perplexity(
    model,
    tokenizer,
    dataset_text: str,
    cache_factory=None,
    max_length: int = 2048,
    stride: int = 512,
    device: str = "cpu",
) -> float:
    """Compute perplexity using sliding window approach.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: Tokenizer for the model.
        dataset_text: Full text to compute perplexity on.
        cache_factory: Callable that returns a fresh TurboQuantKVCache, or None for baseline.
        max_length: Window size for sliding window.
        stride: Stride for sliding window.
        device: Device to run on.
    """
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls = []
    for begin in tqdm(range(0, seq_len - max_length, stride), desc="Perplexity"):
        end = begin + max_length
        target_len = stride if begin > 0 else max_length

        input_chunk = input_ids[:, begin:end]
        target_chunk = input_chunk.clone()
        target_chunk[:, :-target_len] = -100

        with torch.no_grad():
            kwargs = {}
            if cache_factory is not None:
                # Fresh cache per window — DynamicCache subclass, so model uses it natively
                kwargs["past_key_values"] = cache_factory()

            outputs = model(input_chunk, labels=target_chunk, **kwargs)
            nlls.append(outputs.loss.item())

        if len(nlls) >= 50:  # cap for speed
            break

    return math.exp(sum(nlls) / len(nlls))


def run_quality_benchmark(
    model_name: str = "meta-llama/Llama-3.2-1B",
    bits_list: list[int] = [2, 3, 4],
    device: str = "cpu",
    output_dir: str = "results/quality",
):
    os.makedirs(output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.float()

    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    head_dim = model.config.head_dim if hasattr(model.config, "head_dim") else (
        model.config.hidden_size // model.config.num_attention_heads
    )

    results = {"model": model_name, "device": device, "benchmarks": {}}

    # Baseline (no quantization)
    print("\n--- Baseline (fp16) ---")
    ppl_baseline = compute_perplexity(model, tokenizer, text, device=device)
    results["benchmarks"]["fp16"] = {"perplexity": ppl_baseline}
    print(f"  Perplexity: {ppl_baseline:.2f}")

    # Quantized
    from pytq.kv_cache import TurboQuantKVCache
    for bits in bits_list:
        print(f"\n--- {bits}-bit ---")
        factory = lambda b=bits, hd=head_dim, d=device: TurboQuantKVCache(bits=b, head_dim=hd, device=d)
        ppl = compute_perplexity(model, tokenizer, text, cache_factory=factory, device=device)
        delta = ppl - ppl_baseline
        results["benchmarks"][f"{bits}bit"] = {
            "perplexity": ppl,
            "delta_vs_baseline": delta,
            "delta_pct": (delta / ppl_baseline) * 100,
        }
        print(f"  Perplexity: {ppl:.2f} (delta: {delta:+.2f}, {(delta/ppl_baseline)*100:+.1f}%)")

    with open(os.path.join(output_dir, f"{model_name.split('/')[-1]}_quality.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results/quality")
    args = parser.parse_args()
    run_quality_benchmark(model_name=args.model, bits_list=args.bits, device=args.device, output_dir=args.output_dir)
```

- [ ] **Step 2: Commit** (no smoke test — requires model download)

```bash
git add benchmarks/bench_quality.py
git commit -m "feat: quality benchmark — perplexity with TurboQuant KV cache"
```

---

## Task 12: Benchmark — Speed

**Files:**
- Create: `benchmarks/bench_speed.py`

- [ ] **Step 1: Implement bench_speed.py**

```python
# benchmarks/bench_speed.py
"""Benchmark 4: Speed — tokens/sec and attention latency with TurboQuant KV cache.

Measures prefill speed, decode speed, and raw attention latency on CPU and MPS.
"""
import torch
import time
import json
import os
import argparse
import matplotlib.pyplot as plt
from pytq.quantize_mse import TurboQuantMSE
from pytq.kv_cache import TurboQuantKVCache


def bench_quantize_latency(
    dims: list[int] = [64, 96, 128],
    bits_list: list[int] = [2, 3, 4],
    seq_len: int = 1024,
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = "cpu",
) -> dict:
    """Measure quantize + dequantize latency."""
    results = {}
    for dim in dims:
        results[dim] = {}
        for bits in bits_list:
            q = TurboQuantMSE(dim=dim, bits=bits, seed=42, device=device)
            x = torch.randn(1, seq_len, dim, device=device)

            # Warmup
            for _ in range(n_warmup):
                qt = q.quantize(x)
                _ = q.dequantize(qt)

            if device == "mps":
                torch.mps.synchronize()

            # Timed runs
            times_q = []
            times_dq = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                qt = q.quantize(x)
                if device == "mps":
                    torch.mps.synchronize()
                t1 = time.perf_counter()
                _ = q.dequantize(qt)
                if device == "mps":
                    torch.mps.synchronize()
                t2 = time.perf_counter()
                times_q.append(t1 - t0)
                times_dq.append(t2 - t1)

            results[dim][bits] = {
                "quantize_ms": sum(times_q) / len(times_q) * 1000,
                "dequantize_ms": sum(times_dq) / len(times_dq) * 1000,
                "total_ms": (sum(times_q) + sum(times_dq)) / len(times_q) * 1000,
            }
            print(f"  d={dim}, b={bits}: quant={results[dim][bits]['quantize_ms']:.2f}ms, "
                  f"dequant={results[dim][bits]['dequantize_ms']:.2f}ms")
    return results


def bench_kv_cache_throughput(
    head_dim: int = 128,
    n_heads: int = 8,
    bits_list: list[int] = [2, 3, 4],
    seq_lengths: list[int] = [128, 512, 2048],
    device: str = "cpu",
) -> dict:
    """Measure KV cache update + retrieval throughput."""
    results = {}
    for bits in bits_list:
        results[bits] = {}
        for seq_len in seq_lengths:
            cache = TurboQuantKVCache(bits=bits, head_dim=head_dim, device=device)
            key = torch.randn(1, n_heads, seq_len, head_dim, device=device)
            value = torch.randn(1, n_heads, seq_len, head_dim, device=device)

            t0 = time.perf_counter()
            cache.update(key, value, layer_idx=0)
            if device == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            _, _ = cache.get(layer_idx=0)
            if device == "mps":
                torch.mps.synchronize()
            t2 = time.perf_counter()

            results[bits][seq_len] = {
                "update_ms": (t1 - t0) * 1000,
                "retrieve_ms": (t2 - t1) * 1000,
                "tokens_per_sec": seq_len / (t1 - t0),
            }
            print(f"  b={bits}, seq={seq_len}: update={results[bits][seq_len]['update_ms']:.1f}ms, "
                  f"retrieve={results[bits][seq_len]['retrieve_ms']:.1f}ms, "
                  f"tok/s={results[bits][seq_len]['tokens_per_sec']:.0f}")
    return results


def run_speed_benchmark(device: str = "cpu", output_dir: str = "results/speed"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Speed Benchmark (device={device}) ===\n")

    print("--- Quantize/Dequantize Latency ---")
    latency_results = bench_quantize_latency(device=device)

    print("\n--- KV Cache Throughput ---")
    throughput_results = bench_kv_cache_throughput(device=device)

    results = {"device": device, "latency": latency_results, "throughput": throughput_results}

    with open(os.path.join(output_dir, f"speed_{device}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _plot_latency(latency_results, device, output_dir)
    print(f"\nResults saved to {output_dir}/")


def _plot_latency(latency_results, device, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    dims = list(latency_results.keys())
    bits_list = list(latency_results[dims[0]].keys())
    x = range(len(dims))
    width = 0.25
    for i, bits in enumerate(bits_list):
        totals = [latency_results[d][bits]["total_ms"] for d in dims]
        ax.bar([xi + i * width for xi in x], totals, width, label=f"{bits}-bit")
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([f"d={d}" for d in dims])
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Quantize+Dequantize Latency ({device})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(output_dir, f"latency_{device}.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results/speed")
    args = parser.parse_args()
    run_speed_benchmark(device=args.device, output_dir=args.output_dir)
```

- [ ] **Step 2: Smoke test**

Run: `python -m benchmarks.bench_speed --device cpu`
Expected: Produces `results/speed/speed_cpu.json` and PNG plots.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_speed.py
git commit -m "feat: speed benchmark — latency and throughput for TurboQuant"
```

---

## Task 13: Benchmark — End-to-End

**Files:**
- Create: `benchmarks/bench_e2e.py`
- Create: `benchmarks/run_all.py`

- [ ] **Step 1: Implement bench_e2e.py**

```python
# benchmarks/bench_e2e.py
"""Benchmark 5: End-to-end — 'Will it fit on 16GB MacBook Air?'

Combines memory estimation with actual model inference to produce
the final summary table answering the core project question.
"""
import torch
import json
import os
import time
import argparse
from benchmarks.bench_memory import MODEL_CONFIGS, estimate_kv_memory_bytes, find_max_context


def run_e2e_benchmark(
    models: list[str] | None = None,
    bits_list: list[int] = [2, 3, 4, 16],
    output_dir: str = "results/e2e",
):
    os.makedirs(output_dir, exist_ok=True)

    if models is None:
        models = list(MODEL_CONFIGS.keys())

    results = []

    for model_name in models:
        cfg = MODEL_CONFIGS[model_name]
        print(f"\n=== {model_name} ===")

        for bits in bits_list:
            max_ctx = find_max_context(
                cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"], bits, cfg["weight_gb"]
            )

            # Estimate memory at a practical context length
            practical_ctx = min(max_ctx, 8192)
            kv_mem = estimate_kv_memory_bytes(
                cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"], practical_ctx, bits
            )
            total_gb = kv_mem / 1e9 + cfg["weight_gb"]

            # Compression ratio vs fp16
            fp16_kv_mem = estimate_kv_memory_bytes(
                cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"], practical_ctx, 16
            )
            compression_ratio = fp16_kv_mem / max(kv_mem, 1)

            entry = {
                "model": model_name,
                "bits": bits,
                "weight_gb": cfg["weight_gb"],
                "max_context": max_ctx,
                "practical_context": practical_ctx,
                "kv_cache_gb": round(kv_mem / 1e9, 3),
                "total_gb": round(total_gb, 3),
                "fits_16gb": total_gb <= 16.0,
                "compression_ratio": round(compression_ratio, 2),
            }
            results.append(entry)

            bits_label = f"{bits}-bit" if bits < 16 else "fp16"
            status = "OK" if entry["fits_16gb"] else "EXCEEDS 16GB"
            print(f"  {bits_label}: max_ctx={max_ctx:>8,}  total={total_gb:.1f}GB  "
                  f"compress={compression_ratio:.1f}x  [{status}]")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Write markdown summary
    _write_markdown_summary(results, output_dir)
    print(f"\nResults saved to {output_dir}/")


def _write_markdown_summary(results, output_dir):
    lines = [
        "# TurboQuant End-to-End Results: 16GB MacBook Air\n",
        "| Model | Bit-width | Weight (GB) | KV Cache (GB) | Total (GB) | Max Context | Compression | Fits? |",
        "|-------|-----------|-------------|---------------|------------|-------------|-------------|-------|",
    ]
    for r in results:
        bits_label = f"{r['bits']}-bit" if r['bits'] < 16 else "fp16"
        fits = "Yes" if r["fits_16gb"] else "No"
        lines.append(
            f"| {r['model']} | {bits_label} | {r['weight_gb']} | {r['kv_cache_gb']} | "
            f"{r['total_gb']} | {r['max_context']:,} | {r['compression_ratio']}x | {fits} |"
        )

    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4, 16])
    parser.add_argument("--output-dir", type=str, default="results/e2e")
    args = parser.parse_args()
    run_e2e_benchmark(models=args.models, bits_list=args.bits, output_dir=args.output_dir)
```

- [ ] **Step 2: Implement run_all.py**

```python
# benchmarks/run_all.py
"""Run the full benchmark suite."""
import argparse
from benchmarks.bench_distortion import run_distortion_benchmark
from benchmarks.bench_memory import run_memory_benchmark
from benchmarks.bench_speed import run_speed_benchmark
from benchmarks.bench_e2e import run_e2e_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run all TurboQuant benchmarks")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true", help="Run with reduced vector counts")
    parser.add_argument("--with-quality", action="store_true",
                        help="Include quality benchmark (requires model download)")
    parser.add_argument("--quality-model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Model for quality benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("  TurboQuant (pytq) Full Benchmark Suite")
    print("=" * 60)

    n_vectors = 1000 if args.quick else 10_000
    total = 5 if args.with_quality else 4

    print(f"\n[1/{total}] Distortion Validation")
    run_distortion_benchmark(n_vectors=n_vectors, n_queries=min(n_vectors, 1000))

    print(f"\n[2/{total}] Memory Profiling")
    run_memory_benchmark()

    print(f"\n[3/{total}] Speed Benchmark")
    run_speed_benchmark(device=args.device)

    if args.with_quality:
        print(f"\n[4/{total}] Quality Benchmark")
        from benchmarks.bench_quality import run_quality_benchmark
        run_quality_benchmark(model_name=args.quality_model, device=args.device)

    print(f"\n[{total}/{total}] End-to-End Summary")
    run_e2e_benchmark()

    print("\n" + "=" * 60)
    print("  All benchmarks complete. Results in results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test**

Run: `python -m benchmarks.run_all --quick`
Expected: 4 benchmarks run. Quality can be included with `--with-quality --quality-model meta-llama/Llama-3.2-1B`.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_e2e.py benchmarks/run_all.py
git commit -m "feat: end-to-end benchmark and full suite runner"
```

---

## Task 14: Integration Test & Final Verification

**Files:**
- All files

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run quick benchmark suite**

Run: `python -m benchmarks.run_all --quick`
Expected: Completes successfully, produces results in `results/`

- [ ] **Step 3: Verify package installs**

Run: `pip install -e ".[dev]" && python -c "from pytq import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache; print('OK')"`
Expected: Prints "OK"

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: pytq v0.1.0 — TurboQuant implementation with benchmark suite"
```
