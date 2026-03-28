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
    log_norm = gammaln(dim / 2) - 0.5 * math.log(math.pi) - gammaln((dim - 1) / 2)
    exponent = (dim - 3) / 2.0
    log_pdf = log_norm + exponent * np.log(np.maximum(1 - x * x, 1e-300))
    return np.exp(log_pdf)


def _lloyd_max(pdf_values: np.ndarray, grid: np.ndarray, n_centroids: int,
               max_iter: int = 300) -> np.ndarray:
    weights = pdf_values / pdf_values.sum()
    centroids = np.linspace(grid[0], grid[-1], n_centroids)

    for _ in range(max_iter):
        dists = np.abs(grid[:, None] - centroids[None, :])
        assignments = np.argmin(dists, axis=1)

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
    n_centroids = 2 ** bits
    sigma = 1.0 / math.sqrt(dim)
    bound = min(1.0, 6 * sigma)
    grid = np.linspace(-bound, bound, grid_size)
    pdf_vals = _beta_pdf(grid, dim)
    centroids_np = _lloyd_max(pdf_vals, grid, n_centroids)
    centroids = torch.tensor(centroids_np, dtype=torch.float32)
    centroids.requires_grad_(False)
    return centroids.clone()
