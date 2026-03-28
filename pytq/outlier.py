# pytq/outlier.py
"""Outlier channel detection and split bit-width allocation.

Channels with highest magnitude get more bits; remaining channels get fewer bits.
E.g., 2.5-bit = 25% outliers at 3 bits + 75% at 2 bits.
"""
import torch
from dataclasses import dataclass
from pytq.rotation import generate_rotation_matrix
from pytq.codebook import build_codebook
from pytq.utils import QuantizedTensor


@dataclass
class OutlierConfig:
    outlier_fraction: float = 0.25
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
    outlier_indices_map: torch.Tensor
    normal_indices_map: torch.Tensor
    dim: int
    norm: torch.Tensor

    @property
    def bits(self) -> float:
        return (self.outlier_qt.bits * len(self.outlier_indices_map)
                + self.normal_qt.bits * len(self.normal_indices_map)) / self.dim

    @property
    def indices(self) -> torch.Tensor:
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
        self.rotation = generate_rotation_matrix(dim, seed, device=device)
        self.codebook_outlier = build_codebook(dim, config.outlier_bits).to(device)
        self.codebook_normal = build_codebook(dim, config.normal_bits).to(device)

    def _select_outlier_channels(self, y: torch.Tensor) -> tuple:
        avg_magnitude = y.abs().mean(dim=0)
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
        y = x_hat @ self.rotation.T

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
