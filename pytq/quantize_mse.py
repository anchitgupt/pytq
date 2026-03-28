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
        orig_shape = x.shape
        assert orig_shape[-1] == self.dim
        x_flat = x.reshape(-1, self.dim).to(self.device)

        # Step 1: Normalize
        norms = x_flat.norm(dim=-1)
        safe_norms = norms.clamp(min=1e-8)
        x_hat = x_flat / safe_norms.unsqueeze(-1)

        # Step 2: Rotate
        y = x_hat @ self.rotation.T

        # Step 3: Scalar quantize
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
        batch_shape = qt.indices.shape[:-1]
        indices_flat = qt.indices.reshape(-1, self.dim).long()
        norms_flat = qt.norm.reshape(-1)

        # Step 1: Lookup centroids
        y_tilde = self.codebook[indices_flat]

        # Step 2: Inverse rotate
        x_hat_tilde = y_tilde @ self.rotation

        # Step 3: Rescale
        x_tilde = x_hat_tilde * norms_flat.unsqueeze(-1)

        return x_tilde.reshape(*batch_shape, self.dim)
