"""TurboQuant_prod: Inner-product-optimal quantization with QJL residual correction."""
import torch
import math
from dataclasses import dataclass
from pytq.quantize_mse import TurboQuantMSE
from pytq.utils import QuantizedTensor


@dataclass
class QuantizedTensorProd:
    """Compressed representation for TurboQuant_prod."""
    mse_qt: QuantizedTensor
    qjl_signs: torch.Tensor
    residual_norm: torch.Tensor
    bits: int
    dim: int

    @property
    def indices(self) -> torch.Tensor:
        return self.mse_qt.indices

    @property
    def norm(self) -> torch.Tensor:
        return self.mse_qt.norm


class TurboQuantProd:
    """Inner-product-optimal quantizer: MSE quantization + QJL residual correction.

    Uses (b-1) bits for MSE quantization and 1 bit for QJL on the residual.
    """

    def __init__(self, dim: int, bits: int, seed: int = 0, device: str = "cpu"):
        assert bits >= 2, "TurboQuant_prod requires at least 2 bits"
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = device

        self.mse_quantizer = TurboQuantMSE(dim, bits - 1, seed=seed, device=device)
        self._qjl_seed = seed + 1_000_000

    def _get_qjl_matrix(self, device: str = "cpu") -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self._qjl_seed)
        S = torch.randn(self.dim, self.dim, generator=gen, dtype=torch.float32)
        return S.to(device)

    def quantize(self, x: torch.Tensor) -> QuantizedTensorProd:
        mse_qt = self.mse_quantizer.quantize(x)
        x_mse = self.mse_quantizer.dequantize(mse_qt)

        residual = x - x_mse
        batch_shape = residual.shape[:-1]
        residual_flat = residual.reshape(-1, self.dim).to(self.device)

        residual_norm = residual_flat.norm(dim=-1)

        S = self._get_qjl_matrix(self.device)
        projected = residual_flat @ S.T
        signs = (projected >= 0).to(torch.uint8)

        return QuantizedTensorProd(
            mse_qt=mse_qt,
            qjl_signs=signs.reshape(*batch_shape, self.dim),
            residual_norm=residual_norm.reshape(*batch_shape),
            bits=self.bits,
            dim=self.dim,
        )

    def dequantize(self, qt: QuantizedTensorProd) -> torch.Tensor:
        x_mse = self.mse_quantizer.dequantize(qt.mse_qt)

        batch_shape = qt.qjl_signs.shape[:-1]
        signs_flat = qt.qjl_signs.reshape(-1, self.dim).float()
        r_norm_flat = qt.residual_norm.reshape(-1)

        z = 2.0 * signs_flat - 1.0

        S = self._get_qjl_matrix(self.device)
        scale = math.sqrt(math.pi / 2) / self.dim
        residual_hat = scale * r_norm_flat.unsqueeze(-1) * (z @ S)

        x_hat = x_mse + residual_hat.reshape(*batch_shape, self.dim)
        return x_hat
