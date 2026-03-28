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
        dim = 128
        for bits in [1, 2, 3, 4]:
            q = TurboQuantMSE(dim=dim, bits=bits, seed=42)
            x = torch.randn(1000, dim)
            x = x / x.norm(dim=-1, keepdim=True)
            qt = q.quantize(x)
            x_hat = q.dequantize(qt)
            mse = mse_distortion(x, x_hat).item()
            bound = mse_upper_bound(bits)
            assert mse < bound * 1.5, (
                f"bits={bits}: MSE {mse:.4f} exceeds 1.5x bound {bound:.4f}"
            )

    def test_mse_decreases_with_bits(self):
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
        q = TurboQuantMSE(dim=128, bits=4, seed=42)
        x = torch.randn(100, 128) * 5.0
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        norm_orig = x.norm(dim=-1)
        norm_recon = x_hat.norm(dim=-1)
        ratio = norm_recon / norm_orig
        assert (ratio.mean() - 1.0).abs() < 0.2

    def test_deterministic(self):
        q = TurboQuantMSE(dim=64, bits=2, seed=42)
        x = torch.randn(10, 64)
        qt1 = q.quantize(x)
        qt2 = q.quantize(x)
        assert torch.equal(qt1.indices, qt2.indices)

    def test_batch_dims(self):
        q = TurboQuantMSE(dim=64, bits=2, seed=42)
        x = torch.randn(4, 8, 64)
        qt = q.quantize(x)
        assert qt.indices.shape == (4, 8, 64)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == (4, 8, 64)
