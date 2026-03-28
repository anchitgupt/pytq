# tests/test_quantize.py
import torch
import math
import pytest
from pytq.quantize_mse import TurboQuantMSE
from pytq.utils import mse_distortion, mse_upper_bound
from pytq.quantize_prod import TurboQuantProd
from pytq.utils import ip_distortion


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


class TestTurboQuantProd:
    def test_quantize_returns_result(self):
        q = TurboQuantProd(dim=128, bits=2, seed=42)
        x = torch.randn(10, 128)
        result = q.quantize(x)
        assert result.indices.shape == (10, 128)

    def test_inner_product_unbiased(self):
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
