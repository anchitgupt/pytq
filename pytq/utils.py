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
