# pytq/rotation.py
"""Deterministic random orthogonal matrix generation via QR decomposition."""
from __future__ import annotations

import torch


def generate_rotation_matrix(
    dim: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs.unsqueeze(0)
    return Q.to(device)
