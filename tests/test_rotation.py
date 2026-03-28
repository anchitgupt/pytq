# tests/test_rotation.py
import torch
import pytest
from pytq.rotation import generate_rotation_matrix


class TestRotationMatrix:
    def test_orthogonal(self):
        R = generate_rotation_matrix(dim=128, seed=42)
        eye = torch.eye(128)
        assert torch.allclose(R.T @ R, eye, atol=1e-5)

    def test_determinant_one(self):
        R = generate_rotation_matrix(dim=64, seed=42)
        det = torch.linalg.det(R)
        assert abs(abs(det.item()) - 1.0) < 1e-4

    def test_deterministic_with_seed(self):
        R1 = generate_rotation_matrix(dim=128, seed=42)
        R2 = generate_rotation_matrix(dim=128, seed=42)
        assert torch.allclose(R1, R2)

    def test_different_seeds_differ(self):
        R1 = generate_rotation_matrix(dim=128, seed=42)
        R2 = generate_rotation_matrix(dim=128, seed=99)
        assert not torch.allclose(R1, R2)

    def test_shape(self):
        R = generate_rotation_matrix(dim=64, seed=0)
        assert R.shape == (64, 64)

    def test_preserves_norm(self):
        R = generate_rotation_matrix(dim=128, seed=42)
        x = torch.randn(128)
        y = R @ x
        assert torch.allclose(x.norm(), y.norm(), atol=1e-4)
