import torch
import math
import pytest
from pytq.codebook import build_codebook


class TestBuildCodebook:
    def test_1bit_centroid_count(self):
        centroids = build_codebook(dim=128, bits=1)
        assert centroids.shape == (2,)

    def test_2bit_centroid_count(self):
        centroids = build_codebook(dim=128, bits=2)
        assert centroids.shape == (4,)

    def test_3bit_centroid_count(self):
        centroids = build_codebook(dim=128, bits=3)
        assert centroids.shape == (8,)

    def test_4bit_centroid_count(self):
        centroids = build_codebook(dim=128, bits=4)
        assert centroids.shape == (16,)

    def test_centroids_sorted(self):
        for bits in [1, 2, 3, 4]:
            centroids = build_codebook(dim=128, bits=bits)
            assert torch.all(centroids[1:] > centroids[:-1]), f"Not sorted for bits={bits}"

    def test_centroids_symmetric(self):
        for bits in [1, 2, 3, 4]:
            centroids = build_codebook(dim=128, bits=bits)
            assert torch.allclose(centroids, -centroids.flip(0), atol=1e-4), (
                f"Not symmetric for bits={bits}"
            )

    def test_1bit_centroid_values_high_dim(self):
        dim = 1024
        centroids = build_codebook(dim=dim, bits=1)
        expected = math.sqrt(2.0 / (math.pi * dim))
        assert abs(abs(centroids[1].item()) - expected) < 0.01 * expected

    def test_distortion_decreases_with_bits(self):
        dim = 128
        distortions = []
        for bits in [1, 2, 3, 4]:
            centroids = build_codebook(dim=dim, bits=bits)
            max_gap = (centroids[1:] - centroids[:-1]).max().item()
            distortions.append(max_gap)
        for i in range(len(distortions) - 1):
            assert distortions[i] > distortions[i + 1]

    def test_different_dims_give_different_codebooks(self):
        c64 = build_codebook(dim=64, bits=2)
        c256 = build_codebook(dim=256, bits=2)
        assert not torch.allclose(c64, c256)
