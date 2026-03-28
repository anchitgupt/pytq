# tests/test_kv_cache.py
import torch
import pytest
from pytq.kv_cache import TurboQuantKVCache


class TestTurboQuantKVCache:
    def test_update_and_retrieve(self):
        cache = TurboQuantKVCache(bits=2, head_dim=64)
        key = torch.randn(1, 4, 8, 64)
        value = torch.randn(1, 4, 8, 64)
        cache.update(key, value, layer_idx=0)
        k_out, v_out = cache.get(layer_idx=0)
        assert k_out.shape == key.shape
        assert v_out.shape == value.shape

    def test_values_stored_exactly(self):
        cache = TurboQuantKVCache(bits=2, head_dim=64)
        value = torch.randn(1, 4, 8, 64)
        key = torch.randn(1, 4, 8, 64)
        cache.update(key, value, layer_idx=0)
        _, v_out = cache.get(layer_idx=0)
        assert torch.allclose(v_out, value, atol=1e-3)

    def test_keys_approximately_preserved(self):
        cache = TurboQuantKVCache(bits=4, head_dim=64)
        key = torch.randn(1, 4, 8, 64)
        value = torch.randn(1, 4, 8, 64)
        cache.update(key, value, layer_idx=0)
        k_out, _ = cache.get(layer_idx=0)
        cos_sim = torch.nn.functional.cosine_similarity(
            key.reshape(-1, 64), k_out.reshape(-1, 64), dim=-1
        )
        assert cos_sim.mean() > 0.90

    def test_sequential_updates(self):
        cache = TurboQuantKVCache(bits=2, head_dim=64)
        for t in range(5):
            key = torch.randn(1, 4, 1, 64)
            value = torch.randn(1, 4, 1, 64)
            cache.update(key, value, layer_idx=0)
        k_out, v_out = cache.get(layer_idx=0)
        assert k_out.shape == (1, 4, 5, 64)
        assert v_out.shape == (1, 4, 5, 64)

    def test_multiple_layers(self):
        cache = TurboQuantKVCache(bits=2, head_dim=64)
        for layer in range(3):
            key = torch.randn(1, 4, 8, 64)
            value = torch.randn(1, 4, 8, 64)
            cache.update(key, value, layer_idx=layer)
        for layer in range(3):
            k, v = cache.get(layer_idx=layer)
            assert k.shape == (1, 4, 8, 64)

    def test_memory_smaller_than_fp16(self):
        cache = TurboQuantKVCache(bits=2, head_dim=128)
        key = torch.randn(1, 32, 1024, 128)
        value = torch.randn(1, 32, 1024, 128)
        fp16_bytes = key.nelement() * 2
        cache.update(key, value, layer_idx=0)
        compressed_bytes = cache.key_memory_bytes(layer_idx=0)
        assert compressed_bytes < fp16_bytes
