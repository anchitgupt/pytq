"""HuggingFace-compatible KV cache with TurboQuant key compression."""
import torch
import math
from pytq.quantize_mse import TurboQuantMSE
from pytq.outlier import OutlierConfig, OutlierQuantizer, OutlierQuantizedTensor
from pytq.utils import QuantizedTensor


class TurboQuantKVCache:
    """KV cache that quantizes keys using TurboQuant while keeping values in fp16.

    Can be used standalone or passed as past_key_values to HuggingFace models.
    """

    def __init__(
        self,
        bits=2,
        head_dim=128,
        outlier_config=None,
        device="cpu",
    ):
        self.bits = bits
        self.head_dim = head_dim
        self.outlier_config = outlier_config
        self.device = device
        self._quantizers = {}
        self._quantized_keys = {}  # layer_idx -> list of (list of qt per head)
        self._values = {}  # layer_idx -> list of value tensors

    def _get_quantizer(self, layer_idx, head_idx):
        key = (layer_idx, head_idx)
        if key not in self._quantizers:
            seed = layer_idx * 10000 + head_idx
            if self.outlier_config is not None:
                self._quantizers[key] = OutlierQuantizer(
                    self.head_dim, self.outlier_config, seed=seed, device=self.device,
                )
            else:
                self._quantizers[key] = TurboQuantMSE(
                    self.head_dim, int(self.bits), seed=seed, device=self.device,
                )
        return self._quantizers[key]

    def update(self, key, value, layer_idx, cache_kwargs=None):
        """Add new key-value pairs to the cache.

        Args:
            key: (batch, n_heads, seq_len, head_dim)
            value: (batch, n_heads, seq_len, head_dim)
            layer_idx: Which transformer layer.
            cache_kwargs: Unused, for HuggingFace compatibility.

        Returns:
            (dequantized_keys, values) for this layer.
        """
        batch, n_heads, seq_len, head_dim = key.shape

        if layer_idx not in self._quantized_keys:
            self._quantized_keys[layer_idx] = []
            self._values[layer_idx] = []

        # Quantize keys per head
        qt_keys_per_head = []
        for h in range(n_heads):
            quantizer = self._get_quantizer(layer_idx, h)
            k_head = key[:, h, :, :]  # (batch, seq_len, head_dim)
            qt = quantizer.quantize(k_head)
            qt_keys_per_head.append(qt)
        self._quantized_keys[layer_idx].append(qt_keys_per_head)
        self._values[layer_idx].append(value)

        return self.get(layer_idx)

    def get(self, layer_idx):
        """Retrieve dequantized keys and values for a given layer."""
        all_keys = []
        for qt_keys_per_head in self._quantized_keys[layer_idx]:
            keys_this_step = []
            for h, qt in enumerate(qt_keys_per_head):
                quantizer = self._get_quantizer(layer_idx, h)
                k_head = quantizer.dequantize(qt)
                keys_this_step.append(k_head)
            # keys_this_step: list of (batch, seq_len, head_dim)
            # stack along head dim -> (batch, n_heads, seq_len, head_dim)
            keys_this_step = torch.stack(keys_this_step, dim=1)
            all_keys.append(keys_this_step)

        keys = torch.cat(all_keys, dim=2)  # concat along seq_len
        values = torch.cat(self._values[layer_idx], dim=2)
        return keys, values

    def get_seq_length(self, layer_idx=0):
        """Return the current sequence length cached for a layer."""
        if layer_idx not in self._values or not self._values[layer_idx]:
            return 0
        return sum(v.shape[2] for v in self._values[layer_idx])

    def __len__(self):
        """Return the number of layers in the cache."""
        return len(self._values)

    def key_memory_bytes(self, layer_idx):
        """Estimate compressed key memory in bytes for a layer.

        Uses ceil(bits * head_dim / 8) per token for packed index storage,
        plus 4 bytes per token for the fp32 norm.
        """
        total = 0
        for qt_keys_per_head in self._quantized_keys.get(layer_idx, []):
            for qt in qt_keys_per_head:
                if isinstance(qt, OutlierQuantizedTensor):
                    # For outlier quantizer, count tokens from norm
                    n_tokens = qt.norm.nelement()
                    eff_bits = qt.bits
                else:
                    # For standard QuantizedTensor
                    n_tokens = qt.norm.nelement()
                    eff_bits = qt.bits
                total += math.ceil(eff_bits * self.head_dim / 8) * n_tokens
                # Norm storage: fp32 = 4 bytes each
                total += n_tokens * 4
        return total
