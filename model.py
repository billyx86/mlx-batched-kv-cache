import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple, List, Dict

from mlx.nn import QuantizedLinear


class RoPE(nn.Module):
    def __init__(self, dims: int, traditional: bool = False, base: float = 10000.0):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self._compute_inv_freq()

    def _compute_inv_freq(self):
        self.inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims))

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        seq_len = x.shape[2]
        positions = mx.arange(offset, offset + seq_len, dtype=self.inv_freq.dtype)
        freqs = mx.outer(positions, self.inv_freq)
        freqs = freqs.reshape(1, 1, seq_len, -1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos_emb = mx.cos(emb)
        sin_emb = mx.sin(emb)

        if self.traditional:
            x1 = x[..., : self.dims // 2]
            x2 = x[..., self.dims // 2 :]
            rotated_x = mx.concatenate([-x2, x1], axis=-1)
            return (x * cos_emb) + (rotated_x * sin_emb)
        else:
             # Simplified non-traditional RoPE application using complex numbers
            x_complex = mx.view_as_complex(x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2))
            emb_complex = mx.view_as_complex(emb.astype(mx.float32).reshape(*emb.shape[:-1], -1, 2))
            emb_complex = mx.exp(1j * emb_complex) # exp(i * theta) = cos(theta) + i * sin(theta)
            rotated_x_complex = x_complex * emb_complex
            rotated_x = mx.view_as_real(rotated_x_complex).reshape(*x.shape[:-1], -1)
            return rotated_x.astype(x.dtype)


class Attention(nn.Module):
    """ Renamed internal layers to q_proj, k_proj, v_proj, o_proj """
    def __init__(self, dims: int, num_heads: int, num_kv_heads: int, qk_proj_group_size: int, qk_proj_bits: int, v_proj_group_size: int, v_proj_bits: int, o_proj_group_size: int, o_proj_bits: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = QuantizedLinear(dims, num_heads * self.head_dim, group_size=qk_proj_group_size, bits=qk_proj_bits, bias=True)
        self.k_proj = QuantizedLinear(dims, num_kv_heads * self.head_dim, group_size=qk_proj_group_size, bits=qk_proj_bits, bias=True)
        self.v_proj = QuantizedLinear(dims, num_kv_heads * self.head_dim, group_size=v_proj_group_size, bits=v_proj_bits, bias=True)
        self.o_proj = QuantizedLinear(num_heads * self.head_dim, dims, group_size=o_proj_group_size, bits=o_proj_bits, bias=True)

    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[Tuple[mx.array, mx.array]] = None,
                 rope: RoPE = None,
                 offset: int = 0
                 ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        batch_size, seq_len, dims = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if rope is None: raise ValueError("RoPE module must be provided.")
        queries = rope(queries, offset=offset)
        keys = rope(keys, offset=offset)

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)

        updated_key_cache = keys
        updated_value_cache = values

        if self.num_kv_heads < self.num_heads:
            num_repeats = self.num_heads // self.num_kv_heads
            keys = mx.repeat(keys, repeats=num_repeats, axis=1)
            values = mx.repeat(values, repeats=num_repeats, axis=1)

        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None: scores = scores + mask
        attention_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        attention_output = attention_weights @ values
        output_concat = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        output = self.o_proj(output_concat)

        return output, (updated_key_cache, updated_value_cache)


class MLP(nn.Module):
    """ MLP layer. """
    def __init__(self, dims: int, hidden_dims: int, group_size: int, bits: int):
        super().__init__()
        self.gate_proj = QuantizedLinear(dims, hidden_dims, group_size=group_size, bits=bits, bias=True)
        self.down_proj = QuantizedLinear(hidden_dims, dims, group_size=group_size, bits=bits, bias=True)
        self.up_proj = QuantizedLinear(dims, hidden_dims, group_size=group_size, bits=bits, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        dims = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        mlp_dims = config["intermediate_size"]
        norm_eps = config["rms_norm_eps"]

        # Read global quantization settings from config
        quant_config = config.get("quantization", {})
        if not quant_config:
            raise ValueError("Quantization config not found in model config.")
        
        global_group_size = quant_config.get("group_size")
        global_bits = quant_config.get("bits")

        if global_group_size is None or global_bits is None:
             raise ValueError("Global 'group_size' and 'bits' not found in quantization config.")

        # Use global settings for all quantized layers in the block
        attn_qk_group_size = global_group_size
        attn_qk_bits = global_bits
        attn_v_group_size = global_group_size
        attn_v_bits = global_bits
        attn_o_group_size = global_group_size
        attn_o_bits = global_bits
        mlp_group_size = global_group_size
        mlp_bits = global_bits

        self.input_layernorm = nn.RMSNorm(dims, eps=norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(dims, eps=norm_eps)

        self.self_attn = Attention(
             dims, num_heads, num_kv_heads,
             attn_qk_group_size, attn_qk_bits,
             attn_v_group_size, attn_v_bits,
             attn_o_group_size, attn_o_bits
        )
        self.mlp = MLP(dims=dims, hidden_dims=mlp_dims, group_size=mlp_group_size, bits=mlp_bits)

        # Debug print for quantization config used
        # print(f"Block Quant Config: attn_qk={attn_qk_bits}bit/{attn_qk_group_size}g, "
        #       f"attn_v={attn_v_bits}bit/{attn_v_group_size}g, attn_o={attn_o_bits}bit/{attn_o_group_size}g, "
        #       f"mlp={mlp_bits}bit/{mlp_group_size}g")

    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[Tuple[mx.array, mx.array]] = None,
                 rope: RoPE = None,
                 offset: int = 0
                 ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # Use the renamed norm attributes
        residual = x
        h = self.input_layernorm(x) # NORM 1 (Applied before Self-Attention)

        attn_output, updated_cache = self.self_attn(h, mask=mask, cache=cache, rope=rope, offset=offset)
        h = residual + attn_output # RESIDUAL 1

        residual = h
        h_norm = self.post_attention_layernorm(h) # NORM 2 (Applied before MLP)
        ffn_output = self.mlp(h_norm)
        out = residual + ffn_output # RESIDUAL 2

        return out, updated_cache


class Transformer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        dims = config["hidden_size"]
        rope_base = config.get("rope_theta", 10000.0)
        head_dim = dims // config["num_attention_heads"]
        quant_config = config.get("quantization", {}) # Get quant config

        self.embed_tokens = nn.Embedding(self.vocab_size, dims)

        self.rope = RoPE(head_dim, traditional=True, base=rope_base) # Use traditional RoPE

        # Ensure TransformerBlock receives the config dict
        self.layers = [
            TransformerBlock(config=config)
            for _ in range(self.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(dims, eps=config["rms_norm_eps"])

        # Read global quantization settings from config for lm_head
        if quant_config:
            global_group_size = quant_config.get("group_size")
            global_bits = quant_config.get("bits")
            if global_group_size is None or global_bits is None:
                 raise ValueError("Global 'group_size' and 'bits' not found in quantization config for lm_head.")
            self.lm_head = QuantizedLinear(dims, self.vocab_size, group_size=global_group_size, bits=global_bits, bias=True)
        else:
             self.lm_head = nn.Linear(dims, self.vocab_size, bias=False)


    def __call__(self,
                 inputs: mx.array,
                 past_kv_caches: Optional[List[Optional[Tuple[mx.array, mx.array]]]] = None,
                 mask: Optional[mx.array] = None
                 ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:

        h = self.embed_tokens(inputs)
        offset = 0
        if past_kv_caches is not None and past_kv_caches[0] is not None:
            offset = past_kv_caches[0][0].shape[2]

        if past_kv_caches is None: past_kv_caches = [None] * self.num_hidden_layers
        if len(past_kv_caches) != self.num_hidden_layers:
             raise ValueError(f"Incorrect num caches. Expected {self.num_hidden_layers}, got {len(past_kv_caches)}")

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            try:
                h, updated_cache = layer(h, mask=mask, cache=past_kv_caches[i], rope=self.rope, offset=offset)
            except Exception as e:
                print(f"Error in layer {i}: {e}")
                raise e
            new_kv_caches.append(updated_cache)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, new_kv_caches
