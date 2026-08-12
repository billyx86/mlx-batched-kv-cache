"""
Unit tests for mlx-batched-kv-cache model components.
"""
import pytest
import mlx.core as mx
from model import RoPE, Attention, MLP, TransformerBlock, Transformer


def test_rope_initialization():
    rope = RoPE(dims=64)
    assert rope.dims == 64
    assert rope.inv_freq.shape[0] == 32


def test_attention_initialization():
    attn = Attention(
        dims=256,
        num_heads=4,
        num_kv_heads=2,
        qk_proj_group_size=64,
        qk_proj_bits=4,
        v_proj_group_size=64,
        v_proj_bits=4,
        o_proj_group_size=64,
        o_proj_bits=4
    )
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 2
    assert attn.head_dim == 64


def test_mlp_forward():
    mlp = MLP(dims=256, hidden_dims=512, group_size=64, bits=4)
    x = mx.random.normal((2, 10, 256))
    out = mlp(x)
    assert out.shape == (2, 10, 256)


def test_transformer_block_initialization():
    config = {
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-5,
        "quantization": {"group_size": 64, "bits": 4}
    }
    block = TransformerBlock(config)
    assert block is not None


def test_kv_cache_update():
    config = {
        "hidden_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "rms_norm_eps": 1e-5,
        "vocab_size": 1000,
        "num_hidden_layers": 1,
        "quantization": {"group_size": 64, "bits": 4}
    }
    model = Transformer(config)
    inputs = mx.array([[1, 2, 3]])
    logits, caches = model(inputs)
    assert logits.shape[0] == 1
    assert len(caches) == 1
    assert caches[0] is not None
