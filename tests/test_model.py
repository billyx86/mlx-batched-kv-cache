"""
Unit tests for mlx-batched-kv-cache model components.
"""
import pytest
import mlx.core as mx
from model import (
    RoPE,
    Attention,
    MLP,
    TransformerBlock,
    Transformer,
    make_causal_mask,
    make_causal_mask_with_prefix,
)


def make_test_config(num_kv_heads=2):
    return {
        "hidden_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": num_kv_heads,
        "intermediate_size": 256,
        "rms_norm_eps": 1e-5,
        "vocab_size": 1000,
        "num_hidden_layers": 1,
        "quantization": {"group_size": 64, "bits": 4}
    }


def test_rope_initialization():
    rope = RoPE(dims=64)
    assert rope.dims == 64
    assert rope.inv_freq.shape[0] == 32


def test_attention_initialization():
    attn = Attention(dims=256, num_heads=4, num_kv_heads=2, group_size=64, bits=4)
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


def test_make_causal_mask():
    mask = make_causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    m = mask[0, 0]
    # On and below the diagonal: unmasked (0). Above: -inf.
    for i in range(4):
        for j in range(4):
            if j <= i:
                assert m[i, j].item() == 0.0
            else:
                assert m[i, j].item() == float("-inf")


def test_prefill_is_causal():
    """Logits at position 0 of a full prefill must not depend on later tokens."""
    mx.random.seed(0)
    model = Transformer(make_test_config())
    mx.eval(model.parameters())

    seq = mx.array([[1, 2, 3, 4, 5]])
    logits_full, _ = model(seq)
    logits_first, _ = model(seq[:, :1])
    mx.eval(logits_full, logits_first)

    assert mx.allclose(logits_full[:, 0], logits_first, atol=1e-4).item()


def test_kv_cache_matches_full_sequence():
    """Step-by-step generation with KV caches must match a single full forward pass."""
    mx.random.seed(0)
    model = Transformer(make_test_config())
    mx.eval(model.parameters())

    seq = mx.array([[1, 2, 3, 4, 5, 6]])
    logits_full, _ = model(seq)

    logits_cached, caches = model(seq[:, :1])
    for t in range(1, seq.shape[1]):
        logits_cached, caches = model(seq[:, t:t + 1], past_kv_caches=caches)
    mx.eval(logits_full, logits_cached)

    assert mx.allclose(logits_full[:, -1], logits_cached, atol=1e-3).item()


def test_gqa_matches_repeat_reference():
    """Broadcast GQA must produce the same output as materializing repeated KV heads."""
    mx.random.seed(0)
    dims, num_heads, num_kv_heads, seq_len = 128, 4, 2, 5
    attn = Attention(dims, num_heads, num_kv_heads, group_size=64, bits=4)
    rope = RoPE(dims // num_heads)
    x = mx.random.normal((1, seq_len, dims))

    out, _ = attn(x, rope=rope)

    # Reference implementation using mx.repeat
    head_dim = dims // num_heads
    q = attn.q_proj(x).reshape(1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = attn.k_proj(x).reshape(1, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(1, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    q, k = rope(q), rope(k)
    k = mx.repeat(k, num_heads // num_kv_heads, axis=1)
    v = mx.repeat(v, num_heads // num_kv_heads, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) * attn.scale
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
    ref = attn.o_proj((weights @ v).transpose(0, 2, 1, 3).reshape(1, seq_len, -1))

    assert mx.allclose(out, ref, atol=1e-4).item()


def test_gqa_forward():
    """Forward pass with num_kv_heads < num_heads (grouped query attention)."""
    mx.random.seed(0)
    model = Transformer(make_test_config(num_kv_heads=1))
    mx.eval(model.parameters())

    inputs = mx.array([[1, 2, 3, 4]])
    logits, caches = model(inputs)
    assert logits.shape == (1, 4, 1000)
    # KV cache keeps only the single KV head, not the repeated one
    k, v = caches[0]
    assert k.shape[1] == 1
    assert v.shape[1] == 1


class TestCausalMaskWithPrefix:
    """make_causal_mask_with_prefix: the chunked-prefill mask (issue #13)."""

    def test_shape_is_new_by_total(self):
        for new_len, past_len in [(3, 4), (1, 1), (5, 0), (2, 7)]:
            mask = make_causal_mask_with_prefix(new_len, past_len)
            assert mask.shape == (1, 1, new_len, new_len + past_len)

    def test_zero_prefix_equals_plain_causal(self):
        for T in (1, 2, 3, 5):
            m = make_causal_mask_with_prefix(T, 0)
            p = make_causal_mask(T)
            assert mx.allclose(m, p).item()

    def test_pattern_allows_past_blocks_future(self):
        new_len, past_len = 3, 4
        mask = make_causal_mask_with_prefix(new_len, past_len)
        m = mask[0, 0]  # (new_len, new_len + past_len)
        for i in range(new_len):
            abs_pos = past_len + i
            for col in range(new_len + past_len):
                if col <= abs_pos:
                    assert m[i, col].item() == 0.0, (i, col)
                else:
                    assert m[i, col].item() == float("-inf"), (i, col)

    def test_cached_prefix_columns_always_allowed(self):
        mask = make_causal_mask_with_prefix(5, 6)
        prefix = mask[0, 0, :, :6]
        assert mx.all(prefix == 0.0).item()


class TestChunkedPrefill:
    """A chunked prefill through the public API must match a full forward pass.

    Before issue #13 was fixed, passing a multi-token input together with an
    existing cache applied *no* mask, so the new chunk attended to its own
    future — a causality violation. These tests pin the corrected behavior.
    """

    def test_chunked_prefill_matches_full_forward(self):
        """Process [1,2] then [3,4,5,6] via the cache == process all at once."""
        mx.random.seed(0)
        model = Transformer(make_test_config())
        mx.eval(model.parameters())

        seq = mx.array([[1, 2, 3, 4, 5, 6]])
        logits_full, _ = model(seq)

        # Chunk 1: prefill the first two tokens (no cache yet).
        _, caches = model(seq[:, :2])
        # Chunk 2: the remaining four tokens against the existing cache.
        logits_chunked, _ = model(seq[:, 2:], past_kv_caches=caches)
        mx.eval(logits_full, logits_chunked)

        # Chunk 2's per-position logits must equal the corresponding rows of
        # the full prefill (positions 2..5).
        assert mx.allclose(logits_chunked, logits_full[:, 2:], atol=1e-3).item()

    def test_chunked_single_token_decode_still_matches(self):
        """The plain per-token decode path must be unaffected by the fix."""
        mx.random.seed(0)
        model = Transformer(make_test_config())
        mx.eval(model.parameters())

        seq = mx.array([[1, 2, 3, 4, 5]])
        logits_full, _ = model(seq)

        _, caches = model(seq[:, :1])
        for t in range(1, seq.shape[1]):
            logits_step, caches = model(seq[:, t:t + 1], past_kv_caches=caches)
        mx.eval(logits_full, logits_step)

        assert mx.allclose(logits_full[:, -1], logits_step, atol=1e-3).item()

    def test_chunked_prefill_first_token_is_causal(self):
        """The first token of a new chunk must NOT see the chunk's later tokens.

        Position 0 of chunk 2 (absolute position 2) depends only on tokens
        [1, 2] — not on [3, 4, 5, 6] that are present in the same chunk.
        """
        mx.random.seed(0)
        model = Transformer(make_test_config())
        mx.eval(model.parameters())

        # logits for absolute position 2 when processed as part of chunk 2.
        _, caches = model(mx.array([[1, 2]]))
        logits_chunked, _ = model(mx.array([[3, 4, 5, 6]]), past_kv_caches=caches)

        # logits for absolute position 2 when [1,2,3] is prefilled whole.
        logits_ref, _ = model(mx.array([[1, 2, 3]]))
        mx.eval(logits_chunked, logits_ref)

        assert mx.allclose(logits_chunked[:, 0], logits_ref[:, 2], atol=1e-4).item()
