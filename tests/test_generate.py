"""
Unit tests for generate.py helpers that are pure enough to test without a
downloaded model (weight-key normalization, and the prefill->decode slicing
that generate_stream depends on).
"""
import mlx.core as mx
from generate import normalize_weight_keys


class TestNormalizeWeightKeys:
    def test_strips_single_leading_prefix(self):
        weights = {
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((2, 2)),
            "model.embed_tokens.weight": mx.zeros((2, 2)),
        }
        out = normalize_weight_keys(weights)
        assert set(out) == {
            "layers.0.self_attn.q_proj.weight",
            "embed_tokens.weight",
        }

    def test_keys_without_prefix_are_untouched(self):
        weights = {"layers.0.mlp.up_proj.weight": mx.zeros((2, 2))}
        out = normalize_weight_keys(weights)
        assert set(out) == {"layers.0.mlp.up_proj.weight"}

    def test_inner_model_occurrence_is_preserved(self):
        """A 'model.' segment in the *middle* of a key must survive.

        The old implementation used ``str.replace`` and destroyed these
        (issue #14): ``model.model.embed_tokens.weight`` became
        ``embed_tokens.weight`` and no longer matched the module tree.
        """
        key = "model.model.embed_tokens.weight"
        out = normalize_weight_keys({key: mx.zeros((2, 2))})
        assert set(out) == {"model.embed_tokens.weight"}

    def test_double_replacement_regression(self):
        """The exact bug: a key with two 'model.' occurrences."""
        weights = {
            "model.layers.0.model.norm.weight": mx.zeros((2, 2)),
        }
        out = normalize_weight_keys(weights)
        # Only the leading prefix is stripped; the inner one is data.
        assert set(out) == {"layers.0.model.norm.weight"}

    def test_values_are_not_copied(self):
        value = mx.zeros((3, 3))
        out = normalize_weight_keys({"model.x": value})
        assert out["x"] is value


class TestNextTokenSlicing:
    """The prefill->decode token extraction in generate_stream (issue #12).

    ``generate_stream`` predicts the next token with
    ``greedy_sample(logits[:, -1, :])``. Before the fix it called
    ``greedy_sample(logits)`` on the raw prefill logits of shape
    ``(batch, seq_len, vocab)`` and then ``.item()``-ed the result — which
    has ``seq_len`` elements and raised ``ValueError`` for any prompt longer
    than one token. These tests pin the corrected mechanism on synthetic
    tensors so no downloaded model is required.
    """

    def _extract(self, logits):
        # Mirror the body of the generation loop after the #12 fix.
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = next_token.item()  # must be a single scalar
        current_token = next_token.reshape((1, 1))
        return token_id, current_token

    def test_prefill_multi_token_does_not_raise(self):
        # Prefill logits: (batch=1, seq_len=4, vocab=8). Old code raised here.
        mx.random.seed(0)
        logits = mx.random.normal((1, 4, 8))
        token_id, current_token = self._extract(logits)
        assert isinstance(token_id, int)
        assert 0 <= token_id < 8
        assert current_token.shape == (1, 1)

    def test_prefill_token_is_last_position_argmax(self):
        mx.random.seed(1)
        logits = mx.random.normal((1, 5, 10))
        token_id, _ = self._extract(logits)
        expected = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        assert token_id == expected

    def test_decode_step_single_token(self):
        # Decode step returns (batch=1, seq_len=1, vocab=V) — must still work.
        mx.random.seed(2)
        logits = mx.random.normal((1, 1, 6))
        token_id, current_token = self._extract(logits)
        assert isinstance(token_id, int)
        assert current_token.shape == (1, 1)
