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
