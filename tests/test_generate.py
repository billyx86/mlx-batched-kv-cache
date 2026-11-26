"""
Unit tests for generate.py helpers that are pure enough to test without a
downloaded model (weight-key normalization, and the prefill->decode slicing
that generate_stream depends on).
"""
import numpy as np
import pytest
import mlx.core as mx
from generate import (
    generate_stream,
    greedy_sample,
    normalize_weight_keys,
    temperature_sample,
)


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
        token_id, token_tensor = self._extract(logits)
        assert isinstance(token_id, int)
        assert token_tensor.shape == (1, 1)


class TestTemperatureSample:
    """Real temperature sampling (issue #16).

    temperature_sample must be a proper sampler: draw from
    softmax(logits / T), reproducible under a fixed mx.random seed, and
    collapse to greedy / uniform at the low / high temperature limits.
    """

    def _batched_draws(self, target: mx.array, temperature: float, n: int):
        """n independent draws from softmax(target / temperature).

        Repeats the same (1, 1, vocab) logits row n times along the batch
        axis and calls temperature_sample once — one MLX op for n draws, so
        these stay fast on the CI runner. Each row is an independent sample
        from the identical distribution, which is exactly what we want to
        estimate the distribution.
        """
        logits = mx.repeat(target[None, :, :], n, axis=0)  # (n, 1, vocab)
        return temperature_sample(logits, temperature).reshape(-1)

    def test_rejects_nonpositive_temperature(self):
        logits = mx.random.normal((1, 1, 8))
        with pytest.raises(ValueError):
            temperature_sample(logits, 0.0)
        with pytest.raises(ValueError):
            temperature_sample(logits, -1.0)

    def test_fixed_seed_reproducible(self):
        """Same seed -> identical sampled tokens, two independent draws."""
        logits = mx.random.normal((1, 1, 32))
        mx.random.seed(123)
        first = temperature_sample(logits, 1.0).reshape(-1).tolist()
        mx.random.seed(123)
        second = temperature_sample(logits, 1.0).reshape(-1).tolist()
        assert first == second

    def test_different_seeds_can_differ(self):
        """Two seeds over 200 batched draws from a flat-ish distribution differ."""
        logits = mx.random.normal((1, 200, 64))
        mx.random.seed(1)
        a = temperature_sample(logits, 1.0).reshape(-1).tolist()
        mx.random.seed(2)
        b = temperature_sample(logits, 1.0).reshape(-1).tolist()
        assert a != b

    def test_matches_reference_softmax_distribution(self):
        """Empirical frequencies over many draws match softmax(logits / T)."""
        vocab = 8
        temp = 0.7
        n = 20_000
        mx.random.seed(99)
        target = mx.random.normal((1, vocab))  # (1, vocab)
        draws = self._batched_draws(target, temp, n)
        onehot = mx.equal(
            mx.arange(vocab)[None, :], draws.astype(mx.int32)[:, None]
        ).astype(mx.float32)
        empirical = onehot.sum(axis=0) / n
        expected = mx.softmax(target[0] / temp)
        assert mx.allclose(empirical, expected, atol=0.02).item(), (
            empirical.tolist(),
            expected.tolist(),
        )

    def test_low_temperature_tends_to_argmax(self):
        """T -> 0 concentrates almost all mass on the max-logit token."""
        target = mx.array([[0.0, 1.0, 5.0, 1.0, 0.0]])  # (1, vocab)
        mx.random.seed(3)
        draws = self._batched_draws(target, 0.05, 200)
        assert int((draws == 2).sum()) >= 190

    def test_high_temperature_is_near_uniform(self):
        """T -> inf flattens the distribution over all tokens."""
        target = mx.array([[0.0, 0.0, 10.0, 0.0]])  # (1, vocab=4)
        n = 4000
        mx.random.seed(11)
        draws = self._batched_draws(target, 100.0, n)
        for tok in range(4):
            frac = float((draws == tok).sum()) / n
            assert 0.15 <= frac <= 0.35, (tok, frac)

    def test_output_shape_matches_greedy(self):
        """(batch, 1, vocab) in -> (batch, 1) out, like greedy_sample."""
        logits = mx.random.normal((1, 1, 16))
        mx.random.seed(5)
        sampled = temperature_sample(logits, 1.0)
        greedy = greedy_sample(logits)
        assert sampled.shape == greedy.shape
        # The loop downstream does .item() then .reshape((1, 1)).
        assert int(sampled.item()) == sampled.reshape((1, 1)).item()

    def test_wired_into_generate_stream_path(self):
        """generate_stream's decode branch must call temperature_sample, not greedy.

        Pins the #16 fix: at temperature > 0 the loop no longer falls through
        to greedy_sample. We can't run the model here, but we can assert the
        sampling decision in the source uses the temperature branch.
        """
        import inspect

        import generate

        src = inspect.getsource(generate.generate_stream)
        assert "temperature_sample(logits[:, -1, :], temperature)" in src
        # The old placeholder must be gone.
        assert "Placeholder for temperature sampling" not in src


class _StubModel:
    """Minimal model stub returning (1, T, vocab) logits and echoing caches."""

    def __init__(self, vocab: int = 4, target_token: int = 2):
        self.vocab = vocab
        self.target_token = target_token

    def __call__(self, inputs: mx.array, past_kv_caches=None):
        batch, seq_len = inputs.shape
        one_hot = (
            mx.equal(mx.arange(self.vocab), self.target_token).astype(mx.float32)
            * 10.0
        )
        logits = mx.broadcast_to(
            one_hot.reshape(1, 1, self.vocab), (batch, seq_len, self.vocab)
        )
        kv_caches = (
            past_kv_caches
            if past_kv_caches is not None
            else [mx.zeros((1, 1, seq_len, 1))]
        )
        return logits, kv_caches


class _StubTokenizer:
    """Minimal tokenizer stub supporting encode(..., return_tensors='np') and decode."""

    eos_token_id = 0

    def __init__(self, encoded=None):
        self._encoded = (
            encoded if encoded is not None else np.array([[1, 2]], dtype=np.int32)
        )
        self._vocab = {1: "hello", 2: " world", 3: "!"}

    def encode(self, prompt, add_special_tokens=False, return_tensors="np"):
        return self._encoded

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self._vocab[t] for t in token_ids)


class TestGenerateStream:
    """Silence by default, verbose progress output, and empty-prompt guard (#20, #23)."""

    def test_no_stdout_by_default(self, capsys):
        model = _StubModel()
        tokenizer = _StubTokenizer()
        tokens = list(generate_stream(model, tokenizer, "hello", 3))
        assert len(tokens) == 3
        assert capsys.readouterr().out == ""

    def test_verbose_prints_progress(self, capsys):
        model = _StubModel()
        tokenizer = _StubTokenizer()
        tokens = list(generate_stream(model, tokenizer, "hello", 3, verbose=True))
        assert len(tokens) == 3
        out = capsys.readouterr().out
        assert "Encoding prompt..." in out
        assert "Generating tokens..." in out

    def test_empty_prompt_still_raises(self):
        model = _StubModel()
        tokenizer = _StubTokenizer(encoded=np.empty((1, 0), dtype=np.int32))
        with pytest.raises(ValueError):
            list(generate_stream(model, tokenizer, "", 3))
