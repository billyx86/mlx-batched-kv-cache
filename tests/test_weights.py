"""
Unit tests for weights.py — sharded-checkpoint resolution and shard merging.

The file-resolution and dict-merge logic is pure Python, so these run on any
platform (including the macOS CI gate) without a real model or GPU. The only
MLX-touching path (``load_weights``) is covered by the missing-shard
``FileNotFoundError`` test, which exercises the pre-``mx.load`` guard.
"""
import json
import os
import tempfile

import pytest

from weights import (
    WEIGHTS_FILENAME,
    WEIGHTS_INDEX_FILENAME,
    merge_weight_dicts,
    resolve_weights_paths,
    load_weights,
)


class TestResolveWeightsPaths:
    def test_single_file_fast_path(self, tmp_path):
        """A plain weights.safetensors resolves to exactly that one file."""
        single = tmp_path / WEIGHTS_FILENAME
        single.write_bytes(b"")
        assert resolve_weights_paths(str(tmp_path)) == [str(single)]

    def test_single_file_wins_over_index(self, tmp_path):
        """If both are present, the single file is the fast path."""
        (tmp_path / WEIGHTS_FILENAME).write_bytes(b"")
        (tmp_path / WEIGHTS_INDEX_FILENAME).write_text(
            json.dumps({"weight_map": {"a": "weights-00001-of-00002.safetensors"}})
        )
        assert resolve_weights_paths(str(tmp_path)) == [str(tmp_path / WEIGHTS_FILENAME)]

    def test_sharded_index_resolves_sorted_unique_parts(self, tmp_path):
        """
        A sharded index resolves to its unique part files, sorted by name for
        a deterministic load order, even if the weight_map lists them out of
        order or repeats a part.
        """
        index = {
            "metadata": {"total_size": 123},
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "weights-00002-of-00002.safetensors",
                "model.layers.0.mlp.gate_proj.weight": "weights-00001-of-00002.safetensors",
                "model.layers.0.mlp.down_proj.weight": "weights-00002-of-00002.safetensors",
                "model.embed_tokens.weight": "weights-00001-of-00002.safetensors",
            },
        }
        (tmp_path / WEIGHTS_INDEX_FILENAME).write_text(json.dumps(index))

        paths = resolve_weights_paths(str(tmp_path))
        # Two unique parts, sorted by filename.
        assert [os.path.basename(p) for p in paths] == [
            "weights-00001-of-00002.safetensors",
            "weights-00002-of-00002.safetensors",
        ]
        # Each path lives under the model dir.
        assert all(os.path.dirname(p) == str(tmp_path) for p in paths)

    def test_single_shard_index(self, tmp_path):
        """An index that points at exactly one part resolves to one path."""
        index = {"weight_map": {"w": "weights-00001-of-00001.safetensors"}}
        (tmp_path / WEIGHTS_INDEX_FILENAME).write_text(json.dumps(index))
        paths = resolve_weights_paths(str(tmp_path))
        assert len(paths) == 1
        assert os.path.basename(paths[0]) == "weights-00001-of-00001.safetensors"

    def test_neither_file_raises(self, tmp_path):
        """No single file and no index -> a clear FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc:
            resolve_weights_paths(str(tmp_path))
        # The message names both expected layouts so the user knows what to fix.
        assert WEIGHTS_FILENAME in str(exc.value)
        assert WEIGHTS_INDEX_FILENAME in str(exc.value)

    def test_index_missing_weight_map_raises(self, tmp_path):
        """An index without a usable weight_map is malformed -> ValueError."""
        (tmp_path / WEIGHTS_INDEX_FILENAME).write_text(json.dumps({"metadata": {}}))
        with pytest.raises(ValueError, match="weight_map"):
            resolve_weights_paths(str(tmp_path))

    def test_index_empty_weight_map_raises(self, tmp_path):
        (tmp_path / WEIGHTS_INDEX_FILENAME).write_text(json.dumps({"weight_map": {}}))
        with pytest.raises(ValueError, match="weight_map"):
            resolve_weights_paths(str(tmp_path))

    def test_index_invalid_json_raises(self, tmp_path):
        (tmp_path / WEIGHTS_INDEX_FILENAME).write_text("{not valid json")
        with pytest.raises(ValueError, match="weight_map|JSON"):
            resolve_weights_paths(str(tmp_path))


class TestMergeWeightDicts:
    def test_merges_disjoint_parts_in_order(self):
        a = {"embed": "A1", "layers.0.q": "A2"}
        b = {"layers.0.k": "B1", "lm_head": "B2"}
        merged = merge_weight_dicts([a, b])
        assert merged == {
            "embed": "A1",
            "layers.0.q": "A2",
            "layers.0.k": "B1",
            "lm_head": "B2",
        }

    def test_empty_parts_give_empty_dict(self):
        assert merge_weight_dicts([]) == {}
        assert merge_weight_dicts([{}, {}]) == {}

    def test_single_part_is_identity(self):
        part = {"w": "x"}
        assert merge_weight_dicts([part]) == part

    def test_duplicate_key_across_shards_raises(self):
        """A weight in more than one part means the index is malformed."""
        with pytest.raises(ValueError, match="more than one shard"):
            merge_weight_dicts([{"w": 1}, {"w": 2}])

    def test_does_not_mutate_inputs(self):
        a = {"x": 1}
        b = {"y": 2}
        merged = merge_weight_dicts([a, b])
        assert a == {"x": 1}
        assert b == {"y": 2}
        assert merged is not a and merged is not b


class TestLoadWeights:
    def test_missing_shard_raises_before_any_ml(self):
        """
        A referenced shard that does not exist must raise FileNotFoundError.
        This fires in the pre-``mx.load`` existence guard, so it passes even on
        a machine where ``mlx`` is installed but a shard is simply absent.
        """
        with pytest.raises(FileNotFoundError, match="shard not found"):
            load_weights(["/nonexistent/path/weights-00001-of-00002.safetensors"])


class TestEndToEndShardedLayout:
    def test_resolve_then_merge_synthetic(self, tmp_path):
        """
        Synthetic sharded checkpoint: write an index + two part files named
        accordingly, resolve the parts, then merge two dicts shaped like what
        those parts would hold. This is the key-merge-level test the issue
        asked for (no real weights needed).
        """
        part1_name = "weights-00001-of-00002.safetensors"
        part2_name = "weights-00002-of-00002.safetensors"
        (tmp_path / part1_name).write_bytes(b"")
        (tmp_path / part2_name).write_bytes(b"")
        (tmp_path / WEIGHTS_INDEX_FILENAME).write_text(json.dumps({
            "weight_map": {
                "model.embed_tokens.weight": part1_name,
                "model.layers.0.self_attn.q_proj.weight": part1_name,
                "model.layers.0.self_attn.k_proj.weight": part2_name,
            }
        }))

        paths = resolve_weights_paths(str(tmp_path))
        assert len(paths) == 2
        # Both resolved parts exist on disk.
        assert all(os.path.exists(p) for p in paths)

        # Simulate what mx.load would return per part (disjoint key sets).
        part1 = {"model.embed_tokens.weight": "t1", "model.layers.0.self_attn.q_proj.weight": "t2"}
        part2 = {"model.layers.0.self_attn.k_proj.weight": "t3"}
        merged = merge_weight_dicts([part1, part2])
        assert set(merged) == {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
        }
