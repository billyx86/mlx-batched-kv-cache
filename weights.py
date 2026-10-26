"""Checkpoint weight-file resolution and shard merging.

The file-resolution and dict-merge logic that used to be hard-coded inside
``load_model`` is factored out here and kept pure Python wherever it can be,
so it can be unit-tested without an MLX runtime (see issue #17). The one thin
MLX call (``mx.load``) is imported lazily inside ``load_weights`` so this
module stays importable on machines without MLX installed.
"""

import json
import os
from typing import Dict, List

WEIGHTS_FILENAME = "weights.safetensors"
WEIGHTS_INDEX_FILENAME = "weights.safetensors.index.json"


def resolve_weights_paths(model_path: str) -> List[str]:
    """
    Resolve which weight file(s) live under ``model_path``.

    Returns a list of file paths:
      * a single element — the plain ``weights.safetensors`` (fast path), or
      * N elements — the sharded parts named in
        ``weights.safetensors.index.json``, deduplicated and sorted by filename
        for a deterministic load order.

    A sharded checkpoint (the HF/MLX layout for models beyond a couple of
    hundred MB) ships ``weights.safetensors.index.json`` plus
    ``weights-XXXXX-of-YYYYY.safetensors`` parts instead of one big file.
    Before this, ``load_model`` hard-coded ``weights.safetensors`` and raised
    ``FileNotFoundError`` for those repos even though the directory was a
    perfectly good MLX checkpoint (issue #17).

    Args:
        model_path (str): Directory containing the checkpoint.

    Returns:
        List[str]: Ordered paths of the weight file(s) to load.

    Raises:
        FileNotFoundError: if neither the single file nor the index is present.
        ValueError: if the index is present but malformed (bad JSON, or no
            usable ``weight_map``).
    """
    single = os.path.join(model_path, WEIGHTS_FILENAME)
    if os.path.exists(single):
        return [single]

    index = os.path.join(model_path, WEIGHTS_INDEX_FILENAME)
    if os.path.exists(index):
        with open(index, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from {index}: {e}")

        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"{index} has no usable 'weight_map' "
                "(expected a non-empty mapping of weight name -> shard file)"
            )

        # The index maps every weight to the part it lives in; the parts we
        # need are the unique referenced filenames, sorted for determinism.
        shard_names = sorted(set(weight_map.values()))
        return [os.path.join(model_path, name) for name in shard_names]

    raise FileNotFoundError(
        f"No weights found at {model_path}: expected either "
        f"{WEIGHTS_FILENAME} or {WEIGHTS_INDEX_FILENAME} (with its sharded parts)"
    )


def merge_weight_dicts(parts: List[Dict]) -> Dict:
    """
    Merge a sequence of shard weight dicts into one.

    HF/MLX sharding never splits a single tensor across files, so the parts
    are disjoint and a plain insert in order is correct. A key appearing in
    more than one part means the index is malformed, so this raises instead
    of silently last-write-wins.

    Pure function of the dicts (no MLX calls), so it's unit-testable without
    an MLX runtime.

    Args:
        parts (List[Dict]): One dict per shard, in load order.

    Returns:
        Dict: the merged ``{weight_name: tensor}`` mapping.

    Raises:
        ValueError: if a weight name appears in more than one part.
    """
    merged: Dict = {}
    for part in parts:
        for key, value in part.items():
            if key in merged:
                raise ValueError(
                    f"weight '{key}' appears in more than one shard; "
                    "the index is malformed"
                )
            merged[key] = value
    return merged


def load_weights(weights_paths: List[str]) -> Dict:
    """
    Load a (possibly sharded) checkpoint into one merged weight dict.

    Single file: one ``mx.load``. Sharded: ``mx.load`` each part and merge
    the dicts via :func:`merge_weight_dicts`. The result is the same
    ``{weight_name: tensor}`` mapping that ``normalize_weight_keys`` and
    ``model.update`` consume, regardless of how the checkpoint was split.

    Args:
        weights_paths (List[str]): Ordered weight file paths (from
            :func:`resolve_weights_paths`).

    Returns:
        Dict: the merged raw weight dict (leading ``model.`` prefix still
            present — that's ``normalize_weight_keys``' job).

    Raises:
        FileNotFoundError: if a referenced shard file does not exist.
    """
    for path in weights_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weight shard not found at {path}")

    from mlx.core import load as _mx_load  # lazy: keeps this module importable without MLX

    parts = [_mx_load(path) for path in weights_paths]
    return merge_weight_dicts(parts)
