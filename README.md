# mlx-batched-kv-cache

Exploration of the implementation of Batched KV Caching using Apple's MLX framework.

## Overview

This project implements a transformer model with batched key-value caching for efficient autoregressive text generation on Apple Silicon using MLX. The implementation includes quantized linear layers, RoPE positional embeddings, and incremental KV cache management.

## Features

- Batched KV cache for efficient generation
- Quantized linear layers via `mlx.nn.QuantizedLinear`
- RoPE positional embeddings
- Streaming text generation with `generate_stream`
- Support for Mistral-compatible models

## Requirements

- Apple Silicon Mac
- Python 3.10+
- MLX
- transformers
- safetensors
- numpy

## Installation

```bash
pip install -r requirements.txt
```

MLX installation:

```bash
pip install mlx
```

## Quick Start

```python
from generate import load_model, generate_stream

model, tokenizer, config = load_model("./models/Mistral-7B-Instruct-v0.2-mlx")

prompt = "The future of AI is"
for text_delta in generate_stream(model, tokenizer, prompt, max_new_tokens=100):
    print(text_delta, end="", flush=True)
```

Command line:

```bash
python generate.py --model-path ./models/Mistral-7B-Instruct-v0.2-mlx \
  --prompt "Explain batched KV caching" \
  --max-tokens 200
```

## Project Structure

- `model.py` - Transformer architecture with KV caching
- `generate.py` - Model loading and text generation
- `list_weights.py` - Utility to inspect safetensors weights
- `tests/` - Unit tests (pytest)
- `benchmark.py` - Quick batch-size throughput benchmark

## Testing

Tests run on macOS ARM runners (MLX requires Apple Silicon):

```bash
pip install -r requirements.txt pytest
pytest -q
```

The suite covers the causal mask (plain and chunked-prefill variants),
GQA broadcast vs. repeat reference, KV-cache equivalence with a full
forward pass, weight-key normalization, and the prefill→decode token
extraction — without needing a downloaded model.

## API Reference

### `Transformer`

Main model class. See `model.py` for architecture details.

`__call__` supports three usage patterns:

- **Full prefill** — pass the whole prompt with no cache: a causal mask is
  applied automatically.
- **Single-token decode** — pass a `(B, 1)` token with `past_kv_caches` from
  a previous call: O(1) per step, no mask needed.
- **Chunked prefill** — pass a multi-token chunk `(B, T)` together with
  `past_kv_caches` from an earlier chunk (added for issue #13). A
  `(1, 1, T, T + past_len)` mask is built automatically, so the chunk is
  causal within itself while freely attending to the cached prefix. This is
  how to stream long prompts in fixed-size pieces through the public API:

  ```python
  # Fill the cache with the first chunk.
  _, caches = model(chunk_1)
  # Continue with the next chunk — causally correct, no manual mask needed.
  logits, caches = model(chunk_2, past_kv_caches=caches)
  ```

### `make_causal_mask(seq_len)` / `make_causal_mask_with_prefix(new_len, past_len)`

Build `(1, 1, T, T)` and `(1, 1, T, T + past_len)` attention masks
respectively (see `model.py`); the latter exists for chunked prefill.

### `load_model(model_path)`

Loads MLX model, tokenizer, and config. Weight keys with a leading
`model.` prefix (HF-style checkpoints) are normalized automatically;
inner occurrences of `model.` in a key are preserved.

### `generate_stream(model, tokenizer, prompt, max_new_tokens, temperature, verbose=False)`

Yields text deltas token by token.

## Limitations

- Greedy sampling only (temperature > 0 prints warning)
- Single-prompt generation currently; batched inference in progress
- Requires MLX-compatible models

## License

MIT
