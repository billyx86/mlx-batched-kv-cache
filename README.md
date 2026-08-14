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
- `tests/` - Unit tests

## API Reference

### `Transformer`

Main model class. See `model.py` for architecture details.

### `load_model(model_path)`

Loads MLX model, tokenizer, and config.

### `generate_stream(model, tokenizer, prompt, max_new_tokens, temperature, verbose=False)`

Yields text deltas token by token.

## Limitations

- Greedy sampling only (temperature > 0 prints warning)
- Single-prompt generation currently; batched inference in progress
- Requires MLX-compatible models

## License

MIT
