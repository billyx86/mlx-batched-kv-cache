#!/usr/bin/env python3
"""
Benchmark script for batched KV caching performance.
"""
import time
import mlx.core as mx
from model import Transformer

def benchmark_batch_sizes():
    config = {
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-5,
        "vocab_size": 1000,
        "num_hidden_layers": 2,
        "quantization": {"group_size": 64, "bits": 4}
    }
    
    model = Transformer(config)
    mx.eval(model.parameters())
    
    batch_sizes = [1, 2, 4, 8]
    seq_len = 32
    
    print("Batch size benchmark")
    print("=" * 40)
    
    for bs in batch_sizes:
        inputs = mx.random.randint(0, 1000, (bs, seq_len))
        start = time.time()
        logits, caches = model(inputs)
        mx.eval(logits)
        elapsed = time.time() - start
        tokens_per_sec = (bs * seq_len) / elapsed
        print(f"Batch {bs}: {elapsed:.4f}s, {tokens_per_sec:.2f} tokens/s")

if __name__ == "__main__":
    benchmark_batch_sizes()
