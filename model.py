import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple

class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        """
        dims (int): The feature dimensions of the input
        num_heads (int): The number of attention heads
        """
        super().__init__()

        if dims % num_heads != 0:
            raise ValueError(f"Number of heads {num_heads} must evenly divide the feature dimensions {dims}.")
        
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear layers for query, key, and value projections
        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)

        # Linear layer for output projection
        self.o_proj = nn.Linear(dims, dims, bias=False)
    
    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[Tuple[mx.array, mx.array]] = None
                 ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass for the attention layer.

        Args:
            x (mx.array): Input tensor of shape (batch_size, seq_len, dims).
            mask (mx.array, optional): Attention mask of shape (batch_size, 1, 1, seq_len).
            cache (Tuple[mx.array, mx.array], optional): Cached key and value tensors for fast decoding.
        """
        batch_size, seq_len, dims = x.shape

        # Project Q, K, V for current input
        queries = self.q_proj(x)    # (batch_size, seq_len, dims)
        keys = self.k_proj(x)       # (batch_size, seq_len, dims)
        values = self.v_proj(x)     # (batch_size, seq_len, dims)

        # Reshape and transpose for multi-head attention
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Update KV cache if provided
        if cache is not None:
            key_cache, value_cache = cache
            print(f"Key cache shape: {key_cache.shape}, Value cache shape: {value_cache.shape}")

            keys = mx.concat([key_cache, keys], axis=2)
            values = mx.concat([value_cache, values], axis=2)
            print(f"concatenated keys shape: {keys.shape}, concatenated values shape: {values.shape}")

        # Perform scaled dot-product attention
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            print(f"mask shape: {mask.shape}")
            scores = scores + mask

        # Apply softmax to get attention weights and apply to values
        attention_weights = mx.softmax(scores, axis=-1)
        attention_output = attention_weights @ values 

        # Reshape and transpose back to original dimensions
        output_concat = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, dims)  # (batch_size, seq_len, dims)

        # Final linear projection
        output = self.o_proj(output_concat)

        return output, (keys, values)
    
# Example placeholder
if __name__ == '__main__':
    # Example usage
    batch_size = 2
    seq_len = 5
    dims = 128
    num_heads = 4
    cache_len = 10

    attention_layer = Attention(dims, num_heads)

    # Dummy input
    x_input = mx.random.normal((batch_size, seq_len, dims))
    print(f"x_input shape: {x_input.shape}")

    # Dummy cache
    k_cache_shape = (batch_size, num_heads, cache_len, dims // num_heads)
    v_cache_shape = (batch_size, num_heads, cache_len, dims // num_heads)
    dummy_k_cache = mx.zeros(k_cache_shape)
    dummy_v_cache = mx.zeros(v_cache_shape)
    dummy_cache = (dummy_k_cache, dummy_v_cache)
    print(f"Initial K Cache shape: {dummy_k_cache.shape}, Initial V Cache shape: {dummy_v_cache.shape}")

    print("Testing without cache...")
    output_no_cache, cache_first_pass = attention_layer(x_input, mask=None, cache=None)
    print(f"Output shape without cache: {output_no_cache.shape}")
    print(f"Cache after first pass: K = {cache_first_pass[0].shape}, V = {cache_first_pass[1].shape}")

    print("Testing with cache...")
    x_next_token = mx.random.normal((batch_size, 1, dims))
    print(f"x_next_token shape: {x_next_token.shape}")

    # Use the cache from the first pass
    output_with_cache, cache_second_pass = attention_layer(x_next_token, mask=None, cache=cache_first_pass)
    print(f"Output shape with cache: {output_with_cache.shape}")

    expected_cache_len = seq_len + 1
    print(f"Updated K Cache shape: {cache_second_pass[0].shape}, Updated V Cache shape: {cache_second_pass[1].shape}")
    assert cache_second_pass[0].shape[2] == expected_cache_len, f"Expected cache length {expected_cache_len}, but got {cache_second_pass[0].shape[2]}"

    print("All tests passed.")