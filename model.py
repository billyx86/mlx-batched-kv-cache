import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple, List, Dict

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
            print(f"Concatenated keys shape: {keys.shape}, Concatenated values shape: {values.shape}")

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
    

class FeedForward(nn.Module):
    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()

        self.w1 = nn.Linear(dims, hidden_dims, bias=False)
        self.w2 = nn.Linear(hidden_dims, dims, bias=False)

        self.act = nn.GELU()
    
    def __call__(self, x: mx.array) -> mx.array:  
        # (B, L, D) -> (B, L, H) -> (B, L, D)
        return self.w2(self.act(self.w1(x)))
    

class TransformerBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, mlp_dimms: int):
        """
        dims (int): The feature dimensions of the input
        num_heads (int): The number of attention heads
        mlp_dims (int): The hidden dimensions of the feed-forward network
        """
        super().__init__()
        self.n_heads = num_heads
        self.dims = dims

        self.ln1 = nn.LayerNorm(dims)
        self.ln2 = nn.LayerNorm(dims)

        self.attention = Attention(dims, num_heads)

        self.ffn = FeedForward(dims, mlp_dimms)
    
    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[Tuple[mx.array, mx.array]] = None
                 ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass for the transformer block.
        args:
            x (mx.array): Input tensor of shape (batch_size, sequence_len, dims).
            mask (mx.array, optional): Attention mask.
            cache (Tuple[mx.array, mx.array], optional): KV cache for attention layer.

        returns:
§         Tuple[mx.array, Tuple[mx.array, mx.array]]:
            - Output tensor of the block (batch_size, sequence_len, dims).
            - Updated KV cache.
        """

        # Attention path (Pre-LayerNorm)
        residual = x
        h = self.ln1(x)
        attn_output, updated_cache = self.attention(h, mask=mask, cache=cache)
        h = residual + attn_output

        # Feed-forward path (Pre-LayerNorm)
        residual = h
        h_norm = self.ln2(h)
        ffn_output = self.ffn(h_norm)
        out = residual + ffn_output

        return out, updated_cache

class Transformer(nn.Module):
    # Transformer model for language modelling

    def __init__(self, vocab_size: int, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None):
        """
        vocab_size (int): Size of the vocabulary.
        num_layers (int): Number of transformer layers.
        dims (int): Feature dimensions of the input.
        num_heads (int): Number of attention heads.
        mlp_dims (int): Hidden dimensions of the feed-forward network.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dims = dims

        if mlp_dims is None:
            mlp_dims = dims * 4

        self.embedding = nn.Embedding(vocab_size, dims)

        self.layers = [
            TransformerBlock(dims=dims, num_heads=num_heads, mlp_dimms=mlp_dims)
            for _ in range(num_layers)
        ]

        self.ln_f = nn.LayerNorm(dims)

        self.lm_head = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self,
                 inputs: mx.array,
                 past_kv_caches: Optional[List[Optional[Tuple[mx.array, mx.array]]]] = None,
                 mask: Optional[mx.array] = None
                 ) -> Tuple[mx.array, List[Optional[Tuple[mx.array, mx.array]]]]:
        """
        Forward pass for the transformer model.
        
        args:
            inputs (mx.array): Input token IDs (batch_size, sequence_length).
            past_kv_caches (Optional[List[Optional[Tuple[mx.array, mx.array]]]]): 
                Cached key and value tensors for fast decoding, one tuple per layer.
                Defaults to None.
            mask (mx.array, optional): Attention mask. Defaults to None.
            
        returns:
            Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
            - Logits tensor (batch_size, sequence_length, vocab_size).
            - List of updated KV caches for all layers.
        """

        h = self.embedding(inputs)  # (batch_size, seq_len, dims)

        if past_kv_caches is None:
            past_kv_caches = [None] * self.num_layers

        if len(past_kv_caches) != self.num_layers:
             raise ValueError(
                f"Incorrect number of past_kv_caches provided. Expected {self.num_layers}, got {len(past_kv_caches)}"
            )
        
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            h, updated_cache = layer(h, mask=mask, cache=past_kv_caches[i])
            new_kv_caches.append(updated_cache)

        h = self.ln_f(h)

        logits = self.lm_head(h)

        return logits, new_kv_caches

  
# Example placeholder
if __name__ == '__main__':
    # Example config
    batch_size = 2
    prompt_len = 5
    vocab_size = 1000
    num_layers = 4
    dims = 128
    num_heads = 4
    mlp_dims = dims * 4

    # Create Transformer model
    model = Transformer(
        vocab_size=vocab_size,
        num_layers=num_layers,
        dims=dims,
        num_heads=num_heads,
        mlp_dims=mlp_dims
    )

    attention_layer = Attention(dims, num_heads)

    # Dummy prompt
    dummy_inputs = mx.random.randint(0, vocab_size, (batch_size, prompt_len))
    print(f"dummy_inputs shape: {dummy_inputs.shape}")

    print("Testing without cache...")
    logits_no_cache, caches_first_pass = model(dummy_inputs, past_kv_caches=None)
    print(f"Logits shape (no cache): {logits_no_cache.shape}")
    print(f"Number of caches returned: {len(caches_first_pass)}")
    if caches_first_pass:
        print(f"Shape of K cache from first layer (first pass): {caches_first_pass[0][0].shape}")


    print("Testing with cache...")
    next_token_input = mx.random.randint(0, vocab_size, (batch_size, 1))
    print(f"Next token input shape: {next_token_input.shape}")

    # Pass the list of caches obtained from the first pass
    logits_with_cache, caches_second_pass = model(next_token_input, past_kv_caches=caches_first_pass)
    print(f"Logits shape (with cache): {logits_with_cache.shape}") # (batch, 1, vocab_size)
    print(f"Number of caches returned (second pass): {len(caches_second_pass)}")
    if caches_second_pass:
        expected_cache_len = prompt_len + 1
        print(f"Shape of K cache from first layer (second pass): {caches_second_pass[0][0].shape}")
        assert caches_second_pass[0][0].shape[2] == expected_cache_len, "Cache length mismatch in second pass!"

    print("All tests passed!")
