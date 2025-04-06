import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple, List, Dict

class RoPE(nn.Module):
    """
    Implements Rotary Positional Encoding (RoPE) for Transformer models.
    """
    def __init__(self, dims: int, traditional: bool = False, base: float = 10000.0):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.inv_freq = self._compute_inv_freq()

    def _compute_inv_freq(self):
        return 1.0 / (self.base ** (mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims))
    
    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """
        Applies RoPE to the input tensor.
        
        args:
            x (mx.array): Input tensor of shape (batch_size, num_heads, seq_len, head_dim).
            offset (int): The starting position offset for sequences.
        
        returns:
            mx.array: Tensor with RoPE applied.
        """
        seq_len = x.shape[1]
        positions = mx.arange(offset, offset + seq_len, dtype=self.inv_freq.dtype)

        # Calculate frequencies and embeddings
        freqs = mx.outer(positions, self.inv_freq)
        # Shape: (seq_len, dims / 2) -> (1, 1, seq_len, dims / 2) for broadcasting
        freqs = freqs.reshape(1, 1, seq_len, -1)

        emb = mx.concatenate([freqs, freqs], axis=-1)  # (1, 1, seq_len, dims)

        cos_emb = mx.cos(emb)
        sin_emb = mx.sin(emb)

        # Apply rotation based on trad. or alt RoPE implementation
        if self.traditional:
            x1 = x[..., : self.dims // 2]
            x2 = x[..., self.dims // 2:]
            rotated_x = mx.concatenate([-x2, x1], axis=-1)
            return (x * cos_emb) + (rotated_x * sin_emb)
        else:
            # Reshape x for complex multiplication: (..., seq_len, dims) -> (..., seq_len, dims/2, 2)
            x_complex = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
            x_complex = mx.complex(x_complex[..., 0], x_complex[..., 1])

            # Reshape emb for complex multiplication: (..., seq_len, dims) -> (..., seq_len, dims/2*2) -> (..., seq_len, dims/2)
            emb_complex = emb.reshape(*emb.shape[:-1], -1, 2)
            # Convert to complex numbers: cos(emb) + i * sin(emb)
            emb_complex = mx.exp(mx.multiply(emb_complex[..., 1], 1j)) # Equivalent to cos + i*sin

            # Apply rotation in complex plane: x * exp(i * emb)
            rotated_x_complex = x_complex * emb_complex
            # Reshape back: (..., seq_len, dims/2) -> (..., seq_len, dims)
            rotated_x = mx.concatenate([rotated_x_complex.real, rotated_x_complex.imag], axis=-1)

            return rotated_x.astype(x.dtype)

class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int, num_kv_heads: int):
        """
        dims (int): The feature dimensions of the input
        num_heads (int): The number of attention heads
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dims, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dims, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dims, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dims, bias=False)

    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[Tuple[mx.array, mx.array]] = None,
                 rope: RoPE = None,
                 offset: int = 0
                 ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass for the attention layer.

        x (mx.array): Input tensor of shape (batch_size, seq_len, dims).
        mask (mx.array, optional): Attention mask of shape (batch_size, 1, 1, seq_len).
        cache (Tuple[mx.array, mx.array], optional): Cached key and value tensors for fast decoding.
        rope (RoPE, optional): RoPE instance for positional encoding.
        offset (int): The starting position offset for sequences.
        """
        batch_size, seq_len, dims = x.shape

        # Project Q, K, V for current input
        queries = self.wq(x)    # (batch_size, seq_len, dims)
        keys = self.wk(x)       # (batch_size, seq_len, dims)
        values = self.wv(x)     # (batch_size, seq_len, dims)

        # Reshape and transpose for multi-head attention
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE before caching
        if rope is not None:
            raise ValueError("RoPE module must be provided to Attention layer.")
        queries = rope(queries, offset=offset)
        keys = rope(keys, offset=offset)

        # Update KV cache
        if cache is not None:
            key_cache, value_cache = cache
            print(f"Key cache shape: {key_cache.shape}, Value cache shape: {value_cache.shape}")

            keys = mx.concat([key_cache, keys], axis=2)
            values = mx.concat([value_cache, values], axis=2)
            print(f"Concatenated keys shape: {keys.shape}, Concatenated values shape: {values.shape}")

        # Handle GQA / MQA
        if self.num_heads < self.num_kv_heads:
            num_repeats = self.num_heads // self.num_kv_heads
            keys = mx.repeat(keys, repeats=num_repeats, axis=1)
            values = mx.repeat(values, repeats=num_repeats, axis=1)

        # Perform scaled dot-product attention
        # queries: (B, nq, L, D)
        # keys:    (B, nq, S_full, D)
        # scores:  (B, nq, L, S_full)
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            print(f"mask shape: {mask.shape}")
            scores = scores + mask

        # Apply softmax to get attention weights and apply to values, float32 for stability
        attention_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        
        # attention_weights: (B, nq, L, S_full)
        # values:            (B, nq, S_full, D)
        # attention_output:  (B, nq, L, D)
        attention_output = attention_weights @ values 

        # Reshape and Project Output 
        # (B, nq, L, D) -> (B, L, nq, D) -> (B, L, nq*D) = (B, L, dims)
        output_concat = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.wo(output_concat)

        # Final linear projection
        return output, (keys[:, :self.num_kv_heads, :, :], values[:, :self.num_kv_heads, :, :])


class FeedForward(nn.Module):
    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()

        self.w1 = nn.Linear(dims, hidden_dims, bias=False)  # Gating MLP
        self.w2 = nn.Linear(hidden_dims, dims, bias=False)  # Down projectiom
        self.w3 = nn.Linear(dims, hidden_dims, bias=False)  # Activation MLP

    
    def __call__(self, x: mx.array) -> mx.array: 
        # silu(w1(x)) * w3(x) 
        gate = self.w1(x)
        activation = nn.silu(gate) * self.w3(x)

        return self.w2(activation) 
    

class TransformerBlock(nn.Module):
    """
    Transformer block aligned with RMSNorm, RoPE, SwiGLU FFN.
    """
    def __init__(self, dims: int, num_heads: int, num_kv_heads: int, mlp_dims: int, norm_eps: float = 1e-5):
        super().__init__()
        self.n_heads = num_heads
        self.dims = dims

        self.attention_norm = nn.RMSNorm(dims, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dims, eps=norm_eps)

        self.attention = Attention(dims, num_heads, num_kv_heads)
        self.feed_forward = FeedForward(dims, hidden_dims=mlp_dims)
    
    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[Tuple[mx.array, mx.array]] = None,
                 rope: RoPE = None,
                 offset: int = 0
                 ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass for the transformer block.
        args:
            x (mx.array): Input tensor of shape (batch_size, sequence_len, dims).
            mask (mx.array, optional): Attention mask.
            cache (Tuple[mx.array, mx.array], optional): KV cache for attention layer.
            rope (RoPE, optional): RoPE instance for positional encoding.
            offset (int): The starting position offset for sequences.

        returns:
§         Tuple[mx.array, Tuple[mx.array, mx.array]]:
            - Output tensor of the block (batch_size, sequence_len, dims).
            - Updated KV cache.
        """

        # Attention path (Pre-RMSNorm)
        residual = x
        h = self.attention_norm(x)
        attn_output, updated_cache = self.attention(h, mask=mask, cache=cache, rope=rope, offset=offset)
        h = residual + attn_output

        # Feed-forward path (Pre-LayerNorm)
        residual = h
        h_norm = self.ffn_norm(h)
        ffn_output = self.feed_forward(h_norm)
        out = residual + ffn_output

        return out, updated_cache

class Transformer(nn.Module):
    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): Model configuration dictionary containing parameters like:
                           vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                           num_key_value_heads, hidden_dim (for FFN), rms_norm_eps, rope_theta (base)

        """
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        dims = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        mlp_dims = config["intermediate_size"]  # FFN hidden data
        norm_eps = config["rms_norm_eps"]
        rope_base = config.get("rope_theta", 10000.0)  # Default RoPE base if not in config

        self.tok_embeddings = nn.Embedding(self.vocab_size, dims)

        # RoPE Embeddings (shared across layers) assuming head_dim = dims // num_heads
        head_dim = dims // num_heads
        self.rope = RoPE(head_dim, traditional=False, base=rope_base)

        self.layers = [
            TransformerBlock(
                dims=dims,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_dims=mlp_dims,
                norm_eps=norm_eps
            )
            for _ in range(self.num_hidden_layers)
        ]

        # Mistral naming: norm
        self.norm = nn.RMSNorm(dims, eps=norm_eps)

        # Mistral naming: output
        self.output = nn.Linear(dims, self.vocab_size, bias=False)

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

        h = self.embedding(inputs)

        # Determine sequence offset for RoPE
        offset = 0
        if past_kv_caches is not None and past_kv_caches[0] is not None:
            offset = past_kv_caches[0][0].shape[2]

        if past_kv_caches is None:
            past_kv_caches = [None] * self.num_layers

        if len(past_kv_caches) != self.num_layers:
             raise ValueError(
                f"Incorrect number of past_kv_caches provided. Expected {self.num_layers}, got {len(past_kv_caches)}"
            )
        
        new_kv_caches = []

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            h, updated_cache = layer(
                h,
                mask=mask,
                cache=past_kv_caches[i],
                rope=self.rope,
                offset=offset
            )
            new_kv_caches.append(updated_cache)

        h = self.ln_f(h)
        logits = self.lm_head(h)

        return logits, new_kv_caches

    @property
    def layers(self):
        return self._layers
    
    @layers.setter
    def layers(self, value):
        self._layers = value
