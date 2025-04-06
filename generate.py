from typing import List
import mlx.core as mx
import time
import argparse
from model import Transformer

# Simulated tokenizer
EOS_TOKEN_ID = -1
PAD_TOKEN_ID = -2
VOCAB_SIZE = 1000

def greedy_sample(logits: mx.array) -> mx.array:
    """
    Selects the token with the highest probability from the logits.
    args:
        logits (mx.array): logits tensor (batch_size, 1, vocab_size)
    returns:
        mx.array: the chosen token IDs (batch_size, 1)
    """
    # Convert logits to probabilities
    return mx.argmax(logits, axis=-1)

def batch_generate(
        model: Transformer,
        prompts: mx.array,  # (batch_size, prompt_len)
        max_new_tokens: int,
        temperature: float = 0.0,  # 0.0 for greedy sampling
        eos_token_id: int = EOS_TOKEN_ID,
) -> List[List[int]]:
    """
    Generates token sequences for a batch of prompts using KV caching.
    
    args:
        model (Transformer): The transformer model for generation.
        prompts (mx.array): Input token ID sequences.
        max_new_tokens (int): Maximum number of new tokens to generate per sequence.
        temperature (float): Sampling temperature. 0.0 for greedy sampling. (Not yet implemented)
        eos_token_id (int): End-of-sequence token ID.
    returns:
        List[List[int]]: Token ID indicating the end of a sequence.
    """
    if temperature != 0.0:
        # TODO: Implement sampling strategies like top-k, top-p for temp > 0
        print("Warning: Temperature > 0.0 requested, but only greedy sampling (temp=0.0) is implemented. Using greedy.")

    batch_size, prompt_len = prompts.shape
    generated_sequences = [[] for _ in range(batch_size)]
    active_mask = mx.ones(batch_size, dtype=mx.bool_) # Track which sequences are still generating

    print(f"Starting batched generation for batch size {batch_size}...")
    print(f"Processing initial prompt (length {prompt_len})...")
    start_time = time.time()

    # Process prompt phase (initialise KV caches)
    _, kv_caches = model(prompts, past_kv_caches=None)

    current_token = prompts[:, -1:]  # (batch_size, 1)
    logits, kv_caches = model(current_token, past_kv_caches=kv_caches)


    prompt_processing_time = time.time() - start_time
    print(f"Prompt processing time: {prompt_processing_time:.4f} seconds")
    print("Starting token-by-token generation loop...")
    loop_start_time = time.time()

    # logits shape: (batch_size, 1, vocab_size)
    for i in range(max_new_tokens):
        step_start_time = time.time()

        if temperature == 0.0:
            next_token = greedy_sample(logits) # (batch_size, 1)
        else:
            next_token = greedy_sample(logits) # Placeholder for temperature sampling
        
        # next_token_val = next_token.item()
        next_token_list = next_token.tolist()

        for idx in range(batch_size):
            if active_mask[idx].item():
                generated_sequences[idx].append(next_token_list[idx][0])

        # Update the active mask
        just_finished = (next_token == eos_token_id)
        active_mask = active_mask & (~just_finished.squeeze(axis=-1))

        # Stop if all sequences have generated EOS
        if not active_mask.any():
            print(f"All sequences finished generating at step {i}.")
            break

        # Prepare input for the next step
        current_token = next_token

        # Call the model for the next step
        logits, kv_caches = model(current_token, past_kv_caches=kv_caches)

        # Evaluate tensors
        mx.eval(logits, kv_caches)

        step_time = time.time() - step_start_time
        if (i + 1) % 10 == 0:
            print(f"Step {i + 1}/{max_new_tokens} completed in {step_time:.4f} seconds.")

    loop_time = time.time() - loop_start_time
    total_tokens = sum(len(seq) for seq in generated_sequences)
    print(f"\nGeneration loop time: {loop_time:.4f} seconds")
    print(f"Total tokens generated: {total_tokens} across {batch_size} sequences")
    if loop_time > 0 and total_tokens > 0:
        print(f"Average tokens per second: {total_tokens / loop_time:.2f} tokens/sec")

    return generated_sequences


if __name__ == "__main__":
    # TODO: Use argparse for proper command-line argument handling
    # Model Configuration (should match model.py example or loaded model)
    batch_size = 2
    prompt_len = 5
    vocab_size = VOCAB_SIZE
    num_layers = 4
    dims = 128
    num_heads = 4
    mlp_dims = dims * 4

    max_new_tokens = 50

    eos_token_id_for_test = 999

    print("Initializing model...")

    model = Transformer(
        vocab_size=vocab_size,
        num_layers=num_layers,
        dims=dims,
        num_heads=num_heads,
        mlp_dims=mlp_dims
    )
    # Note: model is randomly initialized, so results will not be meaningful
    # Load pre-trained weights if available
    # model.load_weights('path_to_weights.safetensors')

    # Dummy batched prompt
    prompt1 = list(range(1, prompt_len+1))
    prompt2 = list(range(10, 10 + prompt_len))
    prompts_list = [prompt1, prompt2]

    if len(prompts_list) != batch_size:
        print(f"Warning: Expected {batch_size} prompts, but got {len(prompts_list)}. Adjusting to match batch size.")
        batch_size = len(prompts_list)

    prompts_mx = mx.array(prompts_list)  # (batch_size, prompt_len)

    print(f"\nRunning batch generation with batch_size={batch_size}, max_new_tokens={max_new_tokens}...")

    generated_results = batch_generate(
        model=model,
        prompts=prompts_mx,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id_for_test
    )

    print("\nGenerated sequences:")
    for i, seq in enumerate(generated_results):
        print(f"Sequence {i+1}: {seq}")

    print("\nBatch generation completed.")
