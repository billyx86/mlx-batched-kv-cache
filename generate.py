import mlx.core as mx
import time
import argparse
import os
import json
from typing import List, Tuple, Dict, Generator
from transformers import AutoTokenizer
from model import Transformer

def load_model(model_path: str) -> Tuple[Transformer, AutoTokenizer, Dict]:
    """
    Loads the MLX model, tokenizer, and configuration from a specified path.

    Args:
        model_path (str): Path to the directory containing model weights, config, and tokenizer.

    Returns:
        Tuple[Transformer, AutoTokenizer, Dict]: Loaded model, tokenizer, and config.
    """
    print(f"Loading model from {model_path}...")
    start_load = time.time()

    config_path = os.path.join(model_path, "config.json")
    weights_path = os.path.join(model_path, "weights.safetensors")
    tokenizer_path = model_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
    if not os.path.isdir(tokenizer_path):
         raise FileNotFoundError(f"Tokenizer path not found at {tokenizer_path}")

    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print("Configuration loaded.")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Tokenizer loaded.")
    except Exception as e:
        raise RuntimeError(f"Error loading tokenizer: {e}")

    # Instantiate the model using the loaded config
    try:
        model = Transformer(config=config)
        print("Model structure initialized.")
    except Exception as e:
        raise RuntimeError(f"Error initializing model structure: {e}")

    # Load weights, handling potential 'model.' prefix
    try:
        print(f"Loading weights from {weights_path}...")
        weights = mx.load(weights_path)
        
        # Remove 'model.' prefix if present
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
        
        model.update(weights) # Use update with the processed dictionary
        # model.load_weights(weights_path) # Original call replaced
        print("Weights loaded and applied.")
        
    except Exception as e:
        print(f"Error loading weights from {weights_path}: {e}")
        print("Ensure the model structure in model.py matches the weight names/shapes in the file (after potential prefix removal).")
        raise RuntimeError("Weight loading failed.")

    # Ensure weights are loaded before proceeding
    mx.eval(model.parameters())

    # Ensure tokenizer has pad token if needed later (though not used in batch_size=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token.")

    load_time = time.time() - start_load
    print(f"Model, tokenizer, and weights loaded in {load_time:.2f}s")

    return model, tokenizer, config

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

def generate_stream(
        model: Transformer,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
) -> Generator[str, None, None]:
    """
    Generates text token by token for a single prompt using KV caching.
    Yields the generated text delta at each step.

    Args:
        model (Transformer): The loaded Transformer model instance.
        tokenizer (AutoTokenizer): The loaded tokenizer.
        prompt (str): The input prompt string.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature. 0.0 means greedy. (Not fully implemented yet).

    Yields:
        Generator[str, None, None]: Yields text delta at each generation step.
    """
    if temperature != 0.0:
        print("Warning: Temperature > 0.0 requested, but only greedy sampling (temp=0.0) is implemented. Using greedy.")
        # TODO: Implement proper temperature sampling

    print("Encoding prompt...")
    # Encode the prompt, add_special_tokens=True might be needed for some models
    prompt_ids_np = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="np")
    prompt_ids = mx.array(prompt_ids_np)

    if prompt_ids.shape[1] == 0:
        raise ValueError("Prompt is empty after encoding.")
        return
    
    print(f"Processing prompt ({prompt_ids.shape[1]} tokens)...")
    start_time = time.time()

    # Process the prompt to initialize KV caches
    _, kv_caches = model(prompt_ids, past_kv_caches=None)
    current_token = prompt_ids[:, -1:]
    logits, kv_caches = model(current_token, past_kv_caches=kv_caches)

    prompt_time = time.time() - start_time
    print(f"Prompt processing time: {prompt_time:.4f} seconds")

    # Generation Loop Phase
    generated_token_ids = []
    current_decoded_text = ""
    print("Generating tokens...")
    loop_start_time = time.time()

    for i in range(max_new_tokens):
        step_start_time = time.time()

        if temperature == 0.0:
            next_token = greedy_sample(logits)
        else:
            next_token = greedy_sample(logits)  # Placeholder for temperature sampling

        token_id = next_token.item()
        generated_token_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            print(f"\nEnd of sequence token encountered at step {i}.")
            break
            
        # Decode the newly generated token only to yield delta
        # This might produce partial unicode sequences, handle carefully
        # Attempting to decode incrementally
        current_sequence = generated_token_ids
        # Use skip_special_tokens=True to avoid printing EOS
        new_decoded_text = tokenizer.decode(current_sequence, skip_special_tokens=True)

        # Calculate and yield the delta
        text_delta = new_decoded_text[len(current_decoded_text):]
        current_decoded_text = new_decoded_text # Update the full decoded text
        yield text_delta # Yield the newly generated chunk

        # Prepare for next iteration
        current_token = next_token.reshape((1, 1))  # Reshape for the model input
        logits, kv_caches = model(current_token, past_kv_caches=kv_caches)
        mx.eval(logits, kv_caches) # Evaluate for the next step

        step_time = time.time() - step_start_time
        print(f"\nStep {i + 1}/{max_new_tokens} completed in {step_time:.4f} seconds.\n\n")
    
    loop_time = time.time() - loop_start_time
    total_gen_tokens = len(generated_token_ids)
    print(f"\nGeneration loop finished in {loop_time:.4f} seconds")
    print(f"Total tokens generated: {total_gen_tokens}")
    if loop_time > 0 and total_gen_tokens > 0:
        print(f"Average tokens per second: {total_gen_tokens / loop_time:.2f} tokens/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained MLX model with KV caching.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the directory containing MLX model weights, config, and tokenizer.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for text generation.")    
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0.0 for greedy).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    mx.random.seed(args.seed)

    try:
        model, tokenizer, config = load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    print((f"\nPrompt: {args.prompt}"))

    print("\nGeneration")
    full_response = ""
    for text_delta in generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp
    ):
        print(text_delta, end="", flush=True)
        full_response += text_delta

    print("\n\nGeneration complete.")
