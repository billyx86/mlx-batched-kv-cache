import sys
import os
from safetensors import safe_open

weights_path = "./models/Mistral-7B-Instruct-v0.2-mlx/weights.safetensors"

if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at {weights_path}", file=sys.stderr)
    sys.exit(1)

try:
    print(f"Reading keys from: {weights_path}")
    all_keys = set()
    with safe_open(weights_path, framework="mlx") as f:
        for key in f.keys():
            all_keys.add(key)

    print("\n--- Keys found in weights.safetensors ---")
    if not all_keys:
        print("(No keys found)")
    else:
        for key in sorted(list(all_keys)):
            print(key)
    print("--- End of keys ---")

except Exception as e:
    print(f"\nError reading safetensors file: {e}", file=sys.stderr)
    sys.exit(1)