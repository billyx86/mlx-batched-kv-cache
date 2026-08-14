"""List the tensor names stored in a safetensors weights file."""
import argparse
import sys

from safetensors import safe_open

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List tensor names in a safetensors file.")
    parser.add_argument(
        "weights_path",
        nargs="?",
        default="./models/Mistral-7B-Instruct-v0.2-mlx/weights.safetensors",
        help="Path to weights.safetensors",
    )
    args = parser.parse_args()
    weights_path = args.weights_path

    try:
        print(f"Reading keys from: {weights_path}")
        with safe_open(weights_path, framework="mlx") as f:
            keys = sorted(f.keys())

        print("\n--- Keys found in weights.safetensors ---")
        if not keys:
            print("(No keys found)")
        else:
            for key in keys:
                print(key)
        print("--- End of keys ---")
    except FileNotFoundError:
        print(f"Error: Weights file not found at {weights_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError reading safetensors file: {e}", file=sys.stderr)
        sys.exit(1)
