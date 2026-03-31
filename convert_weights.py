#!/usr/bin/env python3
"""
Convert Hugging Face VGGT model.pt (ZIP format) to a standard PyTorch checkpoint
with C++ compatible key names.

Usage:
    python convert_weights.py [--input INPUT.pt] [--output OUTPUT.pt]

Example:
    python convert_weights.py --input weights/model.pt --output weights/model_cpp.pt
"""

import argparse
import sys
import torch
import re
import os
from pathlib import Path


def convert_key_to_cpp(key: str) -> str:
    """
    Convert Python-style state_dict key to C++ libtorch compatible key.
    
    Python naming: module.submodule.0.param
    C++ naming:    module_submodule_0_param
    
    But libtorch named_parameters() uses: module.submodule.0.param (with dots)
    The difference is in numeric indexing:
    - Python: blocks.0, projects.0, trunk.0
    - C++:    blocks_0, projects_0, trunk_0
    """
    # Replace numeric indices like .0. -> _0.
    # Pattern: .<number>. or .<number> at end
    cpp_key = key
    
    # Replace patterns like .0. -> _0.
    # We need to handle: blocks.0.norm, projects.0.weight, trunk.0.attn, etc.
    cpp_key = re.sub(r'\.(\d+)\.', r'_\1.', cpp_key)
    
    # Handle the case where number is at the end: .0 -> _0
    cpp_key = re.sub(r'\.(\d+)$', r'_\1', cpp_key)
    
    return cpp_key


def convert_weights(input_path: str, output_path: str):
    """
    Load Python model weights and convert keys to C++ compatible format.
    
    Args:
        input_path: Path to the input Python model checkpoint
        output_path: Path to save the converted checkpoint
    """
    print(f"Loading model from: {input_path}")
    
    try:
        # Load the state dict
        state_dict = torch.load(input_path, map_location='cpu', weights_only=False)
        print(f"Successfully loaded state dict with {len(state_dict)} keys")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    if not isinstance(state_dict, dict):
        print(f"Error: Expected dict, got {type(state_dict)}")
        return False
    
    # Convert keys to C++ format
    print("\nConverting key names to C++ format...")
    converted_dict = {}
    conversion_examples = []
    
    for key, value in state_dict.items():
        new_key = convert_key_to_cpp(key)
        converted_dict[new_key] = value
        
        # Store first few examples for display
        if len(conversion_examples) < 10 and key != new_key:
            conversion_examples.append((key, new_key))
    
    print(f"Converted {len(converted_dict)} tensors")
    
    # Show conversion examples
    if conversion_examples:
        print("\nKey conversion examples:")
        for old, new in conversion_examples:
            print(f"  {old}")
        print("  →")
        for old, new in conversion_examples:
            print(f"  {new}")
    
    # Save as a standard PyTorch checkpoint
    print(f"\nSaving converted checkpoint to: {output_path}")
    torch.save(converted_dict, output_path)
    
    print("\nConversion complete!")
    print(f"  - Original keys: {len(state_dict)}")
    print(f"  - Converted keys: {len(converted_dict)}")
    
    return True


def print_key_samples(input_path: str, count: int = 30):
    """Print sample keys from the checkpoint for debugging."""
    print(f"Loading keys from: {input_path}")
    
    try:
        state_dict = torch.load(input_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print(f"\nTotal keys: {len(state_dict)}")
    print(f"\nFirst {count} keys:")
    
    for i, key in enumerate(list(state_dict.keys())[:count]):
        new_key = convert_key_to_cpp(key)
        shape = list(state_dict[key].shape)
        print(f"  {key}")
        if key != new_key:
            print(f"    → {new_key}")
        print(f"    shape: {shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Python VGGT model weights to C++ compatible format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input model.pt file (Python format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output converted checkpoint (default: <input>_cpp.pt)"
    )
    parser.add_argument(
        "--samples", "-s",
        action="store_true",
        help="Print sample keys and exit (no conversion)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if args.samples:
        print_key_samples(args.input)
        return 0
    
    if args.output is None:
        # Generate output path from input
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_cpp{input_path.suffix}")
    
    success = convert_weights(args.input, args.output)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
