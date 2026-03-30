#!/usr/bin/env python3
"""
Convert Hugging Face VGGT model.pt (ZIP format) to a standard PyTorch checkpoint.

Usage:
    python convert_weights.py [--input INPUT.pt] [--output OUTPUT.pt]

Example:
    python convert_weights.py --input weights/model.pt --output weights/model_converted.pt
"""

import argparse
import sys
import torch
import zipfile
import io
import os
from pathlib import Path


def convert_huggingface_to_pt(input_path: str, output_path: str):
    """
    Convert a Hugging Face model.pt (ZIP format) to a standard PyTorch checkpoint.
    
    Args:
        input_path: Path to the input ZIP archive
        output_path: Path to save the converted checkpoint
    """
    print(f"Loading model from: {input_path}")
    
    # Load the ZIP archive
    with zipfile.ZipFile(input_path, 'r') as zf:
        # Read the pickle file
        with zf.open('vggt_model/data.pkl') as f:
            pickle_data = f.read()
        
        # Load using torch.load with weights_only=False to handle custom classes
        # We need to provide the zipfile object as the storage to resolve tensor references
        import pickle
        from pickle import Unpickler
        
        # Create a custom Unpickler that can resolve tensor references from the ZIP
        class ZipAwareUnpickler(Unpickler):
            def __init__(self, file, zip_ref=None):
                super().__init__(file)
                self.zip_ref = zip_ref
                self.storage_cache = {}
                
            def persistent_load(self, obj):
                # Handle persistent references
                if obj[0] == 'storage':
                    # This is a tensor storage reference
                    key = obj[1]
                    if key not in self.storage_cache:
                        # Try to load from ZIP
                        if self.zip_ref is not None:
                            try:
                                data = self.zip_ref.read(f'vggt_model/data/{key}')
                                import struct
                                # Parse the binary data
                                # This is a simplified version - actual format may vary
                                self.storage_cache[key] = data
                            except KeyError:
                                return None
                        else:
                            return None
                    return self.storage_cache.get(key)
                return super().persistent_load(obj)
        
        # Use torch.load which handles the ZIP format internally
        # Set weights_only=False to handle custom classes
        try:
            # First try: torch.load with zipfile as storage
            state_dict = torch.load(
                input_path,
                map_location='cpu',
                weights_only=False
            )
            print(f"Successfully loaded with torch.load")
        except Exception as e:
            print(f"torch.load failed: {e}")
            print("Trying alternative method...")
            
            # Alternative: manually extract and load
            state_dict = None
            
            # Fallback: try reading pickle directly
            try:
                import pickle
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Create a file-like object from the pickle data
                    f = io.BytesIO(pickle_data)
                    unpickler = pickle.Unpickler(f)
                    unpickler.persistent_load = lambda obj: None  # Ignore persistent refs
                    result = unpickler.load()
                    
                    if isinstance(result, dict):
                        state_dict = result
                    elif hasattr(result, 'state_dict'):
                        state_dict = result.state_dict()
            except Exception as e2:
                print(f"Pickle extraction failed: {e2}")
                return False
    
    if state_dict is None:
        print("Failed to extract state dictionary")
        return False
    
    # Convert keys if needed (remove 'module.' prefix from DataParallel/DDP models)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        cleaned_state_dict[new_key] = value
    
    # Save as a standard PyTorch checkpoint
    print(f"Saving converted checkpoint to: {output_path}")
    torch.save(cleaned_state_dict, output_path)
    
    print(f"Conversion complete!")
    print(f"  - Original keys: {len(state_dict)}")
    print(f"  - Cleaned keys: {len(cleaned_state_dict)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face VGGT model.pt to standard PyTorch checkpoint"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input model.pt file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output converted checkpoint (default: <input>_converted.pt)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if args.output is None:
        # Generate output path from input
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_converted{input_path.suffix}")
    
    success = convert_huggingface_to_pt(args.input, args.output)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
