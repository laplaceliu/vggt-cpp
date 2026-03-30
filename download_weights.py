#!/usr/bin/env python3
"""
Download pretrained VGGT weights from Hugging Face.

Usage:
    python download_weights.py [--output OUTPUT_DIR] [--token TOKEN]

Example:
    python download_weights.py --output ./weights
    python download_weights.py --token hf_xxx
    
Environment Variables:
    HF_TOKEN: Hugging Face access token (recommended)

Get your token from: https://huggingface.co/settings/tokens
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, login
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please install it with: pip install huggingface_hub")
    sys.exit(1)


def download_vggt_weights(output_dir: str = "./weights", token: str = None):
    """
    Download VGGT pretrained weights from Hugging Face.

    The official model is available at: facebook/VGGT-1B
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Login if token provided
    if token:
        print("Logging in to Hugging Face...")
        login(token=token)
        print("Login successful!")
    elif os.environ.get("HF_TOKEN"):
        print("Using HF_TOKEN from environment...")
        login(token=os.environ.get("HF_TOKEN"))
        print("Login successful!")

    repo_id = "facebook/VGGT-1B"

    # Files to download
    files_to_download = [
        "model.pt",  # Main model weights
    ]

    print(f"Downloading VGGT weights from {repo_id}...")
    print(f"Output directory: {output_path.absolute()}")
    print()

    downloaded_files = []

    for filename in files_to_download:
        try:
            print(f"Downloading {filename}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=output_dir
            )
            downloaded_files.append(downloaded_path)
            print(f"  Saved to: {downloaded_path}")
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")
            print()
            print("Alternative: You can manually download the weights from:")
            print(f"  https://huggingface.co/{repo_id}")
            print()

    if downloaded_files:
        print()
        print("=" * 60)
        print("Download complete!")
        print("=" * 60)
        print()
        print("To use the pretrained weights with demo_vggt:")
        print(f"  ./demo_vggt -m {output_dir}/model.pt -i image1.jpg,image2.jpg")
        print()
        return True
    else:
        print()
        print("=" * 60)
        print("Download failed!")
        print("=" * 60)
        print()
        print("Please manually download the weights from:")
        print(f"  https://huggingface.co/{repo_id}")
        print()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained VGGT weights from Hugging Face"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./weights",
        help="Output directory for downloaded weights (default: ./weights)"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="Hugging Face access token (or set HF_TOKEN env var)"
    )

    args = parser.parse_args()

    success = download_vggt_weights(args.output, args.token)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
