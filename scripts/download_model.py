#!/usr/bin/env python3
"""
Download model before running comparison tests
This downloads and caches the model to avoid delays during testing
"""

import argparse
import os
import sys
from pathlib import Path

def download_model(model_name: str, force: bool = False):
    """Download and cache a HuggingFace model"""
    
    print("="*80)
    print(f"Model Download: {model_name}")
    print("="*80)
    
    # Import here so we can check if packages are installed
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        print(f"\n❌ Missing required packages: {e}")
        print("\nInstall with:")
        print("  pip install transformers torch accelerate")
        return False
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n⚠ Warning: CUDA not available")
        print("  Model will be downloaded but may not run on GPU")
    else:
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Check cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"\nCache directory: {cache_dir}")
    
    if os.path.exists(cache_dir):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        ) / (1024**3)  # Convert to GB
        print(f"Current cache size: {cache_size:.2f} GB")
    else:
        print("Cache directory will be created")
    
    # Download tokenizer
    print(f"\n{'='*80}")
    print("Step 1: Downloading tokenizer...")
    print('='*80)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            force_download=force
        )
        print(f"✓ Tokenizer downloaded")
        print(f"  Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed to download tokenizer: {e}")
        return False
    
    # Download model
    print(f"\n{'='*80}")
    print("Step 2: Downloading model weights...")
    print("This may take 3-5 minutes depending on your connection")
    print('='*80)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",  # Don't load to GPU yet, just download
            low_cpu_mem_usage=True,
            force_download=force
        )
        print(f"\n✓ Model downloaded")
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Model size: ~{total_params * 2 / (1024**3):.2f} GB (FP16)")
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False
    
    # Verify cache
    print(f"\n{'='*80}")
    print("Verification")
    print('='*80)
    
    if os.path.exists(cache_dir):
        new_cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        ) / (1024**3)
        print(f"✓ Cache size: {new_cache_size:.2f} GB")
    
    print(f"\n{'='*80}")
    print("Download Complete!")
    print('='*80)
    print(f"\nModel '{model_name}' is now cached and ready to use.")
    print("\nYou can now run:")
    print(f"  python tests/compare_implementations.py --model '{model_name}'")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download model before running comparison tests'
    )
    parser.add_argument(
        '--model',
        default='Qwen/Qwen2.5-1.5B-Instruct',
        help='Model name to download (default: Qwen/Qwen2.5-1.5B-Instruct)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if cached'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Show recommended models'
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nRecommended models for testing:\n")
        print("Small models (good for quick testing):")
        print("  • Qwen/Qwen2.5-1.5B-Instruct  (~3 GB)")
        print("  • Qwen/Qwen2.5-3B-Instruct    (~6 GB)")
        return 0
    
    success = download_model(args.model, args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())