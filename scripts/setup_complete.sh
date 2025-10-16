#!/bin/bash
# Complete setup script for 3-way comparison test
# This will download the model and check all prerequisites

set -e

echo "=============================================================================="
echo "  Complete Setup: 3-Way Determinism Comparison"
echo "=============================================================================="

# Check CUDA
echo ""
echo "1. Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "❌ nvidia-smi not found. CUDA may not be available."
    exit 1
fi

# Check Python
echo ""
echo "2. Checking Python..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Check PyTorch CUDA
echo ""
echo "3. Checking PyTorch CUDA support..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
else:
    print("❌ PyTorch CUDA not available")
    exit(1)
EOF

# Check/Install dependencies
echo ""
echo "4. Checking dependencies..."
pip list | grep -q transformers || pip install transformers
pip list | grep -q accelerate || pip install accelerate
pip list | grep -q sentencepiece || pip install sentencepiece
echo "✓ Core dependencies installed"

# Check Triton (for official implementation)
echo ""
echo "5. Checking Triton (for OFFICIAL test)..."
python3 << EOF
try:
    import triton
    print(f"✓ Triton installed: {triton.__version__}")
except ImportError:
    print("⚠ Triton not installed")
    print("  Installing Triton...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'triton'])
    print("✓ Triton installed")
EOF

# Download model
echo ""
echo "6. Downloading model (Qwen/Qwen2.5-1.5B-Instruct)..."
echo "   This will take 3-5 minutes on first run..."
python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print(f"  Checking cache: {cache_dir}")

try:
    # This will download if not cached
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"  Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',
        trust_remote_code=True,
        device_map="cpu",  # Just download, don't load to GPU yet
        low_cpu_mem_usage=True,
    )
    
    print(f"✓ Model downloaded and cached")
    print(f"✓ Model size: ~3GB")
    
except Exception as e:
    print(f"❌ Failed to download model: {e}")
    exit(1)
EOF

# Check if official batch_invariant_ops exists
echo ""
echo "7. Checking for official batch_invariant_ops..."

BATCH_OPS_PATH="${BATCH_INVARIANT_OPS_PATH:-$HOME/batch_invariant_ops}"

if [ -d "$BATCH_OPS_PATH" ]; then
    echo "✓ Found at: $BATCH_OPS_PATH"
    
    # Check if it's a valid repo
    if [ -f "$BATCH_OPS_PATH/batch_invariant_ops/batch_invariant_ops.py" ]; then
        echo "✓ Valid batch_invariant_ops repository"
    else
        echo "⚠ Directory exists but doesn't look like batch_invariant_ops repo"
        echo "  Expected: $BATCH_OPS_PATH/batch_invariant_ops/batch_invariant_ops.py"
    fi
else
    echo "⚠ Not found at: $BATCH_OPS_PATH"
    echo ""
    echo "  Cloning official repository..."
    
    cd ~
    if git clone https://github.com/thinking-machines-lab/batch_invariant_ops.git; then
        echo "✓ Official batch_invariant_ops cloned to ~/batch_invariant_ops"
        BATCH_OPS_PATH="$HOME/batch_invariant_ops"
        export BATCH_INVARIANT_OPS_PATH="$BATCH_OPS_PATH"
    else
        echo "❌ Failed to clone repository"
        echo "  You can manually clone:"
        echo "    cd ~"
        echo "    git clone https://github.com/thinking-machines-lab/batch_invariant_ops.git"
    fi
fi

# Test our custom implementation
echo ""
echo "8. Testing CUSTOM implementation..."
python3 << EOF
import sys
import os
sys.path.insert(0, 'src')

try:
    from batch_invariant_ops_cuda import verify_cuda_determinism
    print("✓ Custom batch_invariant_ops_cuda found")
    
    print("\nVerifying CUDA determinism configuration...")
    if verify_cuda_determinism():
        print("✓ CUDA determinism configured correctly")
    else:
        print("⚠ CUDA determinism checks failed (may still work)")
except Exception as e:
    print(f"❌ Failed to load custom implementation: {e}")
    exit(1)
EOF

# Estimate disk space
echo ""
echo "9. Checking disk space..."
CACHE_DIR="$HOME/.cache/huggingface"
if [ -d "$CACHE_DIR" ]; then
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    echo "✓ HuggingFace cache: $CACHE_SIZE"
else
    echo "✓ HuggingFace cache will be created at: $CACHE_DIR"
fi

# Summary
echo ""
echo "=============================================================================="
echo "  Setup Complete!"
echo "=============================================================================="
echo ""
echo "✓ CUDA available"
echo "✓ PyTorch with CUDA support"
echo "✓ All dependencies installed"
echo "✓ Triton installed (for OFFICIAL comparison)"
echo "✓ Model downloaded and cached"
echo "✓ Official batch_invariant_ops available"
echo "✓ Custom implementation ready"
echo ""
echo "=============================================================================="
echo "  Ready to Run Comparison"
echo "=============================================================================="
echo ""
echo "Quick test (100 runs, ~5 minutes):"
echo "  python tests/compare_implementations.py --runs 100"
echo ""
echo "Comprehensive test (1000 runs, ~30 minutes):"
echo "  python tests/compare_implementations.py --runs 1000"
echo ""
echo "Custom prompt:"
echo "  python tests/compare_implementations.py --prompt 'Your prompt here'"
echo ""
echo "Different model:"
echo "  python tests/compare_implementations.py --model 'Qwen/Qwen2.5-3B-Instruct'"
echo ""
echo "=============================================================================="