#!/bin/bash
# Setup script for 3-way comparison test
# Run this before comparing implementations

set -e

echo "=============================================================================="
echo "  Setup: 3-Way Determinism Comparison"
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

# Check if official batch_invariant_ops exists
echo ""
echo "4. Checking for official batch_invariant_ops..."

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
    echo "  To enable OFFICIAL comparison:"
    echo "  1. Clone repo:"
    echo "     cd ~"
    echo "     git clone https://github.com/thinking-machines-lab/batch_invariant_ops.git"
    echo ""
    echo "  2. (Optional) Set path:"
    echo "     export BATCH_INVARIANT_OPS_PATH=~/batch_invariant_ops"
    echo ""
    echo "  Without it, only BASELINE and CUSTOM tests will run."
fi

# Check Triton (for official implementation)
echo ""
echo "5. Checking Triton (for OFFICIAL test)..."
python3 << EOF
try:
    import triton
    print(f"✓ Triton installed: {triton.__version__}")
except ImportError:
    print("⚠ Triton not installed")
    print("  Install with: pip install triton")
    print("  Required for OFFICIAL test")
EOF

# Test our custom implementation
echo ""
echo "6. Testing CUSTOM implementation..."
python3 << EOF
import sys
import os
sys.path.insert(0, 'src')

try:
    from batch_invariant_ops_cuda import verify_cuda_determinism
    print("✓ Custom batch_invariant_ops_cuda found")
    
    if verify_cuda_determinism():
        print("✓ CUDA determinism configured correctly")
    else:
        print("⚠ CUDA determinism checks failed")
except Exception as e:
    print(f"❌ Failed to load custom implementation: {e}")
    exit(1)
EOF

# Summary
echo ""
echo "=============================================================================="
echo "  Setup Complete"
echo "=============================================================================="
echo ""
echo "You can now run the comparison:"
echo ""
echo "  # Quick test (100 runs)"
echo "  python tests/compare_implementations.py --runs 100"
echo ""
echo "  # Comprehensive test (1000 runs)"
echo "  python tests/compare_implementations.py --runs 1000"
echo ""
echo "  # Custom prompt"
echo "  python tests/compare_implementations.py --prompt 'Your prompt here'"
echo ""
echo "=============================================================================="