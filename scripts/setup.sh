#!/bin/bash
# Complete setup for deterministic LLM inference
# Supports: Qwen 2.5, Qwen 3 VL, Hunyuan

set -e

echo "================================================================================"
echo "  Deterministic LLM Inference - Complete Setup"
echo "================================================================================"

# 1. Check CUDA
echo -e "\n1. Checking CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found"
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "✓ CUDA available"

# 2. Check Python
echo -e "\n2. Checking Python..."
python3 --version
echo "✓ Python available"

# 3. Install core dependencies
echo -e "\n3. Installing core dependencies..."
pip install -q torch transformers accelerate sentencepiece protobuf pyyaml
echo "✓ Core installed"

# 4. Install vLLM
echo -e "\n4. Installing vLLM..."
pip install -q vllm
echo "✓ vLLM installed"

# 5. Install Triton (for OFFICIAL)
echo -e "\n5. Installing Triton..."
pip install -q triton
echo "✓ Triton installed"

# 6. Check python3-devel (for OFFICIAL Triton compilation)
echo -e "\n6. Checking python3-devel..."
if python3 -c "import sysconfig; print(sysconfig.get_path('include'))" &> /dev/null; then
    if [ -f "$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")/Python.h" ]; then
        echo "✓ Python.h found"
    else
        echo "⚠ Python.h not found - OFFICIAL test may fail"
        echo "  Install: sudo dnf install python3-devel gcc gcc-c++"
    fi
fi

# 7. Clone batch_invariant_ops
echo -e "\n7. Setting up batch_invariant_ops..."
if [ ! -d "batch_invariant_ops" ]; then
    git clone -q https://github.com/thinking-machines-lab/batch_invariant_ops.git
    echo "✓ Cloned batch_invariant_ops"
else
    echo "✓ batch_invariant_ops already present"
fi

# 8. Download model
echo -e "\n8. Downloading model..."
python3 << 'PYEOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"  Downloading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print("  ✓ Model cached")
PYEOF

# 9. Set environment
echo -e "\n9. Configuring environment..."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "export CUBLAS_WORKSPACE_CONFIG=:4096:8" >> ~/.bashrc
echo "✓ CUBLAS configured"

echo -e "\n================================================================================"
echo "  Setup Complete!"
echo "================================================================================"
echo -e "\nRun comparison test:"
echo "  python compare_parallel.py --prompts complex_prompts.txt --runs 1000"
echo "================================================================================"
