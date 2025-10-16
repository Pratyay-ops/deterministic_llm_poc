#!/bin/bash
# Cleanup script - removes old/duplicate files
# Keep only the new clean structure

set -e

echo "================================================================================"
echo "  Cleaning Up Repository"
echo "================================================================================"

cd /home/ec2-user/pr43308/deterministic_llm_poc

# Backup first (optional)
# echo -e "\n1. Creating backup..."
# Uncomment if you want backup:
# tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz --exclude=venv --exclude=batch_invariant_ops .
echo "  (Skipped - uncomment in script to enable)"

# 2. Remove old/duplicate files from src/
echo -e "\n2. Removing old files from src/..."
rm -f src/batch_invariant_ops_cuda.py          # Duplicate (we have utils/ version)
rm -f src/deterministic_cuda_inference.py      # Old implementation
rm -f src/test_determinism.py                  # Moved to tests/
rm -f src/utils.py                             # Old utils
rm -f src/validation.py                        # Old validation
echo "✓ Removed 5 old files from src/"

# 3. Remove old test files
echo -e "\n3. Removing old test files..."
rm -f tests/benchmark.py                       # Old benchmark
rm -f tests/compare_3_versions.py             # Old comparison
rm -f tests/compare_implementations.py         # Old comparison
rm -f tests/test.py                            # Empty test file
echo "✓ Removed 4 old test files"

# 4. Remove old examples (keep if you want them)
echo -e "\n4. Removing old examples..."
rm -rf examples/
echo "✓ Removed examples/ directory"

# 5. Remove old scripts
echo -e "\n5. Removing old/duplicate scripts..."
rm -f scripts/download_model.py                # Redundant
rm -f scripts/run_validation.sh               # Old validation
rm -f scripts/setup_complete.sh                # Duplicate
echo "✓ Removed 3 old scripts"

# 6. Remove old setup.py
echo -e "\n6. Removing old setup.py..."
rm -f setup.py
echo "✓ Removed setup.py"

# 7. Keep my_prompts.txt but move to configs
echo -e "\n7. Organizing prompts..."
if [ -f "my_prompts.txt" ]; then
    mv my_prompts.txt configs/prompts/ 2>/dev/null || echo "  Already moved"
    echo "✓ Moved my_prompts.txt to configs/prompts/"
fi

# 8. Show what's left
echo -e "\n8. Current clean structure:"
echo "--------------------------------------------------------------------------------"

find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.txt" \) \
    ! -path "./venv/*" \
    ! -path "./.git/*" \
    ! -path "./batch_invariant_ops/*" \
    ! -path "*/__pycache__/*" \
    | sort

echo -e "\n================================================================================"
echo "  Cleanup Complete!"
echo "================================================================================"
echo -e "\nKept:"
echo "  ✓ src/implementations/     (baseline.py, official.py, custom.py)"
echo "  ✓ src/engines/             (engine_base.py, vllm_engine.py, hf_engine.py)"
echo "  ✓ src/utils/               (Empty, ready for future)"
echo "  ✓ tests/compare_parallel.py"
echo "  ✓ configs/models/          (Model configs)"
echo "  ✓ configs/prompts/         (Prompt files)"
echo "  ✓ scripts/setup.sh"
echo "  ✓ requirements.txt"
echo ""
echo "Removed:"
echo "  ✗ Duplicate files (batch_invariant_ops_cuda.py from src/)"
echo "  ✗ Old implementations (deterministic_cuda_inference.py)"
echo "  ✗ Old tests (compare_3_versions.py, compare_implementations.py)"
echo "  ✗ Old scripts (download_model.py, run_validation.sh)"
echo "  ✗ examples/ directory"
echo "  ✗ setup.py"
echo ""
echo "Next: Run your test!"
echo "  python tests/compare_parallel.py \\"
echo "    --model 'Qwen/Qwen2.5-1.5B-Instruct' \\"
echo "    --prompts configs/prompts/complex_prompts.txt \\"
echo "    --engine vllm \\"
echo "    --runs 1000"
echo "================================================================================"