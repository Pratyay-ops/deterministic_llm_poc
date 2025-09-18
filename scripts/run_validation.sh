#!/bin/bash
# scripts/run_validation.sh

echo "Running Deterministic LLM Validation"
echo "===================================="

# Check Python version
python3 --version

# Run determinism tests
echo -e "\n1. Testing determinism..."
python3 tests/test_determinism.py

# Run quick benchmark
echo -e "\n2. Running quick benchmark..."
python3 tests/benchmark.py

# Ask about full benchmark
echo -e "\n3. Run full 1000-iteration benchmark? (y/n)"
read -r response
if [[ "$response" == "y" ]]; then
    echo "Running full benchmark (this will take a while)..."
    python3 tests/benchmark.py --full
fi

echo -e "\nValidation complete!"