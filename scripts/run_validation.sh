#!/bin/bash

echo "Running Deterministic LLM Validation"
echo "===================================="

# Check Python version
python3 --version

# Run tests
echo -e "\n1. Testing batch invariance..."
python3 tests/test_determinism.py

echo -e "\n2. Running benchmarks..."
python3 tests/benchmark.py

echo -e "\n3. Testing examples..."
python3 examples/basic_usage.py

echo -e "\nValidation complete!"