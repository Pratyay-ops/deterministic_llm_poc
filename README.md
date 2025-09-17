# Deterministic LLM Inference

Implementation of batch-invariant operations for deterministic LLM inference on Apple Silicon, based on research from [Thinking Machines Lab](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

## Features

- ✅ 100% deterministic inference
- ✅ Optimized for M4 GPU (Metal Performance Shaders)
- ✅ Batch-invariant operations
- ✅ Support for Qwen 1.5B/3B models
- ✅ Entity extraction, code generation, parsing

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run validation
python tests/test_determinism.py

# Basic usage
python examples/basic_usage.py

```
