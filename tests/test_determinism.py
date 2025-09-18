# tests/test_determinism.py
"""Main determinism test suite - enhanced with variance analysis"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import DeterministicMPSModel, check_mps_availability
from src.validation import validate_batch_invariance, validate_model_outputs
from src.batch_invariant_ops_mps import set_mps_batch_invariant_mode
import time
import hashlib
from collections import Counter

def test_batch_invariance():
    """Test that batch-invariant ops work correctly"""
    print("Testing Batch Invariance...")
    
    if not check_mps_availability():
        print("❌ MPS not available")
        return False
    
    result = validate_batch_invariance()
    
    if result:
        print("✅ Batch invariance test passed")
    else:
        print("❌ Batch invariance test failed")
    
    return result

def test_model_determinism():
    """Test model produces deterministic outputs"""
    print("\nTesting Model Determinism...")
    
    model = DeterministicMPSModel("Qwen/Qwen2.5-1.5B-Instruct")
    
    test_cases = [
        "Extract entities from: Microsoft announced a new product.",
        "Write a function to calculate factorial:",
        "Explain this code: x = lambda n: n*2"
    ]
    
    all_pass = True
    
    for prompt in test_cases:
        print(f"\nTesting: {prompt[:40]}...")
        is_det, outputs = validate_model_outputs(model, prompt)
        
        if is_det:
            print(f"✅ Deterministic")
        else:
            print(f"❌ Non-deterministic: {len(set(outputs))} unique outputs")
            all_pass = False
    
    return all_pass

def test_variance_comparison():
    """
    New test: Compare variance with and without batch invariance
    Following Thinking Machines Lab methodology
    """
    print("\n" + "="*60)
    print("Testing Variance With/Without Batch Invariance")
    print("="*60)
    
    model = DeterministicMPSModel("Qwen/Qwen2.5-1.5B-Instruct")
    
    test_prompt = "Extract companies from: Apple, Microsoft, and Google announced new AI products."
    num_iterations = 50  # Quick test
    
    # Test WITHOUT batch invariance
    print("\nWithout Batch Invariance:")
    outputs_without = []
    with set_mps_batch_invariant_mode(False):
        for i in range(num_iterations):
            output, _ = model.generate(test_prompt, max_new_tokens=50)
            outputs_without.append(output)
    
    unique_without = len(set(outputs_without))
    print(f"  Unique outputs: {unique_without}/{num_iterations}")
    
    # Test WITH batch invariance
    print("\nWith Batch Invariance:")
    outputs_with = []
    with set_mps_batch_invariant_mode(True):
        for i in range(num_iterations):
            output, _ = model.generate(test_prompt, max_new_tokens=50)
            outputs_with.append(output)
    
    unique_with = len(set(outputs_with))
    print(f"  Unique outputs: {unique_with}/{num_iterations}")
    
    # Check improvement
    improvement = unique_without > unique_with
    deterministic = unique_with == 1
    
    print(f"\nResults:")
    print(f"  Variance reduced: {'✅' if improvement else '❌'} ({unique_without} → {unique_with})")
    print(f"  Fully deterministic: {'✅' if deterministic else '❌'}")
    
    return deterministic

def test_concurrent_requests():
    """Test determinism under concurrent load"""
    print("\nTesting Concurrent Determinism...")
    
    from concurrent.futures import ThreadPoolExecutor
    
    model = DeterministicMPSModel("Qwen/Qwen2.5-1.5B-Instruct")
    prompt = "List three programming languages:"
    
    def generate(_):
        output, _ = model.generate(prompt, max_new_tokens=30)
        return output
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate, i) for i in range(10)]
        results = [f.result() for f in futures]
    
    unique = len(set(results))
    
    if unique == 1:
        print("✅ Deterministic under concurrent load")
        return True
    else:
        print(f"❌ {unique} unique responses under concurrent load")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DETERMINISTIC LLM VALIDATION SUITE")
    print("Based on thinking-machines-lab/batch_invariant_ops")
    print("=" * 60)
    
    # Run all tests
    batch_pass = test_batch_invariance()
    model_pass = test_model_determinism()
    variance_pass = test_variance_comparison()  # New test
    concurrent_pass = test_concurrent_requests()
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Batch Invariance:    {'✅ PASS' if batch_pass else '❌ FAIL'}")
    print(f"Model Determinism:   {'✅ PASS' if model_pass else '❌ FAIL'}")
    print(f"Variance Comparison: {'✅ PASS' if variance_pass else '❌ FAIL'}")
    print(f"Concurrent:          {'✅ PASS' if concurrent_pass else '❌ FAIL'}")
    
    if batch_pass and model_pass and variance_pass and concurrent_pass:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nRun 'python tests/benchmark.py --full' for 1000-iteration benchmark")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed")
        sys.exit(1)