"""Main determinism test suite"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import DeterministicMPSModel, check_mps_availability
from src.validation import validate_batch_invariance, validate_model_outputs
import time

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
    concurrent_pass = test_concurrent_requests()
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Batch Invariance:  {'✅ PASS' if batch_pass else '❌ FAIL'}")
    print(f"Model Determinism: {'✅ PASS' if model_pass else '❌ FAIL'}")
    print(f"Concurrent:        {'✅ PASS' if concurrent_pass else '❌ FAIL'}")
    
    if batch_pass and model_pass and concurrent_pass:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed")
        sys.exit(1)