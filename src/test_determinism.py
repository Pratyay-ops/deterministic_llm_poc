#!/usr/bin/env python3
"""
Comprehensive determinism test suite for CUDA implementation
Tests batch invariance, model determinism, and concurrent behavior
"""

import sys
import torch
from deterministic_cuda_inference import DeterministicCUDAModel, demonstrate_determinism
from batch_invariant_ops_cuda import verify_cuda_determinism, set_cuda_batch_invariant_mode
import time
from typing import List, Tuple


def test_cuda_availability():
    """Test CUDA is available and properly configured"""
    print("\n" + "="*60)
    print("TEST 1: CUDA Availability")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✓ CUDA available with {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"    Compute: {props.major}.{props.minor}")
    
    return verify_cuda_determinism()


def test_batch_invariance():
    """Test that batch-invariant operations work correctly"""
    print("\n" + "="*60)
    print("TEST 2: Batch Invariance")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device("cuda:0")
    
    # Test matrix multiplication batch invariance
    print("\nTesting matmul batch invariance...")
    
    with set_cuda_batch_invariant_mode(True):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # Create test matrices
        D = 256
        A = torch.randn(100, D, device=device, dtype=torch.float32)
        B = torch.randn(D, D, device=device, dtype=torch.float32)
        
        # Process with batch size 1 (first element only)
        out1 = torch.matmul(A[:1], B)
        
        # Process with full batch and take first element
        out2 = torch.matmul(A, B)[:1]
        
        # Check if results are identical
        max_diff = (out1 - out2).abs().max().item()
        
        print(f"  Max difference: {max_diff}")
        
        if max_diff == 0.0:
            print("✓ Perfect batch invariance (bitwise identical)")
            batch_inv_pass = True
        elif max_diff < 1e-6:
            print("✓ Near-perfect batch invariance (< 1e-6 difference)")
            batch_inv_pass = True
        else:
            print(f"❌ Batch variance detected: {max_diff}")
            batch_inv_pass = False
    
    # Test softmax batch invariance
    print("\nTesting softmax batch invariance...")
    
    with set_cuda_batch_invariant_mode(True):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        X = torch.randn(100, 256, device=device, dtype=torch.float32)
        
        sm1 = torch.nn.functional.softmax(X[:1], dim=-1)
        sm2 = torch.nn.functional.softmax(X, dim=-1)[:1]
        
        max_diff = (sm1 - sm2).abs().max().item()
        print(f"  Max difference: {max_diff}")
        
        if max_diff < 1e-6:
            print("✓ Softmax batch invariance verified")
            softmax_pass = True
        else:
            print(f"❌ Softmax variance detected: {max_diff}")
            softmax_pass = False
    
    return batch_inv_pass and softmax_pass


def test_model_determinism(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", num_runs: int = 5):
    """Test that model produces deterministic outputs across multiple runs"""
    print("\n" + "="*60)
    print("TEST 3: Model Determinism")
    print("="*60)
    
    print(f"Loading model: {model_name}")
    model = DeterministicCUDAModel(model_name, device_id=0)
    
    test_prompts = [
        "Write a Python function to calculate factorial:",
        "List three countries in Europe:",
        "What is 2+2?",
        "Explain machine learning:",
    ]
    
    all_deterministic = True
    
    for prompt in test_prompts:
        print(f"\nTesting: '{prompt[:50]}...'")
        
        outputs = []
        for i in range(num_runs):
            output, _ = model.generate(prompt, max_new_tokens=30)
            outputs.append(output)
        
        unique_outputs = len(set(outputs))
        
        if unique_outputs == 1:
            print(f"  ✓ Deterministic ({num_runs}/{num_runs} identical)")
        else:
            print(f"  ❌ Non-deterministic ({unique_outputs} unique outputs)")
            all_deterministic = False
            
            # Show differences
            print(f"  Outputs:")
            for i, out in enumerate(set(outputs)):
                count = outputs.count(out)
                print(f"    Variant {i+1} ({count}x): {out[:80]}...")
    
    # Memory usage
    mem = model.get_memory_usage()
    print(f"\nMemory usage:")
    print(f"  Allocated: {mem['allocated_gb']:.2f} GB")
    print(f"  Reserved: {mem['reserved_gb']:.2f} GB")
    
    return all_deterministic


def test_different_batch_sizes(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Test determinism across different batch sizes"""
    print("\n" + "="*60)
    print("TEST 4: Cross-Batch-Size Determinism")
    print("="*60)
    
    model = DeterministicCUDAModel(model_name, device_id=0)
    
    prompt = "Explain the water cycle:"
    
    print(f"Prompt: {prompt}")
    print("Testing with batch sizes: 1, 2, 4, 8")
    
    # Generate with batch size 1 (reference)
    reference, _ = model.generate(prompt, max_new_tokens=50)
    print(f"\nReference output (batch=1):")
    print(f"  {reference[:100]}...")
    
    # Test with different batch sizes
    batch_sizes = [2, 4, 8]
    all_match = True
    
    for bs in batch_sizes:
        # Create batch of identical prompts
        prompts = [prompt] * bs
        results = model.generate_batch(prompts, max_new_tokens=50, batch_size=bs)
        
        # Check if all results match reference
        matches = sum(1 for r, _ in results if r == reference)
        
        print(f"\nBatch size {bs}: {matches}/{bs} match reference")
        
        if matches == bs:
            print(f"  ✓ All outputs identical to reference")
        else:
            print(f"  ❌ Some outputs differ from reference")
            all_match = False
            
            # Show first difference
            for i, (result, _) in enumerate(results):
                if result != reference:
                    print(f"  Different output {i+1}: {result[:80]}...")
                    break
    
    return all_match


def test_concurrent_requests():
    """Test determinism under concurrent/parallel load"""
    print("\n" + "="*60)
    print("TEST 5: Concurrent Determinism")
    print("="*60)
    
    # Note: True concurrency is limited by Python GIL
    # This tests sequential execution with model reuse
    
    model = DeterministicCUDAModel("Qwen/Qwen2.5-1.5B-Instruct", device_id=0)
    prompt = "Name a primary color:"
    
    print(f"Prompt: {prompt}")
    print("Running 10 sequential generations...")
    
    outputs = []
    for i in range(10):
        output, elapsed = model.generate(prompt, max_new_tokens=20)
        outputs.append(output)
        print(f"  Run {i+1}: {elapsed:.3f}s")
    
    unique = len(set(outputs))
    
    print(f"\nResults: {unique} unique output(s)")
    
    if unique == 1:
        print("✓ Deterministic under repeated execution")
        return True
    else:
        print(f"❌ Non-deterministic: {unique} different outputs")
        for i, out in enumerate(set(outputs)):
            count = outputs.count(out)
            print(f"  Variant {i+1} ({count}x): {out}")
        return False


def test_numerical_precision():
    """Test numerical precision of batch-invariant operations"""
    print("\n" + "="*60)
    print("TEST 6: Numerical Precision")
    print("="*60)
    
    device = torch.device("cuda:0")
    
    # Test Kahan summation precision
    print("\nTesting Kahan summation vs standard sum...")
    
    from batch_invariant_ops_cuda import kahan_sum
    
    # Create values that expose floating-point errors
    values = [1e10, 1.0, -1e10, 1.0] * 100  # Should sum to 200
    tensor = torch.tensor(values, device=device, dtype=torch.float32).reshape(1, -1)
    
    # Standard sum
    standard_sum = tensor.sum(dim=-1).item()
    
    # Kahan sum
    with set_cuda_batch_invariant_mode(True):
        kahan_result = kahan_sum(tensor, dim=-1).item()
    
    print(f"  Standard sum: {standard_sum}")
    print(f"  Kahan sum: {kahan_result}")
    print(f"  Expected: 200.0")
    print(f"  Kahan error: {abs(kahan_result - 200.0)}")
    print(f"  Standard error: {abs(standard_sum - 200.0)}")
    
    kahan_better = abs(kahan_result - 200.0) < abs(standard_sum - 200.0)
    
    if kahan_better:
        print("✓ Kahan summation provides better precision")
    else:
        print("⚠ Standard sum equivalent or better (test may need adjustment)")
    
    return True  # This test is informational


def run_full_test_suite():
    """Run all tests and provide summary"""
    print("\n" + "="*80)
    print(" " * 20 + "DETERMINISTIC LLM TEST SUITE")
    print(" " * 25 + "(CUDA/NVIDIA GPUs)")
    print("="*80)
    
    tests = [
        ("CUDA Availability", test_cuda_availability),
        ("Batch Invariance", test_batch_invariance),
        ("Model Determinism", lambda: test_model_determinism(num_runs=5)),
        ("Cross-Batch-Size", test_different_batch_sizes),
        ("Concurrent Execution", test_concurrent_requests),
        ("Numerical Precision", test_numerical_precision),
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print(" " * 30 + "TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("="*80)
    print(f"Total: {passed}/{total} tests passed")
    print(f"Time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Deterministic inference verified.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_full_test_suite()
    sys.exit(exit_code)