#!/usr/bin/env python3
"""
3-Way Determinism Comparison
=============================
Compares 3 implementations:
1. BASELINE: Standard PyTorch (non-deterministic)
2. OFFICIAL: thinking-machines-lab/batch_invariant_ops (Triton kernels)
3. CUSTOM: Our CUDA implementation (this repo)

Usage:
    python tests/compare_implementations.py --runs 100
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
from collections import defaultdict

# Try to import official batch_invariant_ops
OFFICIAL_AVAILABLE = False
try:
    # User needs to: git clone https://github.com/thinking-machines-lab/batch_invariant_ops.git
    # Then set env: export BATCH_INVARIANT_OPS_PATH=/path/to/batch_invariant_ops
    batch_ops_path = os.environ.get('BATCH_INVARIANT_OPS_PATH', 
                                     os.path.expanduser('~/batch_invariant_ops'))
    if os.path.exists(batch_ops_path):
        sys.path.insert(0, batch_ops_path)
        from batch_invariant_ops import set_batch_invariant_mode as official_batch_mode
        OFFICIAL_AVAILABLE = True
        print(f"✓ Official batch_invariant_ops found at: {batch_ops_path}")
    else:
        print(f"⚠ Official batch_invariant_ops not found at: {batch_ops_path}")
        print(f"  Clone: git clone https://github.com/thinking-machines-lab/batch_invariant_ops.git")
        print(f"  Set: export BATCH_INVARIANT_OPS_PATH=/path/to/batch_invariant_ops")
except ImportError as e:
    print(f"⚠ Could not import official batch_invariant_ops: {e}")

# Import our custom implementation
from batch_invariant_ops_cuda import set_cuda_batch_invariant_mode as custom_batch_mode


def load_model(model_name: str, device_id: int = 0):
    """Load model and tokenizer"""
    device = torch.device(f"cuda:{device_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    generation_config = GenerationConfig(
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        top_k=1,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    return model, tokenizer, generation_config, device


def test_baseline(model, tokenizer, generation_config, device, prompt: str, 
                   max_tokens: int, num_runs: int):
    """Test 1: Baseline (standard PyTorch)"""
    print("\n" + "="*80)
    print("TEST 1: BASELINE (Standard PyTorch)")
    print("="*80)
    
    outputs = []
    times = []
    
    for i in range(num_runs):
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            torch.cuda.synchronize(device)
            result = model.generate(
                **inputs,
                generation_config=generation_config,
                max_new_tokens=max_tokens
            )
            torch.cuda.synchronize(device)
        
        output = tokenizer.decode(result[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        outputs.append(output)
        times.append(time.time() - start)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_runs}")
    
    unique = len(set(outputs))
    avg_time = sum(times) / len(times)
    
    print(f"\nResults:")
    print(f"  Unique outputs: {unique}/{num_runs}")
    print(f"  Deterministic: {'YES ✓' if unique == 1 else 'NO ✗'}")
    print(f"  Avg time: {avg_time:.3f}s")
    
    if unique > 1 and unique <= 5:
        print(f"\n  Sample variants:")
        output_counts = defaultdict(int)
        for out in outputs:
            output_counts[out] += 1
        
        for i, (out, count) in enumerate(sorted(output_counts.items(), key=lambda x: -x[1])[:3], 1):
            print(f"    {i}. ({count}x): {out[:70]}...")
    
    return unique == 1, avg_time, outputs


def test_official(model, tokenizer, generation_config, device, prompt: str,
                   max_tokens: int, num_runs: int):
    """Test 2: Official batch_invariant_ops (Triton)"""
    print("\n" + "="*80)
    print("TEST 2: OFFICIAL (thinking-machines-lab/batch_invariant_ops)")
    print("="*80)
    
    if not OFFICIAL_AVAILABLE:
        print("✗ Official batch_invariant_ops not available")
        print("  Skipping this test")
        return None, 0.0, []
    
    outputs = []
    times = []
    
    for i in range(num_runs):
        start = time.time()
        
        with official_batch_mode(True):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                torch.cuda.synchronize(device)
                result = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    max_new_tokens=max_tokens
                )
                torch.cuda.synchronize(device)
            
            output = tokenizer.decode(result[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        outputs.append(output)
        times.append(time.time() - start)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_runs}")
    
    unique = len(set(outputs))
    avg_time = sum(times) / len(times)
    
    print(f"\nResults:")
    print(f"  Unique outputs: {unique}/{num_runs}")
    print(f"  Deterministic: {'YES ✓' if unique == 1 else 'NO ✗'}")
    print(f"  Avg time: {avg_time:.3f}s")
    
    if unique > 1 and unique <= 5:
        print(f"\n  Sample variants:")
        output_counts = defaultdict(int)
        for out in outputs:
            output_counts[out] += 1
        
        for i, (out, count) in enumerate(sorted(output_counts.items(), key=lambda x: -x[1])[:3], 1):
            print(f"    {i}. ({count}x): {out[:70]}...")
    
    return unique == 1, avg_time, outputs


def test_custom(model, tokenizer, generation_config, device, prompt: str,
                 max_tokens: int, num_runs: int):
    """Test 3: Our custom implementation (this repo)"""
    print("\n" + "="*80)
    print("TEST 3: CUSTOM (Our CUDA Implementation)")
    print("="*80)
    
    outputs = []
    times = []
    
    for i in range(num_runs):
        start = time.time()
        
        with custom_batch_mode(True):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                torch.cuda.synchronize(device)
                result = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    max_new_tokens=max_tokens
                )
                torch.cuda.synchronize(device)
            
            output = tokenizer.decode(result[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        outputs.append(output)
        times.append(time.time() - start)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_runs}")
    
    unique = len(set(outputs))
    avg_time = sum(times) / len(times)
    
    print(f"\nResults:")
    print(f"  Unique outputs: {unique}/{num_runs}")
    print(f"  Deterministic: {'YES ✓' if unique == 1 else 'NO ✗'}")
    print(f"  Avg time: {avg_time:.3f}s")
    
    if unique > 1 and unique <= 5:
        print(f"\n  Sample variants:")
        output_counts = defaultdict(int)
        for out in outputs:
            output_counts[out] += 1
        
        for i, (out, count) in enumerate(sorted(output_counts.items(), key=lambda x: -x[1])[:3], 1):
            print(f"    {i}. ({count}x): {out[:70]}...")
    
    return unique == 1, avg_time, outputs


def main():
    parser = argparse.ArgumentParser(description='3-Way Determinism Comparison')
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct', help='Model name')
    parser.add_argument('--prompt', default='Write a Python function to calculate factorial:', 
                        help='Test prompt')
    parser.add_argument('--max-tokens', type=int, default=50, help='Max tokens to generate')
    parser.add_argument('--runs', type=int, default=100, help='Number of runs per test')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    
    args = parser.parse_args()
    
    print("="*80)
    print(" " * 25 + "3-WAY COMPARISON TEST")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Runs per test: {args.runs}")
    print(f"Device: cuda:{args.device}")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    print(f"\nGPU: {torch.cuda.get_device_name(args.device)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    
    # Load model
    print(f"\nLoading model...")
    model, tokenizer, generation_config, device = load_model(args.model, args.device)
    print(f"✓ Model loaded\n")
    
    # Run tests
    results = {}
    
    det_baseline, time_baseline, outs_baseline = test_baseline(
        model, tokenizer, generation_config, device, args.prompt, args.max_tokens, args.runs
    )
    results['BASELINE'] = {
        'deterministic': det_baseline,
        'time': time_baseline,
        'outputs': outs_baseline
    }
    
    det_official, time_official, outs_official = test_official(
        model, tokenizer, generation_config, device, args.prompt, args.max_tokens, args.runs
    )
    if det_official is not None:
        results['OFFICIAL'] = {
            'deterministic': det_official,
            'time': time_official,
            'outputs': outs_official
        }
    
    det_custom, time_custom, outs_custom = test_custom(
        model, tokenizer, generation_config, device, args.prompt, args.max_tokens, args.runs
    )
    results['CUSTOM'] = {
        'deterministic': det_custom,
        'time': time_custom,
        'outputs': outs_custom
    }
    
    # Summary
    print("\n" + "="*80)
    print(" " * 30 + "FINAL SUMMARY")
    print("="*80)
    print(f"\n{'Implementation':<20} {'Deterministic':<15} {'Avg Time':<12} {'Status'}")
    print("-" * 80)
    
    for name, res in results.items():
        det_str = "YES ✓" if res['deterministic'] else "NO ✗"
        time_str = f"{res['time']:.3f}s"
        status = "PASS" if res['deterministic'] else "FAIL"
        print(f"{name:<20} {det_str:<15} {time_str:<12} {status}")
    
    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    deterministic = [name for name, res in results.items() if res['deterministic']]
    
    if not deterministic:
        print("\n❌ No implementation achieved determinism")
        print("   (Note: BASELINE is expected to fail)")
    else:
        print(f"\n✓ {len(deterministic)} implementation(s) achieved determinism:")
        for name in deterministic:
            print(f"   • {name}: {results[name]['time']:.3f}s average")
        
        if len(deterministic) > 1:
            fastest = min(deterministic, key=lambda x: results[x]['time'])
            print(f"\n✓ Fastest deterministic: {fastest}")
    
    # Check if outputs match between deterministic implementations
    if len(deterministic) > 1:
        print(f"\n" + "="*80)
        print("OUTPUT CONSISTENCY CHECK")
        print("="*80)
        
        outputs_list = [results[name]['outputs'][0] for name in deterministic]
        if len(set(outputs_list)) == 1:
            print("✓ All deterministic implementations produce IDENTICAL outputs")
        else:
            print("⚠ Deterministic implementations produce DIFFERENT outputs")
            for name in deterministic:
                print(f"\n{name}:")
                print(f"  {results[name]['outputs'][0][:100]}...")
    
    print("\n" + "="*80)
    
    return 0 if deterministic else 1


if __name__ == "__main__":
    sys.exit(main())