#!/usr/bin/env python3
"""
Parallel 3-Way Comparison with Custom Prompts
==============================================
Tests all 3 implementations with YOUR prompts in parallel across multiple GPUs

GPU Assignment:
- GPU 0: BASELINE (Standard PyTorch)
- GPU 1: OFFICIAL (thinking-machines-lab/batch_invariant_ops)
- GPU 2: CUSTOM (Our CUDA implementation)

Usage:
    python tests/compare_parallel_custom.py --prompts prompts.txt --runs 1000
    python tests/compare_parallel_custom.py --runs 1000  # Uses default prompts
"""

import sys
import os
import argparse
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Queue
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Try to import official batch_invariant_ops
OFFICIAL_AVAILABLE = False
try:
    batch_ops_path = os.environ.get('BATCH_INVARIANT_OPS_PATH', 
                                     str(Path(__file__).parent.parent / 'batch_invariant_ops'))
    if not os.path.exists(batch_ops_path):
        batch_ops_path = os.path.expanduser('~/batch_invariant_ops')
    
    if os.path.exists(batch_ops_path):
        sys.path.insert(0, batch_ops_path)
        OFFICIAL_AVAILABLE = True
except:
    pass


# DEFAULT COMPLEX PROMPTS (designed to expose non-determinism)
DEFAULT_PROMPTS = [
    # Complex reasoning prompts
    "Analyze the following scenario and provide a detailed recommendation: A startup has $50,000 in funding and needs to decide between hiring 2 junior developers for $25,000 each or 1 senior developer for $45,000. They have 3 months to build an MVP. Consider technical debt, time to market, code quality, and future scalability in your analysis.",
    
    # Multi-step problem solving
    "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons? Provide step-by-step instructions and explain your reasoning for each step.",
    
    # Code generation with edge cases
    "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes algorithm. Include: proper type hints, docstring with complexity analysis, handling of edge cases (n<2, n=2, large n), and an example usage. Optimize for readability while maintaining O(n log log n) time complexity.",
    
    # Creative writing with constraints
    "Write a short story (exactly 3 paragraphs) about a programmer who discovers their code is sentient. Include: a twist ending, dialogue, and a metaphor about consciousness. Each paragraph should be 4-5 sentences.",
    
    # Data analysis scenario
    "Given a dataset of customer purchases with columns: customer_id, product_id, quantity, price, timestamp. Write SQL queries to: 1) Find top 5 customers by total spend, 2) Identify products frequently bought together (basket analysis), 3) Calculate month-over-month growth rate. Explain the rationale for each query.",
]


def load_prompts_from_file(filepath: str) -> list:
    """Load prompts from a text file (one prompt per line)"""
    with open(filepath, 'r') as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return prompts


def test_worker(implementation: str, device_id: int, model_name: str, prompts: list, 
                max_tokens: int, num_runs: int, result_queue: Queue):
    """
    Generic worker that tests any implementation
    
    Args:
        implementation: 'BASELINE', 'OFFICIAL', or 'CUSTOM'
        device_id: CUDA device ID
        prompts: List of prompts to test
        num_runs: Number of runs per prompt
    """
    
    try:
        # Set device
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        
        print(f"[{implementation} GPU:{device_id}] Loading model...")
        
        # Load model
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
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        print(f"[{implementation} GPU:{device_id}] Model loaded. Testing {len(prompts)} prompts with {num_runs} runs each...")
        
        # Setup context manager based on implementation
        if implementation == 'OFFICIAL':
            from batch_invariant_ops import set_batch_invariant_mode
            context_mgr = lambda: set_batch_invariant_mode(True)
        elif implementation == 'CUSTOM':
            from batch_invariant_ops_cuda import set_cuda_batch_invariant_mode
            context_mgr = lambda: set_cuda_batch_invariant_mode(True)
        else:  # BASELINE
            from contextlib import nullcontext
            context_mgr = lambda: nullcontext()
        
        # Test each prompt
        all_results = {}
        
        for prompt_idx, prompt in enumerate(prompts):
            print(f"[{implementation} GPU:{device_id}] Prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:60]}...")
            
            outputs = []
            times = []
            
            for run in range(num_runs):
                start = time.time()
                
                with context_mgr():
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        torch.cuda.synchronize(device)
                        result = model.generate(
                            **inputs,
                            generation_config=generation_config,
                            max_new_tokens=max_tokens
                        )
                        torch.cuda.synchronize(device)
                    
                    output = tokenizer.decode(
                        result[0][inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                
                outputs.append(output)
                times.append(time.time() - start)
                
                if (run + 1) % 100 == 0:
                    print(f"[{implementation} GPU:{device_id}]   Run {run + 1}/{num_runs}")
            
            unique = len(set(outputs))
            avg_time = sum(times) / len(times)
            
            all_results[f"prompt_{prompt_idx}"] = {
                'prompt': prompt,
                'unique': unique,
                'total': num_runs,
                'avg_time': avg_time,
                'outputs': outputs,
                'deterministic': unique == 1
            }
            
            print(f"[{implementation} GPU:{device_id}]   Result: {unique}/{num_runs} unique outputs, {avg_time:.3f}s avg")
        
        # Calculate overall statistics
        total_unique = sum(r['unique'] for r in all_results.values())
        total_runs = sum(r['total'] for r in all_results.values())
        avg_time_overall = sum(r['avg_time'] for r in all_results.values()) / len(all_results)
        all_deterministic = all(r['deterministic'] for r in all_results.values())
        
        result_queue.put({
            'name': implementation,
            'all_results': all_results,
            'total_unique': total_unique,
            'total_runs': total_runs,
            'avg_time': avg_time_overall,
            'all_deterministic': all_deterministic,
            'num_prompts': len(prompts)
        })
        
        print(f"[{implementation} GPU:{device_id}] ✓ Complete: {total_unique} total unique outputs across all prompts")
        
    except Exception as e:
        print(f"[{implementation} GPU:{device_id}] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({'name': implementation, 'error': str(e)})


def main():
    parser = argparse.ArgumentParser(description='Parallel 3-Way Determinism Comparison with Custom Prompts')
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct', help='Model name')
    parser.add_argument('--prompts', help='Path to prompts file (one per line)')
    parser.add_argument('--max-tokens', type=int, default=200, help='Max tokens to generate')
    parser.add_argument('--runs', type=int, default=100, help='Number of runs per prompt')
    parser.add_argument('--save-results', help='Save detailed results to JSON file')
    
    args = parser.parse_args()
    
    # Load prompts
    if args.prompts:
        print(f"Loading prompts from: {args.prompts}")
        prompts = load_prompts_from_file(args.prompts)
        print(f"✓ Loaded {len(prompts)} prompts")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"Using {len(prompts)} default complex prompts")
    
    print("\n" + "="*80)
    print(" " * 20 + "PARALLEL 3-WAY COMPARISON")
    print(" " * 18 + f"Testing {len(prompts)} Prompts on 3 GPUs")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Runs per prompt: {args.runs}")
    print(f"Total runs: {len(prompts) * args.runs * 3} across all implementations")
    print("="*80)
    
    # Show prompts
    print("\nPrompts to test:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt[:70]}...")
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available")
        return 1
    
    num_gpus = torch.cuda.device_count()
    print(f"\n✓ {num_gpus} GPU(s) available")
    
    for i in range(min(3, num_gpus)):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)")
    
    if num_gpus < 3:
        print("\n⚠ Warning: Less than 3 GPUs available. Tests will run sequentially on available GPUs.")
    
    print(f"\nGPU Assignment:")
    print(f"  GPU 0: BASELINE")
    print(f"  GPU 1: OFFICIAL {'(requires python3-devel)' if not OFFICIAL_AVAILABLE else ''}")
    print(f"  GPU 2: CUSTOM")
    print()
    
    # Create result queue
    result_queue = Queue()
    
    print("="*80)
    print("Starting parallel tests...")
    print("="*80)
    print()
    
    start_time = time.time()
    
    processes = []
    
    # Process 1: BASELINE on GPU 0
    p1 = mp.Process(
        target=test_worker,
        args=('BASELINE', 0, args.model, prompts, args.max_tokens, args.runs, result_queue)
    )
    p1.start()
    processes.append(p1)
    
    # Process 2: OFFICIAL on GPU 1 (if available)
    if OFFICIAL_AVAILABLE:
        p2 = mp.Process(
            target=test_worker,
            args=('OFFICIAL', 1, args.model, prompts, args.max_tokens, args.runs, result_queue)
        )
        p2.start()
        processes.append(p2)
    else:
        print("[OFFICIAL] Skipped - batch_invariant_ops not available or python3-devel not installed")
        result_queue.put({'name': 'OFFICIAL', 'error': 'Not available - install python3-devel'})
    
    # Process 3: CUSTOM on GPU 2
    p3 = mp.Process(
        target=test_worker,
        args=('CUSTOM', 2, args.model, prompts, args.max_tokens, args.runs, result_queue)
    )
    p3.start()
    processes.append(p3)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # Collect results
    results = {}
    while not result_queue.empty():
        result = result_queue.get()
        results[result['name']] = result
    
    # Display detailed results
    print("\n" + "="*80)
    print(" " * 25 + "DETAILED RESULTS")
    print("="*80)
    
    for impl_name in ['BASELINE', 'OFFICIAL', 'CUSTOM']:
        if impl_name not in results:
            continue
        
        res = results[impl_name]
        
        print(f"\n{impl_name}:")
        print("-" * 80)
        
        if 'error' in res:
            print(f"  ❌ ERROR: {res['error']}")
            continue
        
        if 'all_results' not in res:
            print(f"  ⚠ No results available")
            continue
        
        # Show per-prompt results
        for prompt_key, prompt_res in res['all_results'].items():
            prompt_num = int(prompt_key.split('_')[1]) + 1
            unique = prompt_res['unique']
            total = prompt_res['total']
            avg_time = prompt_res['avg_time']
            is_det = prompt_res['deterministic']
            
            status = "✓" if is_det else "✗"
            print(f"  Prompt {prompt_num}: {unique:3d}/{total} unique | {avg_time:.3f}s | {status}")
        
        # Overall stats
        print(f"\n  Overall:")
        print(f"    Total unique outputs: {res['total_unique']}")
        print(f"    Total runs: {res['total_runs']}")
        print(f"    Avg time per run: {res['avg_time']:.3f}s")
        print(f"    All deterministic: {'YES ✓' if res['all_deterministic'] else 'NO ✗'}")
    
    # Summary table
    print("\n" + "="*80)
    print(" " * 30 + "SUMMARY")
    print("="*80)
    print(f"\nTotal parallel execution time: {total_time:.2f}s")
    if len(processes) > 1:
        print(f"Sequential would have taken: ~{total_time * len(processes):.2f}s")
        print(f"Speedup: {len(processes):.1f}x\n")
    
    print(f"{'Implementation':<15} {'All Det?':<10} {'Unique/Total':<15} {'Avg Time':<12} {'Status'}")
    print("-" * 80)
    
    for name in ['BASELINE', 'OFFICIAL', 'CUSTOM']:
        if name not in results:
            continue
        
        res = results[name]
        
        if 'error' in res:
            print(f"{name:<15} {'ERROR':<10} {'N/A':<15} {'N/A':<12} SKIP")
        else:
            det_str = "YES ✓" if res['all_deterministic'] else "NO ✗"
            unique_str = f"{res['total_unique']}/{res['total_runs']}"
            time_str = f"{res['avg_time']:.3f}s"
            status = "PASS" if res['all_deterministic'] else "FAIL"
            
            print(f"{name:<15} {det_str:<10} {unique_str:<15} {time_str:<12} {status}")
    
    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    deterministic = [
        name for name, res in results.items() 
        if 'error' not in res and res.get('all_deterministic', False)
    ]
    
    if not deterministic:
        print("\n❌ No implementation achieved determinism across all prompts")
        print("\n   Non-deterministic prompts reveal differences in:")
        print("   • Floating-point reduction order")
        print("   • Batch-size dependent operations")
        print("   • GPU scheduling variance")
    else:
        print(f"\n✓ {len(deterministic)} implementation(s) achieved determinism:")
        for name in deterministic:
            print(f"   • {name}: {results[name]['avg_time']:.3f}s average, {results[name]['total_unique']} total unique")
        
        if len(deterministic) > 1:
            fastest = min(deterministic, key=lambda x: results[x]['avg_time'])
            print(f"\n✓ Fastest deterministic: {fastest} ({results[fastest]['avg_time']:.3f}s)")
            
            # Check if outputs match across implementations
            print(f"\n" + "="*80)
            print("OUTPUT CONSISTENCY CHECK")
            print("="*80)
            
            all_match = True
            for prompt_idx in range(len(prompts)):
                prompt_key = f"prompt_{prompt_idx}"
                
                # Get outputs from all deterministic implementations
                impl_outputs = {}
                for impl in deterministic:
                    if prompt_key in results[impl]['all_results']:
                        impl_outputs[impl] = results[impl]['all_results'][prompt_key]['outputs'][0]
                
                # Check if all match
                if len(set(impl_outputs.values())) == 1:
                    print(f"  Prompt {prompt_idx + 1}: ✓ All implementations produce identical output")
                else:
                    print(f"  Prompt {prompt_idx + 1}: ✗ Implementations produce different outputs")
                    all_match = False
            
            if all_match:
                print("\n🎉 All deterministic implementations produce IDENTICAL outputs for ALL prompts!")
            else:
                print("\n⚠ Deterministic implementations produce different outputs")
                print("   This may indicate different reduction strategies or numerical precision")
    
    # Save results if requested
    if args.save_results:
        print(f"\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Prepare serializable results
        save_data = {
            'model': args.model,
            'num_prompts': len(prompts),
            'runs_per_prompt': args.runs,
            'max_tokens': args.max_tokens,
            'total_time': total_time,
            'results': {}
        }
        
        for name, res in results.items():
            if 'error' not in res:
                save_data['results'][name] = {
                    'all_deterministic': res['all_deterministic'],
                    'total_unique': res['total_unique'],
                    'total_runs': res['total_runs'],
                    'avg_time': res['avg_time'],
                    'per_prompt': {
                        k: {
                            'unique': v['unique'],
                            'deterministic': v['deterministic'],
                            'avg_time': v['avg_time'],
                            'sample_output': v['outputs'][0] if v['outputs'] else None
                        }
                        for k, v in res['all_results'].items()
                    }
                }
        
        with open(args.save_results, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"✓ Results saved to: {args.save_results}")
    
    print("\n" + "="*80)
    
    return 0 if deterministic else 1


if __name__ == "__main__":
    # Required for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)
    sys.exit(main())