#!/usr/bin/env python3
"""
Parallel 3-Way Comparison Test
Runs BASELINE, OFFICIAL, CUSTOM in parallel on separate GPUs
"""

import sys
import os
import argparse
import multiprocessing as mp
from multiprocessing import Queue
import time
import json
from pathlib import Path

# Set CUBLAS before any CUDA operations
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch


def load_prompts(filepath: str) -> list:
    """Load prompts from file (one per line, # for comments)"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def test_worker(impl_name: str, engine_type: str, device_id: int, model_name: str,
                prompts: list, max_tokens: int, num_runs: int, result_queue: Queue):
    """Worker process for testing one implementation"""
    
    try:
        # CRITICAL: Add repo root to path for worker processes
        repo_root = Path(__file__).parent.parent.absolute()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        
        # Import implementation
        if impl_name == "BASELINE":
            from src.implementations.baseline import BaselineImplementation as Implementation
        elif impl_name == "OFFICIAL":
            from src.implementations.official import OfficialImplementation as Implementation
        else:  # CUSTOM
            from src.implementations.custom import CustomImplementation as Implementation
        
        # Initialize
        impl = Implementation(
            model_name=model_name,
            device_id=device_id,
            engine=engine_type,
            dtype=torch.float16
        )
        
        print(f"[{impl_name} GPU:{device_id}] Testing {len(prompts)} prompts × {num_runs} runs")
        
        all_results = {}
        
        for prompt_idx, prompt in enumerate(prompts):
            print(f"[{impl_name} GPU:{device_id}] Prompt {prompt_idx + 1}/{len(prompts)}")
            
            outputs, times = [], []
            
            for run in range(num_runs):
                output, elapsed = impl.generate(prompt, max_new_tokens=max_tokens)
                outputs.append(output)
                times.append(elapsed)
                
                if (run + 1) % 100 == 0:
                    print(f"[{impl_name} GPU:{device_id}]   {run + 1}/{num_runs}")
            
            unique = len(set(outputs))
            all_results[f"prompt_{prompt_idx}"] = {
                'unique': unique,
                'total': num_runs,
                'avg_time': sum(times) / len(times),
                'outputs': outputs,
                'deterministic': unique == 1,
                'prompt': prompt
            }
            
            print(f"[{impl_name} GPU:{device_id}]   {unique}/{num_runs} unique")
        
        # Overall stats
        total_unique = sum(r['unique'] for r in all_results.values())
        all_deterministic = all(r['deterministic'] for r in all_results.values())
        avg_time = sum(r['avg_time'] for r in all_results.values()) / len(all_results)
        
        result_queue.put({
            'name': impl_name,
            'all_results': all_results,
            'total_unique': total_unique,
            'all_deterministic': all_deterministic,
            'avg_time': avg_time,
        })
        
        print(f"[{impl_name} GPU:{device_id}] ✓ Complete")
        
        impl.cleanup()
        
    except Exception as e:
        print(f"[{impl_name} GPU:{device_id}] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({'name': impl_name, 'error': str(e)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--prompts', required=True, help='Path to prompts file')
    parser.add_argument('--engine', default='vllm', choices=['vllm', 'hf'])
    parser.add_argument('--max-tokens', type=int, default=200)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--save', help='Save results JSON')
    parser.add_argument('--skip-official', action='store_true', help='Skip OFFICIAL (if no python3-devel)')
    
    args = parser.parse_args()
    
    # Load prompts
    prompts = load_prompts(args.prompts)
    
    print("="*80)
    print("PARALLEL 3-WAY COMPARISON")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Engine: {args.engine.upper()}")
    print(f"Prompts: {len(prompts)}")
    print(f"Runs/prompt: {args.runs}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Total runs: {len(prompts) * args.runs * (2 if args.skip_official else 3)}")
    print("="*80)
    
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p[:70]}...")
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available")
        return 1
    
    num_gpus = torch.cuda.device_count()
    print(f"\n✓ {num_gpus} GPUs available")
    
    # Start workers
    result_queue = Queue()
    processes = []
    start_time = time.time()
    
    # GPU 0: BASELINE
    p1 = mp.Process(target=test_worker, args=(
        'BASELINE', args.engine, 0, args.model, prompts, args.max_tokens, args.runs, result_queue
    ))
    p1.start()
    processes.append(p1)
    
    # GPU 1: OFFICIAL (if not skipped)
    if not args.skip_official:
        p2 = mp.Process(target=test_worker, args=(
            'OFFICIAL', args.engine, 1, args.model, prompts, args.max_tokens, args.runs, result_queue
        ))
        p2.start()
        processes.append(p2)
    
    # GPU 2: CUSTOM
    p3 = mp.Process(target=test_worker, args=(
        'CUSTOM', args.engine, 2, args.model, prompts, args.max_tokens, args.runs, result_queue
    ))
    p3.start()
    processes.append(p3)
    
    # Wait
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # Collect results
    results = {}
    while not result_queue.empty():
        result = result_queue.get()
        results[result['name']] = result
    
    # Display
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for name in ['BASELINE', 'OFFICIAL', 'CUSTOM']:
        if name not in results:
            continue
        
        res = results[name]
        
        if 'error' in res:
            print(f"\n{name}: ERROR - {res['error']}")
            continue
        
        print(f"\n{name}:")
        print(f"  All deterministic: {'YES ✓' if res['all_deterministic'] else 'NO ✗'}")
        print(f"  Total unique: {res['total_unique']}/{len(prompts) * args.runs}")
        print(f"  Avg time: {res['avg_time']:.3f}s")
    
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Speedup: {len(processes)}x")
    
    # Save
    if args.save:
        with open(args.save, 'w') as f:
            # Remove outputs to save space
            for impl in results.values():
                if 'all_results' in impl:
                    for pr in impl['all_results'].values():
                        pr['sample'] = pr['outputs'][0] if pr['outputs'] else None
                        del pr['outputs']
            
            json.dump({'results': results, 'config': vars(args)}, f, indent=2)
        print(f"\n✓ Saved to {args.save}")
    
    print("="*80)
    return 0


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    sys.exit(main())