# tests/benchmark.py
"""
Performance benchmarking with variance analysis
Following Thinking Machines Lab methodology
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import DeterministicMPSModel
from src.batch_invariant_ops_mps import set_mps_batch_invariant_mode
import time
import statistics
import torch
import hashlib
from collections import Counter
from tqdm import tqdm
import json
from datetime import datetime

class BenchmarkWithVariance:
    def __init__(self):
        self.model = None
        self.results = []
    
    def initialize_model(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        """Initialize or reinitialize model"""
        if self.model:
            del self.model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        self.model = DeterministicMPSModel(model_name)
    
    def benchmark_variance(self, prompt, task_name, iterations=1000, max_tokens=100, batch_invariant=True):
        """
        Benchmark with variance measurement - Thinking Machines Lab approach
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {task_name}")
        print(f"Iterations: {iterations}")
        print(f"Batch Invariant: {batch_invariant}")
        print(f"{'='*60}")
        
        outputs = []
        hashes = []
        times = []
        
        pbar = tqdm(range(iterations), desc=task_name)
        
        for i in pbar:
            start = time.time()
            
            if batch_invariant:
                with set_mps_batch_invariant_mode(True):
                    output, _ = self.model.generate(prompt, max_new_tokens=max_tokens)
            else:
                with set_mps_batch_invariant_mode(False):
                    output, _ = self.model.generate(prompt, max_new_tokens=max_tokens)
            
            elapsed = time.time() - start
            
            outputs.append(output)
            hashes.append(hashlib.sha256(output.encode()).hexdigest())
            times.append(elapsed)
            
            # Update progress
            unique_count = len(set(hashes))
            pbar.set_postfix({
                'unique': unique_count,
                'avg_ms': f"{statistics.mean(times)*1000:.1f}"
            })
        
        # Analyze results
        unique_outputs = len(set(hashes))
        hash_counter = Counter(hashes)
        most_common_count = hash_counter.most_common(1)[0][1]
        
        return {
            'task': task_name,
            'iterations': iterations,
            'unique_outputs': unique_outputs,
            'deterministic': unique_outputs == 1,
            'most_common_frequency': most_common_count,
            'avg_time': statistics.mean(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'tokens_per_sec': max_tokens / statistics.mean(times)
        }

def benchmark_performance():
    """Original benchmark function - enhanced"""
    
    print("Performance Benchmark")
    print("=" * 60)
    
    benchmark = BenchmarkWithVariance()
    benchmark.initialize_model("Qwen/Qwen2.5-1.5B-Instruct")
    
    # Test cases
    test_cases = [
        {
            "name": "Entity Extraction",
            "prompt": "Extract all company names from: Apple Inc. announced a partnership with Microsoft Corporation and Google LLC.",
            "iterations": 100  # Start with 100 for quick test
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to calculate factorial:",
            "iterations": 100
        }
    ]
    
    results_without_bi = []
    results_with_bi = []
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        # Without batch invariance
        benchmark.initialize_model()
        result_without = benchmark.benchmark_variance(
            test_case['prompt'],
            f"{test_case['name']} (Standard)",
            iterations=test_case['iterations'],
            batch_invariant=False
        )
        results_without_bi.append(result_without)
        
        # With batch invariance
        benchmark.initialize_model()
        result_with = benchmark.benchmark_variance(
            test_case['prompt'],
            f"{test_case['name']} (Batch-Invariant)",
            iterations=test_case['iterations'],
            batch_invariant=True
        )
        results_with_bi.append(result_with)
        
        # Print comparison
        print(f"\n{test_case['name']} Results:")
        print(f"  Without BI: {result_without['unique_outputs']} unique outputs, {result_without['avg_time']*1000:.1f}ms avg")
        print(f"  With BI:    {result_with['unique_outputs']} unique outputs, {result_with['avg_time']*1000:.1f}ms avg")
        print(f"  Determinism achieved: {result_with['deterministic']}")
        print(f"  Performance impact: {((result_with['avg_time']/result_without['avg_time'])-1)*100:.1f}%")
    
    # Save results
    save_benchmark_results(results_without_bi, results_with_bi)

def save_benchmark_results(results_without, results_with):
    """Save benchmark results to file"""
    os.makedirs("results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/benchmark_{timestamp}.json"
    
    data = {
        "timestamp": timestamp,
        "without_batch_invariance": results_without,
        "with_batch_invariance": results_with
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {filename}")

def run_full_benchmark():
    """Run full 1000-iteration benchmark like Thinking Machines Lab"""
    print("\n" + "="*70)
    print("FULL BENCHMARK - 1000 ITERATIONS")
    print("Replicating Thinking Machines Lab Methodology")
    print("="*70)
    
    benchmark = BenchmarkWithVariance()
    
    # Full test with 1000 iterations
    test_cases = [
        {
            "name": "Entity Extraction", 
            "prompt": "Extract entities from: Tim Cook, CEO of Apple Inc., met with Satya Nadella from Microsoft at the Google campus in Mountain View.",
            "iterations": 1000,
            "max_tokens": 150
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function that implements binary search on a sorted list:",
            "iterations": 1000,
            "max_tokens": 200
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'#'*70}")
        print(f"Task: {test_case['name']}")
        print(f"{'#'*70}")
        
        # Test without batch invariance
        benchmark.initialize_model()
        result_without = benchmark.benchmark_variance(
            test_case['prompt'],
            f"{test_case['name']} - Standard",
            iterations=test_case['iterations'],
            max_tokens=test_case['max_tokens'],
            batch_invariant=False
        )
        
        # Test with batch invariance
        benchmark.initialize_model()
        result_with = benchmark.benchmark_variance(
            test_case['prompt'],
            f"{test_case['name']} - Batch Invariant",
            iterations=test_case['iterations'],
            max_tokens=test_case['max_tokens'],
            batch_invariant=True
        )
        
        # Print detailed comparison
        print(f"\n{'='*60}")
        print(f"RESULTS: {test_case['name']}")
        print(f"{'='*60}")
        print(f"\nWithout Batch Invariance:")
        print(f"  Unique outputs: {result_without['unique_outputs']}/{result_without['iterations']}")
        print(f"  Most common frequency: {result_without['most_common_frequency']}")
        print(f"  Avg time: {result_without['avg_time']:.3f}s")
        print(f"  Throughput: {result_without['tokens_per_sec']:.1f} tokens/s")
        
        print(f"\nWith Batch Invariance:")
        print(f"  Unique outputs: {result_with['unique_outputs']}/{result_with['iterations']}")
        print(f"  Most common frequency: {result_with['most_common_frequency']}")
        print(f"  Avg time: {result_with['avg_time']:.3f}s")
        print(f"  Throughput: {result_with['tokens_per_sec']:.1f} tokens/s")
        
        print(f"\nImprovement:")
        print(f"  Variance reduction: {result_without['unique_outputs']} → {result_with['unique_outputs']}")
        print(f"  Determinism achieved: {'✅' if result_with['deterministic'] else '❌'}")
        print(f"  Performance impact: {((result_with['avg_time']/result_without['avg_time'])-1)*100:.1f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_benchmark()
    else:
        benchmark_performance()
        print("\nRun with --full flag for 1000-iteration benchmark")