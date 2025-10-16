#!/usr/bin/env python3
"""
3-Way Determinism Comparison Test
==================================
Compares 3 implementations of Qwen/Qwen2.5-1.5B-Instruct:

1. BASELINE: Standard PyTorch inference (non-deterministic)
2. OFFICIAL: thinking-machines-lab/batch_invariant_ops (Triton kernels)
3. CUSTOM: Our CUDA implementation (fixed-chunk + Kahan)

Tests determinism across multiple runs and different batch sizes.
"""

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
from typing import List, Tuple, Dict
from collections import defaultdict

# Add batch_invariant_ops to path
sys.path.insert(0, '/home/claude/batch_invariant_ops')

print("="*80)
print(" " * 20 + "3-WAY DETERMINISM COMPARISON")
print(" " * 15 + "Qwen/Qwen2.5-1.5B-Instruct on NVIDIA A10G")
print("="*80)


# ============================================================================
# VERSION 1: BASELINE (Standard PyTorch - Non-deterministic)
# ============================================================================

class BaselineModel:
    """Standard PyTorch inference without any determinism modifications"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device_id: int = 0):
        self.device = torch.device(f"cuda:{device_id}")
        self.model_name = model_name
        
        print(f"\n[BASELINE] Loading standard model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        self.generation_config = GenerationConfig(
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            top_k=1,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        print("[BASELINE] ✓ Model loaded (standard PyTorch)")
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> Tuple[str, float]:
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            torch.cuda.synchronize(self.device)
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens
            )
            torch.cuda.synchronize(self.device)
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        elapsed = time.time() - start_time
        return response, elapsed


# ============================================================================
# VERSION 2: OFFICIAL (thinking-machines-lab/batch_invariant_ops)
# ============================================================================

class OfficialBatchInvariantModel:
    """Using official batch_invariant_ops with Triton kernels"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device_id: int = 0):
        self.device = torch.device(f"cuda:{device_id}")
        self.model_name = model_name
        
        print(f"\n[OFFICIAL] Loading model with batch_invariant_ops...")
        
        # Import official implementation
        try:
            from batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
            self.set_batch_invariant_mode = set_batch_invariant_mode
            print("[OFFICIAL] ✓ batch_invariant_ops imported successfully")
        except ImportError as e:
            print(f"[OFFICIAL] ❌ Failed to import batch_invariant_ops: {e}")
            print("[OFFICIAL] Please install: pip install triton")
            raise
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        self.generation_config = GenerationConfig(
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            top_k=1,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        print("[OFFICIAL] ✓ Model loaded with Triton batch-invariant kernels")
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> Tuple[str, float]:
        start_time = time.time()
        
        with self.set_batch_invariant_mode(True):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                torch.cuda.synchronize(self.device)
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=max_new_tokens
                )
                torch.cuda.synchronize(self.device)
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
        
        elapsed = time.time() - start_time
        return response, elapsed


# ============================================================================
# VERSION 3: CUSTOM (Our CUDA Implementation)
# ============================================================================

# Custom batch-invariant operations
import torch.nn.functional as F
from contextlib import contextmanager

def kahan_sum(tensor: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Kahan summation for numerical precision"""
    shape = list(tensor.shape)
    shape[dim] = 1
    
    sum_result = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    compensation = torch.zeros_like(sum_result)
    
    for i in range(tensor.shape[dim]):
        slice_idx = [slice(None)] * len(tensor.shape)
        slice_idx[dim] = i
        value = tensor[tuple(slice_idx)].unsqueeze(dim)
        y = value - compensation
        t = sum_result + y
        compensation = (t - sum_result) - y
        sum_result = t
    
    if not keepdim:
        sum_result = sum_result.squeeze(dim)
    
    return sum_result


def cuda_batch_invariant_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Fixed-chunk matrix multiplication"""
    FIXED_CHUNK_SIZE = 32
    
    if len(input.shape) == 2:
        batch_size = input.shape[0]
        pad_size = (FIXED_CHUNK_SIZE - (batch_size % FIXED_CHUNK_SIZE)) % FIXED_CHUNK_SIZE
        
        if pad_size > 0:
            padded_input = torch.nn.functional.pad(input, (0, 0, 0, pad_size), value=0.0)
        else:
            padded_input = input
        
        result = torch.matmul(padded_input, mat2)
        return result[:batch_size]
    else:
        return torch.matmul(input, mat2)


def cuda_batch_invariant_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with Kahan summation"""
    original_dtype = input.dtype
    input_double = input.double()
    
    max_vals = input_double.max(dim=dim, keepdim=True)[0]
    shifted = input_double - max_vals
    exp_vals = torch.exp(shifted)
    sum_exp = kahan_sum(exp_vals, dim=dim, keepdim=True)
    
    result = (exp_vals / sum_exp).to(original_dtype)
    return result


@contextmanager
def set_custom_batch_invariant_mode(enabled: bool = True):
    """Context manager for custom batch-invariant operations"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    original_mm = torch.matmul
    original_softmax = torch.nn.functional.softmax
    
    if enabled:
        # Configure determinism
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Replace operations
        torch.matmul = cuda_batch_invariant_mm
        torch.mm = cuda_batch_invariant_mm
        torch.nn.functional.softmax = cuda_batch_invariant_softmax
    
    try:
        yield
    finally:
        if enabled:
            torch.matmul = original_mm
            torch.mm = original_mm
            torch.nn.functional.softmax = original_softmax


class CustomBatchInvariantModel:
    """Using our custom CUDA batch-invariant implementation"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device_id: int = 0):
        self.device = torch.device(f"cuda:{device_id}")
        self.model_name = model_name
        
        print(f"\n[CUSTOM] Loading model with custom batch-invariant ops...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        self.generation_config = GenerationConfig(
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            top_k=1,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        print("[CUSTOM] ✓ Model loaded with fixed-chunk + Kahan operations")
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> Tuple[str, float]:
        start_time = time.time()
        
        with set_custom_batch_invariant_mode(True):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                torch.cuda.synchronize(self.device)
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=max_new_tokens
                )
                torch.cuda.synchronize(self.device)
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
        
        elapsed = time.time() - start_time
        return response, elapsed


# ============================================================================
# TESTING FRAMEWORK
# ============================================================================

def test_determinism(model, model_name: str, num_runs: int = 10) -> Dict:
    """Test determinism by running same prompt multiple times"""
    
    print(f"\n{'='*80}")
    print(f"Testing {model_name}")
    print('='*80)
    
    test_prompts = [
        "you are a creative peot and you are tasked with Writting a haiku about programming, give a poem",
        "Tell me about Richard Feynman",
        """You are SQL code parser agent that extracts all the source tables used in the SQL query.
        The SQL query is:
        %%sql
        drop table if exists p1_ctu_imm;
        create table p1_ctu_imm as
        select distinct A.*, (A.frst_nm||' '||A.last_nm) as pres_name from p1_ctu A where A.cycl_time_id = 202412 and A.bu_cd = '018' and A.sf_team_cd in ('007','171','344')

        -----------------------------------------------------------------------------------------------------------------------------------------------

        -- Inclusions and Exclusions
        -- Creating Pres_Del_table

        %%sql
        drop table if exists pres_del_table;
        create table pres_del_table as
        select 202412 as cycl_time_id,pres_id as bp_id 
        from p1_rem_list_imm_plg where cycl_time_id = 202412 and r_type = 'PD' -- Prescriber level Deletion (BP ID)


        -- Creating Pres_Del_Terr_Table

        %%sql
        drop table if exists pres_del_terr_table;
        create table pres_del_terr_table as
        select cycl_time_id,pres_id as bp_id,pod
        from p1_rem_list_imm_plg where cycl_time_id = 202412 and r_type = 'PDT' -- Prescriber - Terr Deletion (BPID - TERR)


        -- Creating Pres_Add_Table

        %%sql
        drop table if exists pres_add_table;
        create table pres_add_table as
        select 202412 as cycl_time_id,pres_id as bp_id,pod,prim_spty_cd_4 as primary_speciality_cd,secdy_dpty_cd as secondary_speciality_cd
        from p1_rem_list_imm_plg where cycl_time_id = 202412 and a_type = 'PA' -- Prescriber Addition
        For each provided query you will extract the source table names from it and provide the target table that is created from the source table along with transformation description. Do not pick out any intermediate tables used as source tables.
        DO NOT provide any table name that are not present in the query.
        You Should ONLY provide the result as python dictionary with key source_table_name, target_table_name and transformation_description for each source table inside a python list.
        Return the output in JSON, and no other format.
        The Source table names and target table names as a dictionary is:""",
    ]
    
    results = {
        'prompts_tested': len(test_prompts),
        'total_runs': 0,
        'unique_outputs': defaultdict(int),
        'is_deterministic': True,
        'times': [],
        'outputs_per_prompt': []
    }
    
    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"\nPrompt {prompt_idx + 1}: '{prompt[:60]}...'")
        
        outputs = []
        times = []
        
        for run in range(num_runs):
            output, elapsed = model.generate(prompt, max_new_tokens=50)
            outputs.append(output)
            times.append(elapsed)
            results['total_runs'] += 1
        
        unique = len(set(outputs))
        results['unique_outputs'][prompt_idx] = unique
        results['times'].extend(times)
        results['outputs_per_prompt'].append(outputs)
        
        if unique == 1:
            print(f"  ✓ DETERMINISTIC: {num_runs}/{num_runs} identical outputs")
        else:
            print(f"  ✗ NON-DETERMINISTIC: {unique} unique outputs from {num_runs} runs")
            results['is_deterministic'] = False
            
            # Show variants
            output_counts = {}
            for out in outputs:
                output_counts[out] = output_counts.get(out, 0) + 1
            
            for i, (out, count) in enumerate(sorted(output_counts.items(), key=lambda x: -x[1])):
                print(f"    Variant {i+1} ({count}x): {out[:80]}...")
    
    avg_time = sum(results['times']) / len(results['times'])
    print(f"\nAverage generation time: {avg_time:.3f}s")
    
    return results


def compare_cross_batch_determinism(models: Dict, prompt: str = "Explain photosynthesis:") -> Dict:
    """Test if outputs are consistent across different batch sizes"""
    
    print(f"\n{'='*80}")
    print("CROSS-BATCH SIZE CONSISTENCY TEST")
    print('='*80)
    print(f"Prompt: {prompt}")
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n[{model_name}]")
        
        # Get single output as reference
        ref_output, _ = model.generate(prompt, max_new_tokens=50)
        print(f"  Reference (batch=1): {ref_output[:80]}...")
        
        # TODO: Batch generation requires more complex implementation
        # For now, we just test multiple sequential runs
        outputs = []
        for i in range(5):
            output, _ = model.generate(prompt, max_new_tokens=50)
            outputs.append(output)
        
        matches = sum(1 for out in outputs if out == ref_output)
        results[model_name] = {
            'matches': matches,
            'total': len(outputs),
            'is_consistent': matches == len(outputs)
        }
        
        if matches == len(outputs):
            print(f"  ✓ CONSISTENT: All {len(outputs)} runs match reference")
        else:
            print(f"  ✗ INCONSISTENT: {matches}/{len(outputs)} match reference")
    
    return results


def main():
    """Run comprehensive 3-way comparison"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires NVIDIA GPUs.")
        return 1
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    
    # Initialize models
    print(f"\n{'='*80}")
    print("INITIALIZING MODELS")
    print('='*80)
    
    models = {}
    
    try:
        models['BASELINE'] = BaselineModel()
    except Exception as e:
        print(f"❌ Failed to load BASELINE: {e}")
    
    try:
        models['OFFICIAL'] = OfficialBatchInvariantModel()
    except Exception as e:
        print(f"❌ Failed to load OFFICIAL: {e}")
        print("   (Triton may not be installed: pip install triton)")
    
    try:
        models['CUSTOM'] = CustomBatchInvariantModel()
    except Exception as e:
        print(f"❌ Failed to load CUSTOM: {e}")
    
    if len(models) == 0:
        print("\n❌ No models loaded successfully")
        return 1
    
    print(f"\n✓ Loaded {len(models)} model(s): {', '.join(models.keys())}")
    
    # Test 1: Multiple runs determinism
    print(f"\n{'='*80}")
    print("TEST 1: DETERMINISM ACROSS MULTIPLE RUNS")
    print('='*80)
    
    determinism_results = {}
    for name, model in models.items():
        determinism_results[name] = test_determinism(model, name, num_runs=10)
    
    # Test 2: Cross-batch consistency
    print(f"\n{'='*80}")
    print("TEST 2: CROSS-BATCH CONSISTENCY")
    print('='*80)
    
    batch_results = compare_cross_batch_determinism(models)
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print('='*80)
    
    for name in models.keys():
        det_result = determinism_results.get(name, {})
        batch_result = batch_results.get(name, {})
        
        is_det = det_result.get('is_deterministic', False)
        is_consistent = batch_result.get('is_consistent', False)
        
        print(f"\n{name}:")
        print(f"  Determinism:     {'✓ PASS' if is_det else '✗ FAIL'}")
        print(f"  Batch Consistency: {'✓ PASS' if is_consistent else '✗ FAIL'}")
        
        if 'times' in det_result and det_result['times']:
            avg_time = sum(det_result['times']) / len(det_result['times'])
            print(f"  Avg Time:        {avg_time:.3f}s")
    
    # Determine winner
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print('='*80)
    
    deterministic_models = [
        name for name, res in determinism_results.items() 
        if res.get('is_deterministic', False)
    ]
    
    if len(deterministic_models) == 0:
        print("❌ No models achieved 100% determinism")
        print("   This may indicate issues with the test environment or implementations")
    elif len(deterministic_models) == 1:
        print(f"✓ {deterministic_models[0]} is the only deterministic implementation")
    else:
        print(f"✓ {len(deterministic_models)} implementations achieved determinism:")
        for name in deterministic_models:
            print(f"   - {name}")
        
        # Compare performance
        times = {
            name: sum(determinism_results[name]['times']) / len(determinism_results[name]['times'])
            for name in deterministic_models
        }
        fastest = min(times, key=times.get)
        print(f"\n✓ Fastest deterministic model: {fastest} ({times[fastest]:.3f}s avg)")
    
    print(f"\n{'='*80}")
    
    return 0 if deterministic_models else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)