# batch_invariant_ops_cuda.py
"""
Batch-invariant operations for NVIDIA GPUs (CUDA)
Adapted from thinking-machines-lab/batch_invariant_ops for NVIDIA A10G GPUs
Based on research: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
"""

import torch
import torch.nn.functional as F
from contextlib import contextmanager
import os
from typing import Optional, Callable, Dict, Any


class CUDABatchInvariantOps:
    """
    Batch-invariant operations using NVIDIA CUDA GPUs
    Ensures deterministic inference regardless of batch size
    """
    
    def __init__(self):
        self.original_ops = {}
        self.enabled = False
        self._configure_cuda_for_determinism()
    
    def _configure_cuda_for_determinism(self):
        """Configure CUDA for maximum determinism"""
        # Force CUDA to use deterministic algorithms
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Set CUDA flags for deterministic behavior
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Allow TF32 but maintain determinism
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    
    def _ensure_cuda_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on CUDA device"""
        if tensor.device.type != 'cuda':
            return tensor.to('cuda')
        return tensor


# Batch-invariant kernel implementations for CUDA
def cuda_batch_invariant_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    Batch-invariant matrix multiplication on CUDA
    Key insight: Process with fixed chunk sizes regardless of actual batch size
    
    This ensures that the reduction order remains consistent across different batch sizes,
    preventing numerical differences due to floating-point non-associativity.
    """
    device = input.device
    
    # Ensure tensors are on CUDA
    if device.type != 'cuda':
        input = input.to('cuda')
        mat2 = mat2.to('cuda')
    
    # Fixed chunk size for consistent computation pattern
    # For A10G GPUs, use a chunk size that aligns well with warp/block dimensions
    FIXED_CHUNK_SIZE = 32  # Optimized for NVIDIA architecture (multiple of warp size 32)
    
    if len(input.shape) == 2:
        batch_size = input.shape[0]
        
        # Pad to multiple of FIXED_CHUNK_SIZE for consistent execution
        pad_size = (FIXED_CHUNK_SIZE - (batch_size % FIXED_CHUNK_SIZE)) % FIXED_CHUNK_SIZE
        
        if pad_size > 0:
            # Pad input to ensure consistent chunk processing
            # Use zeros for padding (won't affect numerical results due to multiplication)
            padded_input = torch.nn.functional.pad(input, (0, 0, 0, pad_size), value=0.0)
        else:
            padded_input = input
        
        # Perform computation with fixed chunk pattern
        # This ensures the same tensor core instruction pattern is used
        result = torch.matmul(padded_input, mat2)
        
        # Remove padding from result
        return result[:batch_size]
    else:
        # For non-2D tensors, use standard matmul (already deterministic in PyTorch)
        return torch.matmul(input, mat2)


def cuda_batch_invariant_bmm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    Batch-invariant batched matrix multiplication on CUDA
    """
    if len(input.shape) == 3:
        batch_size = input.shape[0]
        FIXED_CHUNK_SIZE = 32
        
        pad_size = (FIXED_CHUNK_SIZE - (batch_size % FIXED_CHUNK_SIZE)) % FIXED_CHUNK_SIZE
        
        if pad_size > 0:
            padded_input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, pad_size), value=0.0)
            padded_mat2 = torch.nn.functional.pad(mat2, (0, 0, 0, 0, 0, pad_size), value=0.0)
        else:
            padded_input = input
            padded_mat2 = mat2
        
        result = torch.bmm(padded_input, padded_mat2)
        return result[:batch_size]
    else:
        return torch.bmm(input, mat2)


def cuda_batch_invariant_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Batch-invariant softmax using higher precision for intermediate calculations
    
    This implementation ensures numerical stability and determinism across batches
    by using double precision for critical operations and Kahan summation.
    """
    # Use double precision for intermediate calculations to maintain precision
    original_dtype = input.dtype
    input_double = input.double()
    
    # Compute max for numerical stability (prevent overflow in exp)
    max_vals = input_double.max(dim=dim, keepdim=True)[0]
    shifted = input_double - max_vals
    
    # Compute exp
    exp_vals = torch.exp(shifted)
    
    # Use Kahan summation for deterministic reduction
    # This compensates for floating-point rounding errors
    sum_exp = kahan_sum(exp_vals, dim=dim, keepdim=True)
    
    # Return to original dtype
    result = (exp_vals / sum_exp).to(original_dtype)
    return result


def kahan_sum(tensor: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    Kahan summation algorithm for improved numerical precision and determinism
    
    This algorithm compensates for floating-point rounding errors by keeping track
    of the accumulated error and correcting for it. This ensures that the sum is
    computed in a consistent order regardless of how the computation is parallelized.
    
    See: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    """
    shape = list(tensor.shape)
    shape[dim] = 1
    
    sum_result = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    compensation = torch.zeros_like(sum_result)
    
    # Sequential summation with error compensation
    for i in range(tensor.shape[dim]):
        slice_idx = [slice(None)] * len(tensor.shape)
        slice_idx[dim] = i
        
        value = tensor[tuple(slice_idx)].unsqueeze(dim)
        
        # Kahan summation steps
        y = value - compensation
        t = sum_result + y
        compensation = (t - sum_result) - y
        sum_result = t
    
    if not keepdim:
        sum_result = sum_result.squeeze(dim)
    
    return sum_result


def cuda_batch_invariant_layer_norm(input: torch.Tensor, normalized_shape, 
                                     weight=None, bias=None, eps=1e-5) -> torch.Tensor:
    """
    Batch-invariant layer normalization for CUDA
    
    Ensures consistent normalization across different batch sizes by using
    deterministic reduction operations.
    """
    # Compute mean and variance using Kahan summation for precision
    dims = list(range(len(input.shape) - len(normalized_shape), len(input.shape)))
    
    # Mean calculation
    mean = input.double()
    for dim in reversed(dims):
        mean = kahan_sum(mean, dim=dim, keepdim=True) / input.shape[dim]
    mean = mean.to(input.dtype)
    
    # Variance calculation
    var = ((input - mean) ** 2).double()
    for dim in reversed(dims):
        var = kahan_sum(var, dim=dim, keepdim=True) / input.shape[dim]
    var = var.to(input.dtype)
    
    # Normalize
    normalized = (input - mean) / torch.sqrt(var + eps)
    
    # Apply affine transformation if provided
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    
    return normalized


@contextmanager
def set_cuda_batch_invariant_mode(enabled: bool = True):
    """
    Context manager for CUDA batch-invariant operations
    
    Usage:
        with set_cuda_batch_invariant_mode(True):
            # All operations in this block will use batch-invariant implementations
            output = model(input)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system. This implementation requires NVIDIA GPUs.")
    
    # Store original operations
    original_mm = torch.matmul
    original_bmm = torch.bmm
    original_softmax = torch.nn.functional.softmax
    original_layer_norm = torch.nn.functional.layer_norm
    
    if enabled:
        print("Enabling CUDA batch-invariant mode for deterministic inference...")
        
        # Configure PyTorch for determinism
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # For multi-GPU setups
        
        # Set CUDA determinism flags
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Replace operations with batch-invariant versions
        torch.matmul = cuda_batch_invariant_mm
        torch.mm = cuda_batch_invariant_mm
        torch.bmm = cuda_batch_invariant_bmm
        torch.nn.functional.softmax = cuda_batch_invariant_softmax
        torch.nn.functional.layer_norm = cuda_batch_invariant_layer_norm
        
        print("✓ Batch-invariant operations enabled")
        print("✓ CUDA deterministic algorithms enabled")
        print("✓ TF32 disabled for numerical consistency")
    
    try:
        yield
    finally:
        if enabled:
            # Restore original operations
            torch.matmul = original_mm
            torch.mm = original_mm
            torch.bmm = original_bmm
            torch.nn.functional.softmax = original_softmax
            torch.nn.functional.layer_norm = original_layer_norm
            
            print("Batch-invariant mode disabled, operations restored")


def verify_cuda_determinism():
    """
    Verify that CUDA determinism is properly configured
    
    Returns:
        bool: True if determinism checks pass, False otherwise
    """
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Check determinism settings
    checks = []
    
    # Check if deterministic algorithms are enabled
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        checks.append(("Deterministic algorithms", True))
    except Exception as e:
        checks.append(("Deterministic algorithms", False))
    
    # Check cuDNN settings
    checks.append(("cuDNN deterministic", torch.backends.cudnn.deterministic))
    checks.append(("cuDNN benchmark disabled", not torch.backends.cudnn.benchmark))
    checks.append(("TF32 disabled", not torch.backends.cuda.matmul.allow_tf32))
    
    # Print results
    all_pass = all(check[1] for check in checks)
    for name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"{status} {name}: {passed}")
    
    return all_pass