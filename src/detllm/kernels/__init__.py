"""Batch-invariant kernels and the in-process switches for each op-level mode.

official.py   thin wrapper around thinking-machines-lab/batch_invariant_ops (git submodule)
extra_ops.py  batch-invariant Triton bmm + softmax (attention ops the library does not cover)
custom.py     FixedChunkMode - invariance by running vendor kernels on fixed-size chunks
activate.py   turn a mode on inside the current process
"""
