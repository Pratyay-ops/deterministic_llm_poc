"""detllm - reproduce "Defeating Nondeterminism in LLM Inference" on NVIDIA GB10.

Thinking Machines Lab, Horace He et al., Sep 2025:
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Importing this package must stay cheap: the scheduler, config loader and report generator never
touch torch/vLLM. GPU code is imported lazily inside the job processes.
"""

__version__ = "0.2.0"
