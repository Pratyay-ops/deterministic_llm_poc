"""In-process Hugging Face transformers engine.

Used for model-level batch-invariance tests where the batch composition must be controlled
exactly - something a serving engine with continuous batching cannot offer. Batches are built from
sequences of identical length, so there is no padding and the target row sees exactly the same
tokens and positions at every batch size. Greedy decoding is a plain loop over the KV cache (no
``generate()`` logits processors), returning token ids and the logprob of every chosen token.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from detllm.config import ModelPreset
from detllm.kernels.activate import KernelMode, activate, public_info
from detllm.modes import HF_ATTN_IMPL, HF_KERNEL_MODE

log = logging.getLogger(__name__)


class HFEngine:
    def __init__(
        self,
        preset: ModelPreset,
        mode: str,
        *,
        attn_implementation: str | None = None,
        device: str = "cuda",
    ):
        from transformers import AutoModelForCausalLM

        from detllm.prompts import load_tokenizer

        self.preset = preset
        self.mode = mode
        self.device = torch.device(device)
        self.dtype = getattr(torch, preset.dtype)
        self.attn_implementation = attn_implementation or HF_ATTN_IMPL[mode]
        self.tokenizer = load_tokenizer(preset.hf_id)

        log.info("loading %s (%s, attn=%s)", preset.hf_id, preset.dtype, self.attn_implementation)
        kwargs: dict[str, Any] = {"attn_implementation": self.attn_implementation}
        try:
            model = AutoModelForCausalLM.from_pretrained(preset.hf_id, dtype=self.dtype, **kwargs)
        except TypeError:  # transformers < 4.56
            model = AutoModelForCausalLM.from_pretrained(preset.hf_id, torch_dtype=self.dtype, **kwargs)
        if next(model.parameters()).dtype != self.dtype:
            model = model.to(self.dtype)
        self.model = model.to(self.device).eval()

        # Kernel overrides are switched on after loading so weight loading is unaffected.
        self.kernel_mode: KernelMode = activate(HF_KERNEL_MODE[mode])

    def info(self) -> dict[str, Any]:
        return {
            "hf_id": self.preset.hf_id,
            "dtype": self.preset.dtype,
            "attn_implementation": self.attn_implementation,
            "kernels": public_info(self.kernel_mode),
        }

    def _ids(self, batch: list[list[int]]) -> torch.Tensor:
        lengths = {len(row) for row in batch}
        if len(lengths) != 1:
            raise ValueError(f"all sequences in a batch must have the same length, got {sorted(lengths)}")
        return torch.tensor(batch, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def forward_logits(self, batch: list[list[int]], rows: list[int]) -> torch.Tensor:
        """Logits (fp32) of the selected rows at every position: shape (len(rows), L, vocab)."""
        ids = self._ids(batch)
        with self.kernel_mode.context():
            out = self.model(input_ids=ids, use_cache=False)
            return out.logits[rows].float()

    @torch.inference_mode()
    def greedy(self, batch: list[list[int]], max_new_tokens: int) -> tuple[list[list[int]], list[list[float]]]:
        """Greedy decode a fixed number of tokens for every row (no early stop on EOS, so all rows
        keep the same length and the batch never needs padding)."""
        ids = self._ids(batch)
        tokens: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        with self.kernel_mode.context():
            out = self.model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            for step in range(max_new_tokens):
                logp = torch.log_softmax(logits, dim=-1)
                nxt = torch.argmax(logits, dim=-1)
                tokens.append(nxt)
                logprobs.append(logp.gather(1, nxt[:, None]).squeeze(1))
                if step + 1 == max_new_tokens:
                    break
                out = self.model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()
        tok = torch.stack(tokens, dim=1).cpu().tolist()
        lps = torch.stack(logprobs, dim=1).cpu().tolist()
        return tok, lps

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
