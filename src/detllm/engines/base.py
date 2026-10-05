"""Engine-independent result record."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Completion:
    index: int
    prompt_id: str
    token_ids: list[int] | None
    text: str | None
    logprobs: list[float] | None = None  # logprob of each generated (sampled) token
    finish_reason: str | None = None
    num_prompt_tokens: int | None = None
    latency_s: float | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Completion:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
