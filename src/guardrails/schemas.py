from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GuardrailResult:
    guard_name: str
    status: str
    reason_code: str
    confidence: float = 1.0
    message_for_user: str | None = None
    next_action: str = "continue"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalSignals:
    top1_score: float
    topk_mean_score: float
    distinct_sources: int
    evidence_chars: int
    result_count: int

