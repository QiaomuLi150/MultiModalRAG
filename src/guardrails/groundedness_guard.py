from __future__ import annotations

import re

from .config import GuardrailConfig
from .schemas import GuardrailResult


class GroundednessGuard:
    def __init__(self, config: GuardrailConfig):
        self.config = config

    def run(self, answer: str, evidence: list[str]) -> GuardrailResult:
        if not self.config.enable_groundedness_check:
            return GuardrailResult("groundedness_guard", "pass", "disabled")

        evidence_blob = " ".join(evidence).lower()
        unsupported_claims: list[str] = []
        for sentence in _split_sentences(answer):
            words = [word for word in re.findall(r"[a-z0-9]{4,}", sentence.lower())]
            if not words:
                continue
            overlap = sum(1 for word in words if word in evidence_blob)
            ratio = overlap / max(1, len(words))
            if ratio < 0.2 and "[" not in sentence:
                unsupported_claims.append(sentence.strip())

        if unsupported_claims:
            return GuardrailResult(
                guard_name="groundedness_guard",
                status="review",
                reason_code="partially_supported",
                confidence=0.7,
                message_for_user="I found related material, but I cannot verify all of the answer strongly enough from the retrieved evidence.",
                next_action=self.config.partial_groundedness_action,
                metadata={"unsupported_claims": unsupported_claims[:3]},
            )

        return GuardrailResult(
            guard_name="groundedness_guard",
            status="pass",
            reason_code="entailed",
            confidence=0.7,
        )


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]
