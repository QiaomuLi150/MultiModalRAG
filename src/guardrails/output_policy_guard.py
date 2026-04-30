from __future__ import annotations

import re

from .config import GuardrailConfig
from .schemas import GuardrailResult


class OutputPolicyGuard:
    def __init__(self, config: GuardrailConfig):
        self.config = config

    def run(self, answer_payload: dict) -> GuardrailResult:
        answer = str(answer_payload.get("answer", "")).strip()
        sources = answer_payload.get("sources") or []
        confidence = str(answer_payload.get("confidence", "")).strip().lower()

        if not answer:
            return GuardrailResult(
                guard_name="output_policy_guard",
                status="fail",
                reason_code="empty_answer",
                message_for_user="I could not produce a grounded answer for this question.",
                next_action="return_safe_refusal",
            )
        if not sources:
            return GuardrailResult(
                guard_name="output_policy_guard",
                status="fail",
                reason_code="missing_citations",
                message_for_user="I found related material, but I cannot return an answer without source citations.",
                next_action="return_safe_refusal",
            )
        if "[" not in answer or "]" not in answer:
            return GuardrailResult(
                guard_name="output_policy_guard",
                status="review",
                reason_code="inline_citations_missing",
                message_for_user="The answer was not grounded with inline citations strongly enough.",
                next_action="return_safe_refusal",
            )
        sentence_count = len([part for part in re.split(r"(?<=[.!?])\s+", answer) if part.strip()])
        if sentence_count > self.config.max_answer_sentences:
            return GuardrailResult(
                guard_name="output_policy_guard",
                status="review",
                reason_code="answer_too_long",
                message_for_user="The answer is too long for the amount of retrieved evidence.",
                next_action="return_safe_refusal",
            )
        if confidence not in {"high", "medium", "low"}:
            return GuardrailResult(
                guard_name="output_policy_guard",
                status="fail",
                reason_code="confidence_missing",
                message_for_user="The answer payload is missing a confidence label.",
                next_action="return_safe_refusal",
            )
        return GuardrailResult(
            guard_name="output_policy_guard",
            status="pass",
            reason_code="compliant",
            confidence=0.9,
        )

