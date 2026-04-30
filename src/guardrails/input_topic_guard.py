from __future__ import annotations

import re

from .config import GuardrailConfig
from .schemas import GuardrailResult


PROMPT_INJECTION_PATTERNS = (
    r"ignore (all|previous|prior) instructions",
    r"reveal (the )?(system|hidden) prompt",
    r"do not use (the )?(retrieved|indexed) (docs|documents|materials)",
    r"override (your|the) instructions",
)

OUT_OF_DOMAIN_HINTS = (
    "weather",
    "sports score",
    "lottery",
    "celebrity gossip",
    "political prediction",
)


class InputTopicGuard:
    def __init__(self, config: GuardrailConfig):
        self.config = config

    def run(self, query: str, history: list[dict] | None = None) -> GuardrailResult:
        cleaned = (query or "").strip()
        if not cleaned:
            return GuardrailResult(
                guard_name="input_topic_guard",
                status="fail",
                reason_code="empty_query",
                confidence=1.0,
                message_for_user="Please enter a question before asking the system to search.",
                next_action="return_safe_refusal",
            )
        if len(cleaned) < 3:
            return GuardrailResult(
                guard_name="input_topic_guard",
                status="review",
                reason_code="too_short",
                confidence=0.9,
                message_for_user="Please make the question a bit more specific so the system can retrieve useful evidence.",
                next_action="ask_clarifying_question",
            )

        lowered = cleaned.lower()
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lowered):
                return GuardrailResult(
                    guard_name="input_topic_guard",
                    status="fail",
                    reason_code="prompt_injection_suspected",
                    confidence=0.95,
                    message_for_user="I can answer questions about the indexed materials, but I cannot follow instruction-override requests.",
                    next_action="return_safe_refusal",
                )

        if not self.config.allow_out_of_domain and any(hint in lowered for hint in OUT_OF_DOMAIN_HINTS):
            return GuardrailResult(
                guard_name="input_topic_guard",
                status="review",
                reason_code="out_of_domain",
                confidence=0.75,
                message_for_user="This system is intended for questions about the indexed materials. Please ask about uploaded documents or indexed sources.",
                next_action=self.config.out_of_domain_action,
            )

        return GuardrailResult(
            guard_name="input_topic_guard",
            status="pass",
            reason_code="in_domain",
            confidence=0.8,
        )
