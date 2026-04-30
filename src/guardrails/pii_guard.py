from __future__ import annotations

import re

from .config import GuardrailConfig
from .schemas import GuardrailResult


PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


class PIIGuard:
    def __init__(self, config: GuardrailConfig):
        self.config = config

    def redact(self, text: str) -> GuardrailResult:
        redacted = text or ""
        found: list[str] = []
        for pii_type, pattern in PII_PATTERNS.items():
            if pattern.search(redacted):
                found.append(pii_type)
                redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)

        if not found:
            return GuardrailResult(
                guard_name="pii_guard",
                status="pass",
                reason_code="no_pii",
                metadata={"redacted_text": text},
            )

        if self.config.redact_pii:
            return GuardrailResult(
                guard_name="pii_guard",
                status="redacted",
                reason_code="pii_redacted",
                message_for_user="Sensitive personal information was redacted from the response.",
                next_action="redact_and_continue",
                metadata={"pii_types_detected": found, "redacted_text": redacted},
            )

        return GuardrailResult(
            guard_name="pii_guard",
            status="fail",
            reason_code="pii_blocked",
            message_for_user="I cannot return that information because it may include sensitive personal data.",
            next_action="return_safe_refusal",
            metadata={"pii_types_detected": found},
        )

