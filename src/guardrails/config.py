from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GuardrailConfig:
    allow_out_of_domain: bool = False
    redact_pii: bool = True
    block_sensitive_topics: bool = True
    enable_groundedness_check: bool = True
    enable_output_rewrite: bool = False
    max_regeneration_attempts: int = 0
    min_top1_score: float = 0.05
    min_topk_mean_score: float = 0.02
    min_evidence_chars: int = 200
    min_distinct_sources: int = 1
    max_answer_sentences: int = 8
    weak_retrieval_action: str = "return_safe_refusal"
    partial_groundedness_action: str = "return_safe_refusal"
    out_of_domain_action: str = "ask_clarifying_question"
    supported_domains: list[str] = field(
        default_factory=lambda: ["uploaded documents", "indexed materials", "enterprise files", "course materials"]
    )
    banned_topics: list[str] = field(default_factory=list)
    banned_entities: list[str] = field(default_factory=list)


def config_for_level(level: str) -> GuardrailConfig:
    normalized = (level or "balanced").strip().lower()
    if normalized == "off":
        return GuardrailConfig(
            allow_out_of_domain=True,
            redact_pii=False,
            block_sensitive_topics=False,
            enable_groundedness_check=False,
            min_top1_score=-1.0,
            min_topk_mean_score=-1.0,
            min_evidence_chars=0,
            max_answer_sentences=100,
            weak_retrieval_action="continue",
            partial_groundedness_action="continue",
            out_of_domain_action="continue",
        )
    if normalized == "relaxed":
        return GuardrailConfig(
            allow_out_of_domain=True,
            redact_pii=True,
            block_sensitive_topics=False,
            enable_groundedness_check=True,
            min_top1_score=0.01,
            min_topk_mean_score=0.005,
            min_evidence_chars=80,
            max_answer_sentences=12,
            weak_retrieval_action="continue",
            partial_groundedness_action="continue",
            out_of_domain_action="continue",
        )
    if normalized == "strict":
        return GuardrailConfig(
            allow_out_of_domain=False,
            redact_pii=True,
            block_sensitive_topics=True,
            enable_groundedness_check=True,
            min_top1_score=0.12,
            min_topk_mean_score=0.05,
            min_evidence_chars=350,
            max_answer_sentences=6,
            weak_retrieval_action="return_safe_refusal",
            partial_groundedness_action="return_safe_refusal",
            out_of_domain_action="return_safe_refusal",
        )
    return GuardrailConfig()
