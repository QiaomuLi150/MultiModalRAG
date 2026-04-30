from __future__ import annotations

from .config import GuardrailConfig
from .schemas import GuardrailResult, RetrievalSignals


class RetrievalAdequacyGuard:
    def __init__(self, config: GuardrailConfig):
        self.config = config

    def run(self, retrieval_results: list[dict]) -> GuardrailResult:
        signals = self._signals(retrieval_results)
        if signals.result_count == 0:
            return GuardrailResult(
                guard_name="retrieval_adequacy_guard",
                status="fail",
                reason_code="no_relevant_results",
                confidence=1.0,
                message_for_user="I could not find enough support in the indexed materials to answer reliably.",
                next_action=self.config.weak_retrieval_action,
                metadata={"signals": signals.__dict__},
            )
        if signals.evidence_chars < self.config.min_evidence_chars:
            return GuardrailResult(
                guard_name="retrieval_adequacy_guard",
                status="fail",
                reason_code="low_evidence_chars",
                confidence=0.85,
                message_for_user="I found related material, but there is not enough retrieved evidence to answer reliably.",
                next_action=self.config.weak_retrieval_action,
                metadata={"signals": signals.__dict__},
            )
        if signals.top1_score < self.config.min_top1_score and signals.topk_mean_score < self.config.min_topk_mean_score:
            return GuardrailResult(
                guard_name="retrieval_adequacy_guard",
                status="review",
                reason_code="weak_top_score",
                confidence=0.8,
                message_for_user="The retrieved evidence looks weak. Please try a more specific query or inspect the retrieved evidence first.",
                next_action=self.config.weak_retrieval_action,
                metadata={"signals": signals.__dict__},
            )
        return GuardrailResult(
            guard_name="retrieval_adequacy_guard",
            status="pass",
            reason_code="strong_evidence",
            confidence=0.8,
            metadata={"signals": signals.__dict__},
        )

    def _signals(self, retrieval_results: list[dict]) -> RetrievalSignals:
        if not retrieval_results:
            return RetrievalSignals(0.0, 0.0, 0, 0, 0)
        scores = [float(item.get("score", 0.0)) for item in retrieval_results]
        evidence_chars = sum(len(str(item.get("text", ""))) for item in retrieval_results)
        distinct_sources = len({str(item.get("source_name", "")) for item in retrieval_results})
        return RetrievalSignals(
            top1_score=max(scores),
            topk_mean_score=sum(scores) / max(1, len(scores)),
            distinct_sources=distinct_sources,
            evidence_chars=evidence_chars,
            result_count=len(retrieval_results),
        )
