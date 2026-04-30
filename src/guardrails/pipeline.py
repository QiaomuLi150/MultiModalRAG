from __future__ import annotations

from dataclasses import asdict

from .config import GuardrailConfig
from .groundedness_guard import GroundednessGuard
from .input_topic_guard import InputTopicGuard
from .output_policy_guard import OutputPolicyGuard
from .pii_guard import PIIGuard
from .retrieval_adequacy_guard import RetrievalAdequacyGuard


class GuardrailPipeline:
    def __init__(self, config: GuardrailConfig | None = None):
        self.config = config or GuardrailConfig()
        self.input_topic_guard = InputTopicGuard(self.config)
        self.pii_guard = PIIGuard(self.config)
        self.retrieval_adequacy_guard = RetrievalAdequacyGuard(self.config)
        self.groundedness_guard = GroundednessGuard(self.config)
        self.output_policy_guard = OutputPolicyGuard(self.config)

    def run_pre_retrieval(self, query: str, history: list[dict] | None = None) -> dict:
        topic_result = self.input_topic_guard.run(query, history)
        pii_result = self.pii_guard.redact(query)
        query_to_use = pii_result.metadata.get("redacted_text", query)
        if topic_result.next_action != "continue":
            return _decision(topic_result, response=_safe_response(topic_result.message_for_user), query=query_to_use)
        return _decision(
            topic_result,
            response=None,
            query=query_to_use,
            extra={"results": {"input_topic_guard": asdict(topic_result), "pii_guard_input": asdict(pii_result)}},
        )

    def run_post_retrieval(self, retrieval_results: list[dict]) -> dict:
        result = self.retrieval_adequacy_guard.run(retrieval_results)
        if result.next_action != "continue":
            return _decision(result, response=_safe_response(result.message_for_user))
        return _decision(result, response=None, extra={"results": {"retrieval_adequacy_guard": asdict(result)}})

    def run_post_generation(self, answer_payload: dict, evidence: list[str]) -> dict:
        groundedness = self.groundedness_guard.run(str(answer_payload.get("answer", "")), evidence)
        if groundedness.next_action != "continue":
            return _decision(groundedness, response=_safe_response(groundedness.message_for_user))

        pii_result = self.pii_guard.redact(str(answer_payload.get("answer", "")))
        if pii_result.status == "redacted":
            answer_payload = {**answer_payload, "answer": pii_result.metadata.get("redacted_text", answer_payload.get("answer", ""))}
        elif pii_result.next_action != "continue":
            return _decision(pii_result, response=_safe_response(pii_result.message_for_user))

        output_result = self.output_policy_guard.run(answer_payload)
        if output_result.next_action != "continue":
            return _decision(output_result, response=_safe_response(output_result.message_for_user))

        summary = {
            "groundedness_guard": groundedness.status,
            "pii_guard_output": pii_result.status,
            "output_policy_guard": output_result.status,
        }
        return {
            "action": "continue",
            "response": answer_payload,
            "results": {
                "groundedness_guard": asdict(groundedness),
                "pii_guard_output": asdict(pii_result),
                "output_policy_guard": asdict(output_result),
            },
            "summary": summary,
        }


def _decision(result, response, query: str | None = None, extra: dict | None = None) -> dict:
    payload = {
        "action": result.next_action,
        "response": response,
        "results": {result.guard_name: asdict(result)},
    }
    if query is not None:
        payload["query"] = query
    if extra:
        payload.update(extra)
    return payload


def _safe_response(message: str | None) -> dict:
    return {
        "answer": message or "I could not answer that safely from the indexed materials.",
        "sources": [],
        "confidence": "low",
        "guardrail_summary": {},
    }

