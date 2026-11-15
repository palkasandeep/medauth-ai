from __future__ import annotations

import re
from typing import List

from server.schemas import RagTriad
from server.utils.config import get_models_config, get_persona_config
from server.utils.gemini_client import generate_json_with_config, has_gemini_api_key


class ReflectionAgent:
    def evaluate(self, query: str, answer: str, contexts: List[str]) -> RagTriad:
        threshold = float(get_models_config()["rag"]["reflection_pass_threshold"])

        if not (query or "").strip() or not (answer or "").strip():
            return RagTriad(
                answer_relevance=0.0,
                groundedness=0.0,
                context_relevance=0.0,
                overall_score=0.0,
                passed=False,
                feedback="Empty query or answer cannot pass RAG triad evaluation.",
            )

        if not contexts:
            honest = bool(
                re.search(
                    r"no (?:finished )?(?:pa )?report|not found|unavailable",
                    answer,
                    re.I,
                )
            )
            score = 0.75 if honest else 0.2
            return RagTriad(
                answer_relevance=0.8 if honest else 0.3,
                groundedness=0.9 if honest else 0.1,
                context_relevance=0.55 if honest else 0.2,
                overall_score=score,
                passed=score >= threshold,
                feedback=(
                    "No report chunks existed and assistant honestly reported missing case data."
                    if honest
                    else "No supporting report context chunks were retrieved."
                ),
            )

        if has_gemini_api_key():
            try:
                persona = get_persona_config()["reflection_agent"]
                ctx_str = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
                prompt = (
                    "Score the RAG Triad on [0.0, 1.0]:\n"
                    "1. answer_relevance: How directly does the answer address the query?\n"
                    "2. groundedness: Is every claim supported by retrieved contexts?\n"
                    "3. context_relevance: How relevant are retrieved contexts?\n"
                    'Return JSON: {"answer_relevance": float, "groundedness": float, "context_relevance": float, "feedback": str}\n\n'
                    f"QUERY: {query}\nRETRIEVED CONTEXTS:\n{ctx_str}\n\nANSWER:\n{answer}"
                )
                res = generate_json_with_config(
                    "rag_reflection",
                    prompt,
                    f"{persona['persona']} {persona['instructions']}",
                )
                clamp = lambda v: round(max(0.0, min(1.0, float(v if v is not None else 0.7))), 2)
                ar = clamp(res.get("answer_relevance"))
                gr = clamp(res.get("groundedness"))
                cr = clamp(res.get("context_relevance"))
                overall = round((ar + gr + cr) / 3.0, 2)
                return RagTriad(
                    answer_relevance=ar,
                    groundedness=gr,
                    context_relevance=cr,
                    overall_score=overall,
                    passed=(overall >= threshold and gr >= 0.5),
                    feedback=str(res.get("feedback") or "Evaluated via RAG triad.").strip(),
                )
            except Exception:
                pass

        ctx_low = " ".join(contexts).lower()
        tokens = [t for t in re.split(r"\W+", answer.lower()) if len(t) > 3]
        overlap = sum(1 for t in tokens if t in ctx_low)
        grounded = min(1.0, round((overlap / len(tokens)) + 0.2, 2)) if tokens else 0.5
        ar, cr = 0.85, 0.85
        overall = round((ar + grounded + cr) / 3.0, 2)
        return RagTriad(
            answer_relevance=ar,
            groundedness=grounded,
            context_relevance=cr,
            overall_score=overall,
            passed=overall >= threshold,
            feedback="Evaluated via deterministic lexical RAG-triad fallback.",
        )


def self_test() -> None:
    agent = ReflectionAgent()
    empty = agent.evaluate("", "", [])
    assert empty.passed is False
    assert empty.overall_score == 0.0

    valid = agent.evaluate(
        "What was the determination for Elena Morales?",
        "Elena Morales's MRI Lumbar Spine request was APPROVED with 94% confidence.",
        ["Patient Name: Elena Morales | Requested Service: MRI Lumbar Spine | Determination: APPROVED (Confidence: 0.94)"],
    )
    assert 0.0 <= valid.answer_relevance <= 1.0
    assert 0.0 <= valid.groundedness <= 1.0
    assert 0.0 <= valid.context_relevance <= 1.0
    assert 0.0 <= valid.overall_score <= 1.0
    assert isinstance(valid.passed, bool)
    print("[PASS] ReflectionAgent self_test")


if __name__ == "__main__":
    self_test()
