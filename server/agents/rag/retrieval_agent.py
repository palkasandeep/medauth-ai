from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from server.agents.rag.indexing_agent import IndexingAgent
from server.agents.rag.vector_store import case_vector_store, cosine_similarity
from server.schemas import Decision, DecisionRouting, Report
from server.utils.config import get_models_config
from server.utils.gemini_client import embed_text


class RetrievalAgent:
    def __init__(
        self, custom_embedder: Optional[Callable[[str], List[float]]] = None
    ) -> None:
        self._embedder = custom_embedder or embed_text

    def retrieve(
        self, patient_id: str, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        candidates = case_vector_store.get_chunks_for_patient(patient_id)
        if not candidates:
            return []

        k = top_k if top_k is not None else int(get_models_config()["rag"]["top_k"])
        q_vec = self._embedder(query or "case summary")

        scored = [
            {
                "chunk_id": c.chunk_id,
                "patient_id": c.patient_id,
                "report_id": c.report_id,
                "text": c.text,
                "score": round(cosine_similarity(q_vec, c.embedding), 4),
            }
            for c in candidates
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]


def self_test() -> None:
    case_vector_store.clear()

    def mock_embed(txt: str) -> List[float]:
        return [1.0 if "MRI" in txt else 0.0, 1.0 if "Knee" in txt else 0.0, 0.1]

    indexer = IndexingAgent(mock_embed)
    base_dec = Decision(
        decision="APPROVED",
        confidence_score=0.9,
        key_findings=["OK"],
        rule_results=[],
        reasoning="Approved.",
        routing=DecisionRouting(escalate_to_human=False, reason="Clean"),
    )
    r1 = Report(
        report_id="RPT-A",
        case_id="CA",
        patient_id="PAT-A",
        patient_name="Alice",
        requested_service="MRI",
        generated_at=datetime.now(timezone.utc).isoformat(),
        rules_applied=[],
        decision=base_dec,
        human_review_required=False,
        status="auto_finalized",
    )
    r2 = r1.model_copy(
        update={
            "report_id": "RPT-B",
            "patient_id": "PAT-B",
            "patient_name": "Bob",
            "requested_service": "Knee Surgery",
        }
    )
    indexer.index_report(r1)
    indexer.index_report(r2)

    retriever = RetrievalAgent(mock_embed)
    res_a = retriever.retrieve("PAT-A", "Why was MRI approved?")
    assert len(res_a) > 0
    assert all(c["patient_id"] == "PAT-A" for c in res_a)
    assert retriever.retrieve("PAT-UNKNOWN", "Any info?") == []
    print("[PASS] RetrievalAgent self_test")


if __name__ == "__main__":
    self_test()
