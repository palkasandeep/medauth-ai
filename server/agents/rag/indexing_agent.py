from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from server.agents.rag.vector_store import IndexedChunk, case_vector_store
from server.schemas import Decision, DecisionRouting, Report, RuleResult
from server.utils.config import get_models_config
from server.utils.gemini_client import embed_text, has_gemini_api_key


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    norm = text.strip()
    if not norm:
        return []
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    i = 0
    while i < len(norm):
        chunks.append(norm[i : i + chunk_size])
        if i + chunk_size >= len(norm):
            break
        i += step
    return chunks


def format_report_for_indexing(report: Report) -> str:
    elig_line = (
        f"Eligibility: eligible={report.eligibility.eligible}, "
        f"coverage={report.eligibility.coverage_percentage}%, "
        f"patient_responsibility=${report.eligibility.patient_responsibility}, "
        f"reason={report.eligibility.eligibility_reason}"
        if report.eligibility
        else "Eligibility: Not evaluated"
    )
    risk_line = (
        f"Risk Assessment: level={report.risk.risk_level}, score={report.risk.risk_score}, "
        f"crosswalk_match={report.risk.crosswalk_match}, duplicate={report.risk.is_duplicate}, "
        f"provider_verified={report.risk.provider_verified}, "
        f"fraud_indicators={'; '.join(report.risk.fraud_indicators) or 'None'}, "
        f"reasoning={report.risk.reasoning}"
        if report.risk
        else "Risk Assessment: Not evaluated"
    )
    rules_line = " | ".join(
        f"{r.rule_id} [{'PASS' if r.passed else 'FAIL'}]: {r.reason}"
        for r in report.rules_applied
    )
    return "\n".join(
        [
            f"Report ID: {report.report_id}",
            f"Patient ID: {report.patient_id} | Patient Name: {report.patient_name}",
            f"Requested Service: {report.requested_service}",
            f"Determination: {report.decision.decision} (Confidence: {report.decision.confidence_score})",
            f"Human Review Required: {report.human_review_required} ({report.decision.routing.reason})",
            f"Status: {report.status}",
            f"Decision Reasoning: {report.decision.reasoning}",
            f"Key Findings: {'; '.join(report.decision.key_findings)}",
            elig_line,
            risk_line,
            f"Policy Rules Applied: {rules_line}",
        ]
    )


class IndexingAgent:
    def __init__(
        self, custom_embedder: Optional[Callable[[str], List[float]]] = None
    ) -> None:
        self._embedder = custom_embedder or embed_text

    def index_report(self, report: Report) -> Dict[str, object]:
        rag_cfg = get_models_config()["rag"]
        text = format_report_for_indexing(report)
        text_chunks = chunk_text(text, int(rag_cfg["chunk_size"]), int(rag_cfg["chunk_overlap"]))

        indexed: List[IndexedChunk] = []
        for idx, chunk in enumerate(text_chunks, start=1):
            vec = self._embedder(chunk)
            indexed.append(
                IndexedChunk(
                    chunk_id=f"{report.report_id}#chunk-{idx}",
                    collection=rag_cfg["collection_name"],
                    patient_id=report.patient_id,
                    report_id=report.report_id,
                    text=chunk,
                    embedding=vec,
                )
            )

        case_vector_store.save_report(report)
        case_vector_store.set_chunks_for_report(report.patient_id, report.report_id, indexed)

        return {
            "collection": rag_cfg["collection_name"],
            "patient_id": report.patient_id,
            "report_id": report.report_id,
            "chunks_indexed": len(indexed),
        }


def self_test() -> None:
    rag_cfg = get_models_config()["rag"]
    assert rag_cfg["chunk_size"] == 256
    assert rag_cfg["chunk_overlap"] == 64

    rpt = Report(
        report_id="RPT-PAT-1001-1",
        case_id="C1",
        patient_id="PAT-1001",
        patient_name="Elena Morales",
        requested_service="MRI Lumbar Spine",
        generated_at=datetime.now(timezone.utc).isoformat(),
        rules_applied=[RuleResult(rule_id="GEN-ELIG-001", passed=True, reason="Active")],
        decision=Decision(
            decision="APPROVED",
            confidence_score=0.94,
            key_findings=["All criteria met"],
            rule_results=[RuleResult(rule_id="GEN-ELIG-001", passed=True, reason="Active")],
            reasoning="Approved cleanly.",
            routing=DecisionRouting(escalate_to_human=False, reason="Clean"),
        ),
        human_review_required=False,
        status="auto_finalized",
    )

    case_vector_store.clear()
    agent = IndexingAgent(
        None if has_gemini_api_key() else (lambda txt: [float(len(txt) % 10), 1.0, 0.5])
    )
    res = agent.index_report(rpt)
    assert res["collection"] == "case_reports"
    assert res["patient_id"] == "PAT-1001"
    assert int(res["chunks_indexed"]) > 0
    print("[PASS] IndexingAgent self_test")


if __name__ == "__main__":
    self_test()
