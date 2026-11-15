from __future__ import annotations

from typing import Any, Dict, List, Optional
from google.genai import types

from server.agents.rag.retrieval_agent import RetrievalAgent
from server.agents.rag.vector_store import case_vector_store
from server.utils.config import resolve_agent_model
from server.utils.gemini_client import get_gemini_client


class GenerationAgent:
    def __init__(self, retrieval_agent: Optional[RetrievalAgent] = None) -> None:
        self.retrieval_agent = retrieval_agent or RetrievalAgent()

    def generate(
        self,
        patient_id: str,
        query: str,
        system_prompt: str,
        initial_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        collected_chunks: Dict[str, Dict[str, Any]] = {
            c["chunk_id"]: c for c in (initial_chunks or [])
        }

        if not (query or "").strip():
            return {
                "answer": "Please enter a question about this patient's prior authorization report.",
                "tools_used": [],
                "citations": list(collected_chunks.values()),
            }

        report = case_vector_store.get_report(patient_id)
        patient_chunks = case_vector_store.get_chunks_for_patient(patient_id)

        if report is None and not patient_chunks:
            return {
                "answer": f"No finished PA report was found for patient '{patient_id}'. Please process a case first before asking questions.",
                "tools_used": [],
                "citations": [],
            }

        tools_used: List[str] = []

        def search_docs(search_query: str) -> Any:
            """Search indexed report chunks for the active patient using a semantic query."""
            tools_used.append("search_docs")
            found = self.retrieval_agent.retrieve(patient_id, search_query)
            for c in found:
                collected_chunks[c["chunk_id"]] = c
            return found or {"message": "No matching report chunks found for this patient."}

        def get_report() -> Dict[str, Any]:
            """Retrieve the saved PA report overview for the active patient."""
            tools_used.append("get_report")
            if not report:
                return {"found": False, "message": f"No report for '{patient_id}'."}
            return {
                "report_id": report.report_id,
                "patient_id": report.patient_id,
                "patient_name": report.patient_name,
                "requested_service": report.requested_service,
                "status": report.status,
                "human_review_required": report.human_review_required,
                "eligibility": report.eligibility.model_dump() if report.eligibility else None,
            }

        def get_risk() -> Dict[str, Any]:
            """Retrieve risk assessment, crosswalk status, lab anomalies, and fraud indicators."""
            tools_used.append("get_risk")
            if not report or not report.risk:
                return {"message": "No risk assessment recorded on this report."}
            return report.risk.model_dump()

        def get_decision() -> Dict[str, Any]:
            """Retrieve the PA determination, confidence score, policy rule results, and routing."""
            tools_used.append("get_decision")
            if not report:
                return {"found": False}
            return {
                "decision": report.decision.model_dump(),
                "rules_applied": [r.model_dump() for r in report.rules_applied],
                "reviewer_decision": report.reviewer_decision,
            }

        def summarize() -> Dict[str, Any]:
            """Generate a concise structured summary of the active patient's PA report."""
            tools_used.append("summarize")
            if not report:
                return {"found": False}
            return {
                "report_id": report.report_id,
                "patient": f"{report.patient_name} ({report.patient_id})",
                "service": report.requested_service,
                "determination": report.decision.decision,
                "confidence": report.decision.confidence_score,
                "risk_level": report.risk.risk_level if report.risk else "N/A",
                "human_review_required": report.human_review_required,
                "reasoning": report.decision.reasoning,
            }

        try:
            client = get_gemini_client()
            model, temperature = resolve_agent_model("rag_generation")
            chat = client.chats.create(
                model=model,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_prompt,
                    tools=[search_docs, get_report, get_risk, get_decision, summarize],
                ),
            )
            resp = chat.send_message(query)
            answer_text = (resp.text or "").strip() or (
                "Unable to synthesize a response from the available report context."
            )
            return {
                "answer": answer_text,
                "tools_used": list(dict.fromkeys(tools_used)),
                "citations": [
                    {"chunk_id": c["chunk_id"], "text": c["text"], "score": c.get("score")}
                    for c in collected_chunks.values()
                ],
            }
        except Exception as exc:
            fallback = (
                f"Report {report.report_id} ({report.patient_name}): Determination is "
                f"{report.decision.decision} (confidence {report.decision.confidence_score * 100:.0f}%). "
                f"{report.decision.reasoning}"
                if report
                else f"Unable to query Gemini ({exc})."
            )
            return {
                "answer": fallback,
                "tools_used": list(dict.fromkeys(tools_used)),
                "citations": [
                    {"chunk_id": c["chunk_id"], "text": c["text"], "score": c.get("score")}
                    for c in collected_chunks.values()
                ],
            }


def self_test() -> None:
    case_vector_store.clear()
    agent = GenerationAgent()

    missing = agent.generate("PAT-UNKNOWN", "Why was this denied?", "Ground in report.")
    assert "No finished PA report" in missing["answer"]
    assert missing["citations"] == []

    empty_q = agent.generate("PAT-UNKNOWN", "   ", "Ground in report.")
    assert len(empty_q["answer"]) > 0
    print("[PASS] GenerationAgent self_test")


if __name__ == "__main__":
    self_test()
