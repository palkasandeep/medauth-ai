from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import httpx

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from server.agents.completeness_agent import CompletenessAgent
from server.agents.decision_agent import DecisionAgent
from server.agents.document_agent import DocumentAgent
from server.agents.eligibility_agent import EligibilityAgent
from server.agents.human_review_agent import HumanReviewAgent
from server.agents.information_extraction_agent import InformationExtractionAgent
from server.agents.peer_review_agent import PeerReviewAgent
from server.agents.rag.augmentation_agent import AugmentationAgent
from server.agents.rag.generation_agent import GenerationAgent
from server.agents.rag.indexing_agent import IndexingAgent
from server.agents.rag.reflection_agent import ReflectionAgent
from server.agents.rag.retrieval_agent import RetrievalAgent
from server.agents.reporting_agent import ReportingAgent
from server.agents.risk_agent import RiskAgent
from server.agents.verification_agent import VerificationAgent
from server.schemas import (
    Decision,
    DecisionRouting,
    EligibilityResult,
    PatientCase,
    PeerReviewResult,
    Report,
    RiskAssessment,
    RuleResult,
    VerificationResult,
)
from server.utils.config import (
    get_rules_config,
    get_workflow_dev_config,
    resolve_project_path,
)
from server.utils.gemini_client import has_gemini_api_key


def _merge_dicts(a: Optional[Dict[str, str]], b: Optional[Dict[str, str]]) -> Dict[str, str]:
    merged = dict(a or {})
    merged.update(b or {})
    return merged


class CaseState(TypedDict, total=False):
    case_id: str
    thread_id: str
    file_paths: List[str]
    parsed_documents: List[Any]
    document_records: List[Any]
    patient_case: Any
    completeness_result: Any
    is_complete: bool
    eligibility_result: Any
    verification_result: Any
    risk_assessment: Any
    peer_review_result: Any
    applicable_rules: List[Dict[str, Any]]
    rule_results: List[Any]
    decision: Any
    decision_reflection: Any
    report: Any
    indexing_result: Dict[str, Any]
    human_review_required: bool
    human_review_result: Dict[str, Any]
    status: str
    error: Optional[str]
    node_statuses: Annotated[Dict[str, str], _merge_dicts]

    chat_patient_id: str
    chat_query: str
    chat_history: List[Dict[str, str]]
    chat_expanded_query: str
    chat_system_prompt: str
    chat_retrieved_chunks: List[Dict[str, Any]]
    chat_citations: List[Dict[str, Any]]
    chat_tools_used: List[str]
    chat_answer: str
    chat_reflection: Any


def _is_stub(node_name: str) -> bool:
    return get_workflow_dev_config().get("nodes", {}).get(node_name, {}).get("mode") == "stub"


_requested_review_report_ids: set[str] = set()


def _create_sqlite_checkpointer(
    checkpoint_path: Optional[str | Path] = None,
) -> tuple[SqliteSaver, sqlite3.Connection]:
    db_path = (
        Path(checkpoint_path).resolve()
        if checkpoint_path
        else resolve_project_path(get_workflow_dev_config()["checkpoint_store_path"])
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver, conn


def build_case_graph(
    checkpoint_path: Optional[str | Path] = None,
    mock_payer_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
    checkpointer: Optional[SqliteSaver] = None,
):
    document_agent = DocumentAgent()
    extraction_agent = InformationExtractionAgent()
    completeness_agent = CompletenessAgent()
    eligibility_agent = EligibilityAgent(mock_payer_base_url, http_client)
    verification_agent = VerificationAgent()
    risk_agent = RiskAgent(mock_payer_base_url, http_client)
    peer_review_agent = PeerReviewAgent()
    decision_agent = DecisionAgent()
    reporting_agent = ReportingAgent()
    human_review_agent = HumanReviewAgent(reporting_agent)
    indexing_agent = IndexingAgent(
        None if has_gemini_api_key() else (lambda txt: [float(len(txt) % 17), 1.0, 0.25])
    )

    def node_document(state: CaseState) -> Dict[str, Any]:
        if _is_stub("document"):
            stub_case = PatientCase(
                case_id=state.get("case_id") or "CASE-STUB",
                patient_id="PAT-1001",
                member_id="MEM-MC-9001",
                patient_name="Elena Morales",
                diagnosis_code="M54.5",
                procedure_code="72148",
                requested_service="MRI Lumbar Spine Without Contrast",
                provider_npi="1234567890",
                ordering_physician="Dr. Marcus Vance, MD",
                urgency="Routine",
                clinical_justification="8 weeks persistent lumbar radiculopathy refractory to 6 weeks PT.",
                plan_name="MedCore Gold PPO",
                lab_values={"wbc": 6.8, "glucose": 94},
                document_types_present=get_rules_config()["mandatory_docs"],
                documents=[],
            )
            return {
                "patient_case": stub_case.model_dump(),
                "status": "documents_extracted",
                "node_statuses": {"document": "completed"},
            }

        with ThreadPoolExecutor(max_workers=4) as pool:
            parsed_docs = list(pool.map(document_agent.process_file, state.get("file_paths", [])))
            records = list(pool.map(extraction_agent.extract, parsed_docs))
        merged = extraction_agent.merge_documents(records, state.get("case_id") or None)
        return {
            "document_records": [r.model_dump() for r in records],
            "patient_case": merged.model_dump(),
            "status": "documents_extracted",
            "node_statuses": {"document": "completed"},
        }

    def node_completeness(state: CaseState) -> Dict[str, Any]:
        if _is_stub("completeness"):
            return {
                "completeness_result": {"is_complete": True, "missing_docs": [], "reason": "Stub passed."},
                "is_complete": True,
                "status": "completeness_verified",
                "node_statuses": {"completeness": "completed"},
            }

        p_case = PatientCase.model_validate(state["patient_case"])
        comp = completeness_agent.check(p_case)
        if not comp.is_complete:
            pended = Decision(
                decision="PENDED_ADDITIONAL_INFO",
                confidence_score=0.99,
                key_findings=[comp.reason],
                rule_results=[RuleResult(rule_id="COMPLETENESS-CHECK", passed=False, reason=comp.reason)],
                reasoning=comp.reason,
                routing=DecisionRouting(
                    escalate_to_human=True,
                    reason="Incomplete submission: missing mandatory documents or required fields.",
                ),
            )
            return {
                "completeness_result": comp.model_dump(),
                "is_complete": False,
                "rule_results": [r.model_dump() for r in pended.rule_results],
                "decision": pended.model_dump(),
                "human_review_required": True,
                "status": "pended_incomplete",
                "node_statuses": {
                    "completeness": "completed",
                    "eligibility": "skipped",
                    "risk": "skipped",
                    "retrieval": "skipped",
                    "decision": "skipped",
                    "reflection": "skipped",
                },
            }

        return {
            "completeness_result": comp.model_dump(),
            "is_complete": True,
            "status": "completeness_verified",
            "node_statuses": {"completeness": "completed"},
        }

    def route_after_completeness(state: CaseState) -> str:
        return "eligibility" if state.get("is_complete") else "reporting"

    def node_eligibility(state: CaseState) -> Dict[str, Any]:
        if _is_stub("eligibility"):
            elig = EligibilityResult(
                eligible=True,
                eligibility_reason="Stub active MedCore Gold PPO coverage.",
                coverage_percentage=100,
                procedure_cost=1250.0,
                patient_responsibility=0.0,
                plan_name="MedCore Gold PPO",
                member_status="ACTIVE",
            )
            return {
                "eligibility_result": elig.model_dump(),
                "status": "eligibility_checked",
                "node_statuses": {"eligibility": "completed"},
            }

        p_case = PatientCase.model_validate(state["patient_case"])
        elig = eligibility_agent.verify(p_case)
        return {
            "eligibility_result": elig.model_dump(),
            "status": "eligibility_checked",
            "node_statuses": {"eligibility": "completed"},
        }

    def node_risk(state: CaseState) -> Dict[str, Any]:
        if _is_stub("risk"):
            ver = VerificationResult(plausible=True, confidence=0.94, reasoning="Stub plausible.")
            risk = RiskAssessment(
                risk_level="low",
                risk_score=0.08,
                crosswalk_match=True,
                is_duplicate=False,
                provider_verified=True,
                provider_sanctioned=False,
                abnormal_labs=[],
                fraud_indicators=[],
                requires_peer_review=False,
                reasoning="Stub low-risk assessment.",
            )
            peer = PeerReviewResult(
                narrative="Stub clean low-risk peer review summary.",
                risk_level="low",
                risk_score=0.08,
                fraud_indicators=[],
                requires_peer_review=False,
            )
            return {
                "verification_result": ver.model_dump(),
                "risk_assessment": risk.model_dump(),
                "peer_review_result": peer.model_dump(),
                "status": "risk_assessed",
                "node_statuses": {"risk": "completed"},
            }

        p_case = PatientCase.model_validate(state["patient_case"])
        ver = verification_agent.verify_clinical_justification(
            p_case.diagnosis_code,
            p_case.procedure_code,
            p_case.requested_service,
            p_case.clinical_justification,
        )
        risk = risk_agent.assess(p_case)
        peer = peer_review_agent.synthesize(risk, p_case)
        return {
            "verification_result": ver.model_dump(),
            "risk_assessment": risk.model_dump(),
            "peer_review_result": peer.model_dump(),
            "status": "risk_assessed",
            "node_statuses": {"risk": "completed"},
        }

    def node_retrieval(state: CaseState) -> Dict[str, Any]:
        if _is_stub("retrieval"):
            return {
                "applicable_rules": get_rules_config()["policy_rules"]["general"],
                "status": "rules_retrieved",
                "node_statuses": {"retrieval": "completed"},
            }
        p_case = PatientCase.model_validate(state["patient_case"])
        rules = decision_agent.get_applicable_rules(p_case.procedure_code)
        return {
            "applicable_rules": rules,
            "status": "rules_retrieved",
            "node_statuses": {"retrieval": "completed"},
        }

    def node_decision(state: CaseState) -> Dict[str, Any]:
        if _is_stub("decision"):
            stub_dec = Decision(
                decision="APPROVED",
                confidence_score=0.94,
                key_findings=["Stub all rules satisfied"],
                rule_results=[RuleResult(rule_id="GEN-ELIG-001", passed=True, reason="Active")],
                reasoning="Stub approved determination.",
                routing=DecisionRouting(escalate_to_human=False, reason="Clean stub auto-approval."),
            )
            return {
                "rule_results": [r.model_dump() for r in stub_dec.rule_results],
                "decision": stub_dec.model_dump(),
                "human_review_required": False,
                "status": "decision_synthesized",
                "node_statuses": {"decision": "completed"},
            }

        p_case = PatientCase.model_validate(state["patient_case"])
        elig = EligibilityResult.model_validate(state["eligibility_result"])
        ver = (
            VerificationResult.model_validate(state["verification_result"])
            if state.get("verification_result")
            else None
        )
        risk = RiskAssessment.model_validate(state["risk_assessment"])
        peer = (
            PeerReviewResult.model_validate(state["peer_review_result"])
            if state.get("peer_review_result")
            else None
        )

        rule_results = decision_agent.apply_rules(
            state.get("applicable_rules", []),
            patient_case=p_case,
            eligibility=elig,
            verification=ver,
            risk=risk,
        )
        dec = decision_agent.synthesize(
            patient_case=p_case,
            eligibility=elig,
            risk=risk,
            rule_results=rule_results,
            verification=ver,
            peer_review=peer,
            has_error=bool(state.get("error")),
        )
        return {
            "rule_results": [r.model_dump() for r in rule_results],
            "decision": dec.model_dump(),
            "human_review_required": dec.routing.escalate_to_human,
            "status": "decision_synthesized",
            "node_statuses": {"decision": "completed"},
        }

    def node_reflection(state: CaseState) -> Dict[str, Any]:
        if _is_stub("reflection"):
            return {
                "decision_reflection": {
                    "groundedness": 1.0,
                    "relevance": 1.0,
                    "confidence_alignment": 1.0,
                    "passed": True,
                    "feedback": "Stub self-check passed.",
                },
                "status": "decision_reflected",
                "node_statuses": {"reflection": "completed"},
            }

        dec = Decision.model_validate(state["decision"])
        refl = decision_agent.reflect(dec)
        escalate = dec.routing.escalate_to_human
        if not refl.passed:
            escalate = True
            dec = dec.model_copy(
                update={
                    "routing": DecisionRouting(
                        escalate_to_human=True,
                        reason=f"{dec.routing.reason}; Self-check failed: {refl.feedback}",
                    )
                }
            )
        return {
            "decision_reflection": refl.model_dump(),
            "decision": dec.model_dump(),
            "human_review_required": escalate,
            "status": "decision_reflected",
            "node_statuses": {"reflection": "completed"},
        }

    def node_reporting(state: CaseState) -> Dict[str, Any]:
        p_case = PatientCase.model_validate(state["patient_case"])
        dec = Decision.model_validate(state["decision"])
        rules_applied = [RuleResult.model_validate(r) for r in state.get("rule_results", [])]
        elig = (
            EligibilityResult.model_validate(state["eligibility_result"])
            if state.get("eligibility_result")
            else None
        )
        ver = (
            VerificationResult.model_validate(state["verification_result"])
            if state.get("verification_result")
            else None
        )
        risk = (
            RiskAssessment.model_validate(state["risk_assessment"])
            if state.get("risk_assessment")
            else None
        )
        peer = (
            PeerReviewResult.model_validate(state["peer_review_result"])
            if state.get("peer_review_result")
            else None
        )

        if _is_stub("reporting"):
            rpt = Report(
                report_id=f"RPT-{p_case.patient_id or 'STUB'}-STUB",
                case_id=state.get("case_id") or "CASE-STUB",
                patient_id=p_case.patient_id or "PAT-1001",
                patient_name=p_case.patient_name or "Stub Patient",
                requested_service=p_case.requested_service or "Stub Service",
                generated_at="2026-11-15T00:00:00Z",
                eligibility=elig,
                risk=risk,
                rules_applied=rules_applied,
                decision=dec,
                human_review_required=bool(state.get("human_review_required", False)),
                status="awaiting_human_review" if state.get("human_review_required") else "auto_finalized",
            )
            return {
                "report": rpt.model_dump(),
                "status": rpt.status,
                "node_statuses": {"reporting": "completed"},
            }

        escalate = (
            state["human_review_required"]
            if "human_review_required" in state
            else dec.routing.escalate_to_human
        )
        rpt = reporting_agent.generate(
            patient_case=p_case,
            rules_applied=rules_applied,
            decision=dec,
            eligibility=elig,
            verification=ver,
            risk=risk,
            peer_review=peer,
            status="awaiting_human_review" if escalate else "auto_finalized",
            case_id=state.get("case_id") or p_case.case_id,
            human_review_required=escalate,
        )
        return {
            "report": rpt.model_dump(),
            "human_review_required": rpt.human_review_required,
            "status": rpt.status,
            "node_statuses": {"reporting": "completed"},
        }

    def node_indexing(state: CaseState) -> Dict[str, Any]:
        rpt = Report.model_validate(state["report"]) if state.get("report") else None
        if _is_stub("indexing"):
            idx_res = {
                "collection": "case_reports",
                "patient_id": rpt.patient_id if rpt else "STUB",
                "chunks_indexed": 2,
                "stub": True,
            }
        elif rpt:
            idx_res = indexing_agent.index_report(rpt)
        else:
            idx_res = {"chunks_indexed": 0}

        if rpt and not state.get("human_review_required", False):
            auto_hr = human_review_agent.auto_finalize(rpt.report_id, rpt.patient_id)
            updated_rpt = rpt.model_copy(
                update={
                    "status": auto_hr["status"],
                    "reviewer_decision": auto_hr["reviewer_decision"],
                }
            )
            if updated_rpt.json_path and Path(updated_rpt.json_path).exists():
                Path(updated_rpt.json_path).write_text(
                    updated_rpt.model_dump_json(indent=2), encoding="utf-8"
                )
            return {
                "indexing_result": idx_res,
                "human_review_result": auto_hr,
                "report": updated_rpt.model_dump(),
                "status": "auto_finalized",
                "node_statuses": {"indexing": "completed", "human_review": "skipped"},
            }

        return {
            "indexing_result": idx_res,
            "status": "awaiting_human_review",
            "node_statuses": {"indexing": "completed"},
        }

    def route_after_indexing(state: CaseState) -> str:
        return "human_review" if state.get("human_review_required") else END

    def node_human_review(state: CaseState) -> Dict[str, Any]:
        if _is_stub("human_review"):
            return {
                "status": "awaiting_human_review",
                "human_review_result": {"status": "pending_review"},
                "node_statuses": {"human_review": "completed"},
            }

        rpt = Report.model_validate(state["report"])
        if rpt.report_id not in _requested_review_report_ids:
            _requested_review_report_ids.add(rpt.report_id)
            human_review_agent.request_review(
                rpt.report_id,
                rpt.patient_id,
                rpt.decision.routing.reason,
            )

        payload = interrupt(
            {
                "status": "awaiting_human_review",
                "report_id": rpt.report_id,
                "patient_id": rpt.patient_id,
                "decision": rpt.decision.decision,
                "reason": rpt.decision.routing.reason,
            }
        )
        reviewer_decision = (payload or {}).get("reviewerDecision", "APPROVED")
        comments = (payload or {}).get("comments", "")

        completed = human_review_agent.submit_reviewer_decision(
            rpt.report_id, rpt.patient_id, reviewer_decision, comments
        )
        updated_rpt = rpt.model_copy(
            update={
                "status": "complete",
                "reviewer_decision": reviewer_decision,
                "reviewer_comments": comments,
            }
        )
        return {
            "human_review_result": completed,
            "report": updated_rpt.model_dump(),
            "status": "complete",
            "node_statuses": {"human_review": "completed"},
        }

    graph = StateGraph(CaseState)
    graph.add_node("document", node_document)
    graph.add_node("completeness", node_completeness)
    graph.add_node("eligibility", node_eligibility)
    graph.add_node("risk", node_risk)
    graph.add_node("retrieval", node_retrieval)
    graph.add_node("decision", node_decision)
    graph.add_node("reflection", node_reflection)
    graph.add_node("reporting", node_reporting)
    graph.add_node("indexing", node_indexing)
    graph.add_node("human_review", node_human_review)

    graph.add_edge(START, "document")
    graph.add_edge("document", "completeness")
    graph.add_conditional_edges("completeness", route_after_completeness)
    graph.add_edge("eligibility", "risk")
    graph.add_edge("risk", "retrieval")
    graph.add_edge("retrieval", "decision")
    graph.add_edge("decision", "reflection")
    graph.add_edge("reflection", "reporting")
    graph.add_edge("reporting", "indexing")
    graph.add_conditional_edges("indexing", route_after_indexing)
    graph.add_edge("human_review", END)

    if checkpointer is None:
        checkpointer, _ = _create_sqlite_checkpointer(checkpoint_path)

    return graph.compile(checkpointer=checkpointer)


def build_chat_graph():
    retrieval_agent = RetrievalAgent(
        None if has_gemini_api_key() else (lambda txt: [float(len(txt) % 17), 1.0, 0.25])
    )
    augmentation_agent = AugmentationAgent()
    generation_agent = GenerationAgent(retrieval_agent)
    reflection_agent = ReflectionAgent()

    def chat_node(state: CaseState) -> Dict[str, Any]:
        if _is_stub("chat"):
            pid = state.get("chat_patient_id") or "UNKNOWN"
            q = state.get("chat_query") or "No query"
            return {
                "chat_answer": f"Stub Case Assistant answer for patient {pid}: {q}",
                "chat_citations": [],
                "chat_reflection": {
                    "answer_relevance": 0.9,
                    "groundedness": 0.9,
                    "context_relevance": 0.9,
                    "overall_score": 0.9,
                    "passed": True,
                    "feedback": "Stub RAG reflection passed.",
                },
            }

        patient_id = (state.get("chat_patient_id") or "").strip()
        query = (state.get("chat_query") or "").strip()
        history = state.get("chat_history") or []

        try:
            initial_chunks = (
                retrieval_agent.retrieve(patient_id, query or "case overview")
                if patient_id
                else []
            )
            aug = augmentation_agent.augment(
                patient_id=patient_id or "UNKNOWN",
                query=query,
                chunks=initial_chunks,
                history=history,
            )
            refined = (
                retrieval_agent.retrieve(patient_id, aug["expanded_query"])
                if patient_id and aug["expanded_query"] != query
                else initial_chunks
            )

            merged_map = {c["chunk_id"]: c for c in (initial_chunks + refined)}
            active_chunks = list(merged_map.values())

            gen = generation_agent.generate(
                patient_id=patient_id or "UNKNOWN",
                query=query,
                system_prompt=aug["system_prompt"],
                initial_chunks=active_chunks,
            )
            ctx_texts = (
                [c["text"] for c in gen["citations"]]
                if gen["citations"]
                else [c["text"] for c in active_chunks]
            )
            refl = reflection_agent.evaluate(query, gen["answer"], ctx_texts)

            return {
                "chat_expanded_query": aug["expanded_query"],
                "chat_system_prompt": aug["system_prompt"],
                "chat_retrieved_chunks": active_chunks,
                "chat_citations": gen["citations"],
                "chat_tools_used": gen["tools_used"],
                "chat_answer": gen["answer"],
                "chat_reflection": refl.model_dump(),
            }
        except Exception as exc:
            return {
                "chat_answer": f"Unable to process Case Assistant query safely: {exc}",
                "chat_citations": [],
                "chat_tools_used": [],
                "chat_reflection": {
                    "answer_relevance": 0.0,
                    "groundedness": 0.0,
                    "context_relevance": 0.0,
                    "overall_score": 0.0,
                    "passed": False,
                    "feedback": "Fallback safe response.",
                },
            }

    g = StateGraph(CaseState)
    g.add_node("chat_node", chat_node)
    g.add_edge(START, "chat_node")
    g.add_edge("chat_node", END)
    return g.compile()


def run_case(
    file_paths: List[str],
    thread_id: Optional[str] = None,
    case_id: Optional[str] = None,
    checkpoint_path: Optional[str | Path] = None,
    mock_payer_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    tid = thread_id or f"thread-{uuid.uuid4()}"
    cid = case_id or f"CASE-{str(uuid.uuid4())[:8]}"
    saver, conn = _create_sqlite_checkpointer(checkpoint_path)
    try:
        graph = build_case_graph(
            mock_payer_base_url=mock_payer_base_url,
            http_client=http_client,
            checkpointer=saver,
        )
        result = graph.invoke(
            {"case_id": cid, "thread_id": tid, "file_paths": file_paths},
            config={"configurable": {"thread_id": tid}},
        )
        interrupted = bool(result.get("__interrupt__"))
        return {
            "threadId": tid,
            "caseId": cid,
            "interrupted": interrupted,
            "state": result,
            "report": result.get("report"),
            "status": "awaiting_human_review" if interrupted else result.get("status"),
            "nodeStatuses": result.get("node_statuses", {}),
        }
    finally:
        conn.close()


def resume_case_human_review(
    thread_id: str,
    reviewer_decision: str,
    comments: str = "",
    checkpoint_path: Optional[str | Path] = None,
    mock_payer_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    saver, conn = _create_sqlite_checkpointer(checkpoint_path)
    try:
        graph = build_case_graph(
            mock_payer_base_url=mock_payer_base_url,
            http_client=http_client,
            checkpointer=saver,
        )
        result = graph.invoke(
            Command(
                resume={
                    "reviewerDecision": reviewer_decision,
                    "comments": comments,
                }
            ),
            config={"configurable": {"thread_id": thread_id}},
        )
        return {
            "threadId": thread_id,
            "status": result.get("status"),
            "report": result.get("report"),
            "humanReviewResult": result.get("human_review_result"),
            "nodeStatuses": result.get("node_statuses", {}),
        }
    finally:
        conn.close()


def run_case_chat(
    patient_id: str,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    chat_graph = build_chat_graph()
    return chat_graph.invoke(
        {
            "chat_patient_id": patient_id or "",
            "chat_query": query or "",
            "chat_history": history or [],
        }
    )
