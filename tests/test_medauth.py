from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from server.agents.completeness_agent import self_test as completeness_self_test
from server.agents.decision_agent import self_test as decision_self_test
from server.agents.document_agent import self_test as document_self_test
from server.agents.eligibility_agent import self_test as eligibility_self_test
from server.agents.human_review_agent import self_test as human_review_self_test
from server.agents.information_extraction_agent import self_test as extraction_self_test
from server.agents.monitor_agent import self_test as monitor_self_test
from server.agents.peer_review_agent import self_test as peer_review_self_test
from server.agents.rag.augmentation_agent import self_test as augmentation_self_test
from server.agents.rag.generation_agent import self_test as generation_self_test
from server.agents.rag.indexing_agent import self_test as indexing_self_test
from server.agents.rag.reflection_agent import self_test as reflection_self_test
from server.agents.rag.retrieval_agent import self_test as retrieval_self_test
from server.agents.reporting_agent import self_test as reporting_self_test
from server.agents.risk_agent import self_test as risk_self_test
from server.agents.verification_agent import self_test as verification_self_test
from server.main import app
from server.utils.config import get_rules_config, resolve_project_path
from server.utils.gemini_client import has_gemini_api_key
from server.workflows.medauth_workflow import (
    resume_case_human_review,
    run_case,
    run_case_chat,
)


def test_all_core_agent_self_tests():
    monitor_self_test()
    document_self_test()
    extraction_self_test()
    completeness_self_test()
    eligibility_self_test()
    verification_self_test()
    risk_self_test()
    peer_review_self_test()
    decision_self_test()
    reporting_self_test()
    human_review_self_test()


def test_all_rag_agent_self_tests():
    indexing_self_test()
    retrieval_self_test()
    augmentation_self_test()
    generation_self_test()
    reflection_self_test()


def test_empty_chat_resilience():
    res = run_case_chat(patient_id="PAT-NONEXISTENT", query="", history=[])
    assert isinstance(res.get("chat_answer"), str)
    assert len(res["chat_answer"]) > 0


def test_stub_workflow_and_sqlite_checkpoint_resume():
    wf_cfg_path = resolve_project_path("server/configs/workflow_dev.json")
    original = wf_cfg_path.read_text(encoding="utf-8")
    tmp_dir = Path(tempfile.mkdtemp(prefix="medauth-py-e2e-"))
    try:
        cfg = json.loads(original)
        for k in cfg["nodes"]:
            cfg["nodes"][k]["mode"] = "stub"
        wf_cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        res = run_case(["dummy.txt"], checkpoint_path=tmp_dir / "cp.db")
        assert res["status"] == "auto_finalized"
        assert res["report"]["decision"]["decision"] == "APPROVED"

        chat_res = run_case_chat("PAT-1001", "Is it approved?", [])
        assert "Stub Case Assistant" in chat_res["chat_answer"]
    finally:
        wf_cfg_path.write_text(original, encoding="utf-8")
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_e2e_missing_docs_interrupt_and_resume():
    client = TestClient(app)
    if not has_gemini_api_key():
        health = client.get("/api/health")
        assert health.status_code == 200
        elig = client.get("/payer/eligibility/PAT-1001")
        assert elig.status_code == 200
        assert elig.json()["eligible"] is True
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="medauth-py-hr-"))
    try:
        samples_dir = resolve_project_path(get_rules_config()["reporting"]["sample_bundles_dir"])
        files = [str(f) for f in sorted((samples_dir / "pat_1004_missing_docs").iterdir()) if f.is_file()]
        cp_db = tmp_dir / "checkpoints.db"

        paused = run_case(
            files,
            checkpoint_path=cp_db,
            mock_payer_base_url="http://testserver/payer",
            http_client=client,
        )
        assert paused["interrupted"] is True
        assert paused["report"]["decision"]["decision"] == "PENDED_ADDITIONAL_INFO"
        assert cp_db.exists()

        resumed = resume_case_human_review(
            paused["threadId"],
            "PENDED_ADDITIONAL_INFO",
            "Requested missing documents.",
            checkpoint_path=cp_db,
            mock_payer_base_url="http://testserver/payer",
            http_client=client,
        )
        assert resumed["status"] == "complete"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
