from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from server.agents.reporting_agent import ReportingAgent
from server.utils.config import get_rules_config, resolve_project_path


class HumanReviewAgent:
    def __init__(
        self,
        reporting_agent: Optional[ReportingAgent] = None,
        json_dir: Optional[str | Path] = None,
    ) -> None:
        self.reporting_agent = reporting_agent or ReportingAgent()
        self.json_dir = (
            Path(json_dir).resolve()
            if json_dir
            else resolve_project_path(get_rules_config()["reporting"]["json_dir"])
        )

    def auto_finalize(self, report_id: str, patient_id: str) -> Dict[str, Any]:
        self.reporting_agent.log_event(
            "auto_finalized_no_review_required",
            {"report_id": report_id, "patient_id": patient_id},
        )
        return {
            "status": "auto_finalized",
            "reviewer_decision": "auto_approved",
            "report_id": report_id,
            "patient_id": patient_id,
        }

    def request_review(
        self, report_id: str, patient_id: str, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        details: Dict[str, Any] = {"report_id": report_id, "patient_id": patient_id}
        if reason:
            details["reason"] = reason
        self.reporting_agent.log_event("clinical_review_requested", details)
        return {
            "status": "pending_review",
            "report_id": report_id,
            "patient_id": patient_id,
        }

    def submit_reviewer_decision(
        self,
        report_id: str,
        patient_id: str,
        reviewer_decision: str,
        comments: str = "",
    ) -> Dict[str, Any]:
        self.reporting_agent.log_event(
            "clinical_review_completed",
            {
                "report_id": report_id,
                "patient_id": patient_id,
                "reviewer_decision": reviewer_decision,
                "comments": comments,
            },
        )
        rpt_file = self.json_dir / f"{report_id}.json"
        if rpt_file.exists():
            data = json.loads(rpt_file.read_text(encoding="utf-8"))
            data["status"] = "complete"
            data["reviewer_decision"] = reviewer_decision
            data["reviewer_comments"] = comments
            rpt_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        return {
            "status": "complete",
            "reviewer_decision": reviewer_decision,
            "reviewer_comments": comments,
            "report_id": report_id,
            "patient_id": patient_id,
        }


def self_test() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="medauth-hr-"))
    try:
        log1 = tmp_dir / "audit1.log"
        hr1 = HumanReviewAgent(
            ReportingAgent(tmp_dir / "json", tmp_dir / "html", log1),
            tmp_dir / "json",
        )
        auto_res = hr1.auto_finalize("RPT-1", "PAT-1")
        assert auto_res["status"] == "auto_finalized"
        assert auto_res["reviewer_decision"] == "auto_approved"
        txt1 = log1.read_text(encoding="utf-8")
        assert "auto_finalized_no_review_required" in txt1
        assert "clinical_review_requested" not in txt1

        log2 = tmp_dir / "audit2.log"
        hr2 = HumanReviewAgent(
            ReportingAgent(tmp_dir / "json", tmp_dir / "html", log2),
            tmp_dir / "json",
        )
        req_res = hr2.request_review("RPT-2", "PAT-2", "High risk")
        assert req_res["status"] == "pending_review"
        txt2 = log2.read_text(encoding="utf-8")
        assert "clinical_review_requested" in txt2
        assert "auto_finalized_no_review_required" not in txt2

        sub_res = hr2.submit_reviewer_decision("RPT-2", "PAT-2", "DENIED", "Confirmed mismatch")
        assert sub_res["status"] == "complete"
        assert sub_res["reviewer_decision"] == "DENIED"
        print("[PASS] HumanReviewAgent self_test")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    self_test()
