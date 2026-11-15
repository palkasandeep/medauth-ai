from __future__ import annotations

from datetime import datetime, timezone
import html
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.schemas import (
    Decision,
    DecisionReflection,
    DecisionRouting,
    EligibilityResult,
    PatientCase,
    PeerReviewResult,
    Report,
    RiskAssessment,
    RuleResult,
    VerificationResult,
)
from server.utils.config import get_rules_config, resolve_project_path


class ReportingAgent:
    def __init__(
        self,
        json_dir: Optional[str | Path] = None,
        html_dir: Optional[str | Path] = None,
        audit_log_path: Optional[str | Path] = None,
    ) -> None:
        cfg = get_rules_config()["reporting"]
        self.json_dir = Path(json_dir).resolve() if json_dir else resolve_project_path(cfg["json_dir"])
        self.html_dir = Path(html_dir).resolve() if html_dir else resolve_project_path(cfg["html_dir"])
        self.audit_log_path = (
            Path(audit_log_path).resolve()
            if audit_log_path
            else resolve_project_path(cfg["audit_log_path"])
        )

    def log_event(self, event: str, details: Optional[Dict[str, Any]] = None) -> str:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        line = f"{ts} | {event} | {json.dumps(details or {})}"
        with self.audit_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return line

    def generate(
        self,
        patient_case: PatientCase,
        rules_applied: List[RuleResult],
        decision: Decision,
        eligibility: Optional[EligibilityResult] = None,
        verification: Optional[VerificationResult] = None,
        risk: Optional[RiskAssessment] = None,
        peer_review: Optional[PeerReviewResult] = None,
        decision_reflection: Optional[DecisionReflection] = None,
        status: Optional[str] = None,
        case_id: Optional[str] = None,
        human_review_required: Optional[bool] = None,
    ) -> Report:
        patient_id = patient_case.patient_id or "UNKNOWN"
        patient_name = patient_case.patient_name or "Unknown Patient"
        service = patient_case.requested_service or "Unspecified Service"
        ts_ms = int(time.time() * 1000)
        report_id = f"RPT-{patient_id}-{ts_ms}"

        hr_req = (
            human_review_required
            if human_review_required is not None
            else (decision.routing.escalate_to_human if decision and decision.routing else True)
        )

        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.json_dir / f"{report_id}.json"
        html_path = self.html_dir / f"{report_id}.html"

        report = Report(
            report_id=report_id,
            case_id=case_id or patient_case.case_id,
            patient_id=patient_id,
            patient_name=patient_name,
            requested_service=service,
            generated_at=datetime.now(timezone.utc).isoformat(),
            eligibility=eligibility,
            verification=verification,
            risk=risk,
            peer_review=peer_review,
            rules_applied=rules_applied,
            decision=decision,
            decision_reflection=decision_reflection,
            human_review_required=hr_req,
            status=status or ("awaiting_human_review" if hr_req else "auto_finalized"),
            json_path=str(json_path),
            html_path=str(html_path),
        )

        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        rows = "".join(
            f"<tr><td>{html.escape(r.rule_id)}</td><td>{'PASS' if r.passed else 'FAIL'}</td>"
            f"<td>{html.escape(r.reason)}</td></tr>"
            for r in rules_applied
        )
        elig_txt = (
            f"{'Eligible' if eligibility.eligible else 'Ineligible'} "
            f"({eligibility.coverage_percentage}% coverage, Patient Resp: ${eligibility.patient_responsibility:.2f})"
            if eligibility
            else "Not Evaluated"
        )
        risk_txt = f"{risk.risk_level} (score {risk.risk_score})" if risk else "Not Evaluated"
        badge_cls = "flag-yes" if hr_req else "flag-no"
        badge_txt = "YES — Manual Sign-Off Required" if hr_req else "NO — Auto-Finalized"

        html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MedCore PA Report - {html.escape(report_id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #0f172a; background: #f8fafc; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin-bottom: 18px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }}
    .flag-yes {{ background: #fef3c7; color: #92400e; }}
    .flag-no {{ background: #d1fae5; color: #065f46; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>MedCore Prior Authorization Report</h1>
    <p><strong>Report ID:</strong> {html.escape(report_id)} | <strong>Generated:</strong> {html.escape(report.generated_at)}</p>
    <p><strong>Patient Name:</strong> {html.escape(patient_name)} (<strong>ID:</strong> {html.escape(patient_id)})</p>
    <p><strong>Requested Service:</strong> {html.escape(service)}</p>
    <p><strong>Determination:</strong> {html.escape(decision.decision)} (Confidence: {decision.confidence_score * 100:.1f}%)</p>
    <p><strong>Human Review Required:</strong> <span class="badge {badge_cls}">{badge_txt}</span></p>
  </div>
  <div class="card">
    <h2>Eligibility &amp; Risk Summary</h2>
    <p><strong>Eligibility:</strong> {html.escape(elig_txt)}</p>
    <p><strong>Risk Level:</strong> {html.escape(risk_txt)}</p>
    <p><strong>Decision Reasoning:</strong> {html.escape(decision.reasoning)}</p>
  </div>
  <div class="card">
    <h2>Policy Rules Applied</h2>
    <table><thead><tr><th>Rule ID</th><th>Status</th><th>Reason</th></tr></thead><tbody>{rows}</tbody></table>
  </div>
</body>
</html>"""
        html_path.write_text(html_doc, encoding="utf-8")

        self.log_event(
            "report_generated",
            {
                "report_id": report_id,
                "patient_id": patient_id,
                "decision": decision.decision,
                "human_review_required": hr_req,
            },
        )
        return report


def self_test() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="medauth-rpt-"))
    try:
        agent = ReportingAgent(
            json_dir=tmp_dir / "json",
            html_dir=tmp_dir / "html",
            audit_log_path=tmp_dir / "audit.log",
        )
        p_case = PatientCase(
            case_id="C100",
            patient_id="PAT-1001",
            patient_name="Elena Morales",
            requested_service="MRI Lumbar Spine",
        )
        dec = Decision(
            decision="APPROVED",
            confidence_score=0.94,
            key_findings=["All criteria met"],
            rule_results=[RuleResult(rule_id="GEN-ELIG-001", passed=True, reason="Active")],
            reasoning="Clean approval.",
            routing=DecisionRouting(escalate_to_human=False, reason="Auto-approved"),
        )
        rpt = agent.generate(p_case, dec.rule_results, dec)
        assert rpt.report_id.startswith("RPT-PAT-1001-")
        assert rpt.human_review_required is False
        assert Path(rpt.json_path).exists()
        assert Path(rpt.html_path).exists()
        html_txt = Path(rpt.html_path).read_text(encoding="utf-8")
        assert "Elena Morales" in html_txt
        assert "Human Review Required" in html_txt

        override_rpt = agent.generate(p_case, dec.rule_results, dec, human_review_required=True)
        assert override_rpt.human_review_required is True
        print("[PASS] ReportingAgent self_test")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    self_test()
