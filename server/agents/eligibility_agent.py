from __future__ import annotations

from typing import Optional
import httpx

from server.schemas import EligibilityResult, PatientCase
from server.utils.config import get_workflow_dev_config


class EligibilityAgent:
    def __init__(
        self,
        mock_payer_base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = (
            mock_payer_base_url or get_workflow_dev_config()["mock_payer_base_url"]
        ).rstrip("/")
        self._client = http_client

    def calculate_patient_responsibility(
        self, procedure_cost: float, coverage_percentage: float
    ) -> float:
        pct = max(0.0, min(100.0, float(coverage_percentage)))
        if pct >= 100.0:
            return 0.0
        return round(float(procedure_cost) * (1.0 - pct / 100.0), 2)

    def _get_json(self, url: str) -> tuple[int, dict]:
        if self._client is not None:
            resp = self._client.get(url)
            return resp.status_code, resp.json()
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            return resp.status_code, resp.json()

    def verify(self, patient_case: PatientCase) -> EligibilityResult:
        lookup_id = patient_case.member_id or patient_case.patient_id
        if not lookup_id:
            return EligibilityResult(
                eligible=False,
                eligibility_reason="Missing both member_id and patient_id; unable to verify MedCore eligibility.",
                coverage_percentage=0,
                procedure_cost=0.0,
                patient_responsibility=0.0,
                plan_name=patient_case.plan_name or "Unknown",
                member_status="MISSING_ID",
            )

        _, member = self._get_json(f"{self.base_url}/eligibility/{lookup_id}")

        procedure_cost = 0.0
        if patient_case.procedure_code:
            status_code, cost_data = self._get_json(
                f"{self.base_url}/procedure-cost/{patient_case.procedure_code}"
            )
            if status_code == 200:
                procedure_cost = float(cost_data.get("allowed_amount", 0.0))

        is_active = bool(member.get("eligible")) and str(member.get("status", "")).upper() == "ACTIVE"
        covered_list = member.get("covered_procedures") or []
        proc_covered = (
            len(covered_list) == 0
            or not patient_case.procedure_code
            or patient_case.procedure_code in covered_list
        )
        eligible = is_active and proc_covered
        cov_pct = float(member.get("coverage_percentage", 0.0)) if eligible else 0.0
        patient_resp = (
            self.calculate_patient_responsibility(procedure_cost, cov_pct)
            if eligible
            else procedure_cost
        )
        reason = str(member.get("reason") or "").strip() or (
            f"Member {lookup_id} holds active {member.get('plan_name', 'MedCore')} coverage."
            if eligible
            else f"Member {lookup_id} is ineligible (status: {member.get('status', 'UNKNOWN')})."
        )

        return EligibilityResult(
            eligible=eligible,
            eligibility_reason=reason,
            coverage_percentage=cov_pct,
            procedure_cost=procedure_cost,
            patient_responsibility=patient_resp,
            plan_name=str(member.get("plan_name") or patient_case.plan_name or "Unknown"),
            member_status=str(member.get("status") or "UNKNOWN"),
        )


def self_test() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from server.mockPayer.router import router as payer_router

    app = FastAPI()
    app.include_router(payer_router)
    client = TestClient(app)

    agent = EligibilityAgent("http://testserver/payer", http_client=client)
    assert agent.calculate_patient_responsibility(1250, 100) == 0.0
    assert agent.calculate_patient_responsibility(1000, 80) == 200.0

    r1 = agent.verify(
        PatientCase(
            case_id="T1",
            patient_id="PAT-1001",
            member_id="MEM-MC-9001",
            procedure_code="72148",
        )
    )
    assert r1.eligible is True
    assert r1.coverage_percentage == 100.0
    assert r1.patient_responsibility == 0.0
    assert len(r1.eligibility_reason) > 0

    r2 = agent.verify(
        PatientCase(
            case_id="T2",
            patient_id="PAT-1002",
            member_id="MEM-MC-9002",
            procedure_code="27447",
        )
    )
    assert r2.eligible is False
    assert len(r2.eligibility_reason) > 0
    print("[PASS] EligibilityAgent self_test")


if __name__ == "__main__":
    self_test()
