from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import httpx

from server.schemas import AbnormalLabFinding, PatientCase, RiskAssessment
from server.utils.config import (
    get_persona_config,
    get_rules_config,
    get_workflow_dev_config,
)
from server.utils.gemini_client import generate_json_with_config, has_gemini_api_key


class RiskAgent:
    def __init__(
        self,
        mock_payer_base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = (
            mock_payer_base_url or get_workflow_dev_config()["mock_payer_base_url"]
        ).rstrip("/")
        self._client = http_client

    def _get_json(self, url: str) -> tuple[int, dict]:
        if self._client is not None:
            resp = self._client.get(url)
            return resp.status_code, resp.json()
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            return resp.status_code, resp.json()

    def is_crosswalk_match(self, diagnosis_code: str, procedure_code: str) -> bool:
        allowed = get_rules_config()["crosswalk"].get(procedure_code.strip(), [])
        if not allowed:
            return False
        norm_dx = diagnosis_code.strip().upper()
        return any(
            norm_dx == c.strip().upper()
            or norm_dx.startswith(c.strip().upper())
            or c.strip().upper().startswith(norm_dx)
            for c in allowed
        )

    def is_duplicate_request(
        self,
        prior_claims: List[Dict[str, Any]],
        procedure_code: str,
        reference_date: Optional[datetime] = None,
        window_days: Optional[int] = None,
    ) -> bool:
        window = window_days if window_days is not None else int(get_rules_config()["duplicate_window_days"])
        ref = reference_date or datetime.now(timezone.utc)
        target = procedure_code.strip()

        for claim in prior_claims:
            if str(claim.get("procedure_code", "")).strip() != target:
                continue
            sd = claim.get("service_date")
            if not sd:
                continue
            try:
                claim_dt = datetime.fromisoformat(str(sd)).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if abs((ref - claim_dt).total_seconds()) / 86400.0 <= window:
                return True
        return False

    def is_lab_abnormal(
        self, lab_values: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, List[AbnormalLabFinding]]:
        if not lab_values:
            return False, []
        rules = get_rules_config()
        ranges = rules["lab_reference_ranges"]
        band = float(rules["lab_tolerance_band"])
        findings: List[AbnormalLabFinding] = []

        for raw_key, raw_val in lab_values.items():
            key = str(raw_key).lower().strip()
            rng = ranges.get(key)
            if not rng:
                continue
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                continue

            low_allowed = 0.0 if rng["min"] == 0 else rng["min"] * (1.0 - band)
            high_allowed = rng["max"] * (1.0 + band)
            if val < low_allowed or val > high_allowed:
                sev = (
                    "critical"
                    if (val > high_allowed * 1.3 or val < low_allowed * 0.7)
                    else "abnormal"
                )
                findings.append(
                    AbnormalLabFinding(
                        test_name=key,
                        value=val,
                        reference_min=float(rng["min"]),
                        reference_max=float(rng["max"]),
                        severity=sev,
                    )
                )
        return len(findings) > 0, findings

    def assess_with_mock(
        self,
        patient_id: str,
        diagnosis_code: str,
        procedure_code: str,
        provider_npi: str,
        lab_values: Optional[Dict[str, Any]] = None,
        clinical_justification: Optional[str] = None,
    ) -> RiskAssessment:
        rules = get_rules_config()
        fraud_indicators: List[str] = []

        crosswalk_match = self.is_crosswalk_match(diagnosis_code, procedure_code)
        if not crosswalk_match:
            fraud_indicators.append(
                f"Diagnosis-Procedure Crosswalk Mismatch: ICD-10 '{diagnosis_code}' is not aligned with CPT '{procedure_code}'."
            )

        prior_claims: List[Dict[str, Any]] = []
        try:
            code, payload = self._get_json(f"{self.base_url}/prior-claims/{patient_id}")
            if code == 200:
                prior_claims = payload.get("claims", [])
        except Exception:
            pass

        is_dup = self.is_duplicate_request(prior_claims, procedure_code)
        if is_dup:
            fraud_indicators.append(
                f"Duplicate PA / Claim Request: CPT '{procedure_code}' submitted within {rules['duplicate_window_days']} days."
            )

        provider_verified = True
        provider_sanctioned = False
        try:
            code, prov = self._get_json(f"{self.base_url}/provider-verification/{provider_npi}")
            if code == 200:
                provider_verified = bool(prov.get("verified"))
                provider_sanctioned = bool(prov.get("sanctioned"))
        except Exception:
            pass

        if not provider_verified:
            fraud_indicators.append(
                f"Unverified Ordering Provider: NPI '{provider_npi}' could not be verified."
            )
        if provider_sanctioned:
            fraud_indicators.append(
                f"Sanctioned/Excluded Provider: NPI '{provider_npi}' is flagged as sanctioned."
            )

        abnormal, lab_findings = self.is_lab_abnormal(lab_values)
        if abnormal:
            details = ", ".join(
                f"{f.test_name}={f.value} (ref {f.reference_min}-{f.reference_max})"
                for f in lab_findings
            )
            fraud_indicators.append(f"Abnormal Lab Values Outside Tolerance Band: {details}")

        score = 0.08
        if not crosswalk_match:
            score += 0.55
        if is_dup:
            score += 0.25
        if not provider_verified:
            score += 0.20
        if provider_sanctioned:
            score += 0.35
        if abnormal:
            score += 0.35
        score = round(min(1.0, score), 2)

        if not crosswalk_match or score > float(rules["risk_tiers"]["medium"]["max_score"]):
            risk_level = "high"
        elif score > float(rules["risk_tiers"]["low"]["max_score"]):
            risk_level = "medium"
        else:
            risk_level = "low"

        requires_peer = (
            not crosswalk_match
            or risk_level == "high"
            or abnormal
            or provider_sanctioned
            or len(fraud_indicators) > 0
        )

        reasoning = (
            f"Crosswalk matched ({diagnosis_code} -> {procedure_code}), provider NPI {provider_npi} verified, "
            f"no duplicate claims within {rules['duplicate_window_days']} days, and no abnormal lab flags."
            if not fraud_indicators
            else f"Elevated integrity findings: {' '.join(fraud_indicators)}"
        )

        if has_gemini_api_key():
            try:
                persona = get_persona_config()["risk_agent"]
                prompt = (
                    "Summarize the risk and fraud assessment objectively based ONLY on these facts.\n"
                    'Return JSON: {"reasoning": string}\n'
                    f"Patient: {patient_id}, ICD-10: {diagnosis_code}, CPT: {procedure_code}, NPI: {provider_npi}\n"
                    f"Crosswalk: {crosswalk_match}, Duplicate: {is_dup}, Verified: {provider_verified}, "
                    f"Sanctioned: {provider_sanctioned}, Fraud Indicators: {json.dumps(fraud_indicators)}, "
                    f"Risk: {risk_level} ({score}), Justification: {clinical_justification or 'N/A'}"
                )
                llm_res = generate_json_with_config(
                    "risk", prompt, f"{persona['persona']} {persona['instructions']}"
                )
                if str(llm_res.get("reasoning") or "").strip():
                    reasoning = str(llm_res["reasoning"]).strip()
            except Exception:
                pass

        return RiskAssessment(
            risk_level=risk_level,
            risk_score=score,
            crosswalk_match=crosswalk_match,
            is_duplicate=is_dup,
            provider_verified=provider_verified,
            provider_sanctioned=provider_sanctioned,
            abnormal_labs=lab_findings,
            fraud_indicators=fraud_indicators,
            requires_peer_review=requires_peer,
            reasoning=reasoning,
        )

    def assess(self, patient_case: PatientCase) -> RiskAssessment:
        return self.assess_with_mock(
            patient_case.patient_id or "UNKNOWN",
            patient_case.diagnosis_code or "",
            patient_case.procedure_code or "",
            patient_case.provider_npi or "",
            patient_case.lab_values,
            patient_case.clinical_justification,
        )


def self_test() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from server.mockPayer.router import router as payer_router

    app = FastAPI()
    app.include_router(payer_router)
    client = TestClient(app)

    agent = RiskAgent("http://testserver/payer", http_client=client)
    clean = agent.assess_with_mock("PAT-1001", "M54.5", "72148", "1234567890")
    assert clean.crosswalk_match is True
    assert clean.risk_level == "low"
    assert clean.requires_peer_review is False
    assert clean.fraud_indicators == []

    fraud = agent.assess_with_mock(
        "PAT-1003", "J06.9", "27447", "9999999999", {"wbc": 19.5, "glucose": 295}
    )
    assert fraud.crosswalk_match is False
    assert fraud.risk_level == "high"
    assert fraud.provider_sanctioned is True
    assert len(fraud.abnormal_labs) >= 2
    assert fraud.requires_peer_review is True
    assert len(fraud.fraud_indicators) >= 3
    print("[PASS] RiskAgent self_test")


if __name__ == "__main__":
    self_test()
