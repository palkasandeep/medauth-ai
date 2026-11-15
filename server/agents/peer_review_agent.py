from __future__ import annotations

import json
from typing import Optional

from server.schemas import PatientCase, PeerReviewResult, RiskAssessment
from server.utils.gemini_client import generate_json_with_config, has_gemini_api_key


class PeerReviewAgent:
    def synthesize(
        self,
        risk_assessment: RiskAssessment,
        patient_case: Optional[PatientCase] = None,
    ) -> PeerReviewResult:
        if risk_assessment.requires_peer_review:
            narrative = (
                f"Peer review is recommended ({risk_assessment.risk_level} risk, "
                f"score {risk_assessment.risk_score}). Findings: "
                f"{'; '.join(risk_assessment.fraud_indicators) or risk_assessment.reasoning}"
            )
        else:
            narrative = (
                f"Case presents low utilization risk (score {risk_assessment.risk_score}) "
                "with no fraud or integrity indicators requiring peer escalation."
            )

        if has_gemini_api_key():
            try:
                prompt = (
                    "Write a concise, plain-English clinical peer review narrative summarizing this risk assessment.\n"
                    "Do NOT alter risk level, risk score, or indicators.\n"
                    'Return JSON: {"narrative": "..."}\n'
                    f"Patient: {patient_case.patient_id if patient_case else 'N/A'} "
                    f"({patient_case.requested_service if patient_case else 'N/A'})\n"
                    f"Risk: {risk_assessment.risk_level} ({risk_assessment.risk_score}), "
                    f"Peer Review: {risk_assessment.requires_peer_review}, "
                    f"Indicators: {json.dumps(risk_assessment.fraud_indicators)}, "
                    f"Reasoning: {risk_assessment.reasoning}"
                )
                res = generate_json_with_config(
                    "peer_review",
                    prompt,
                    "You are a clinical peer-review narrator at MedCore. Summarize objectively in plain English.",
                )
                if str(res.get("narrative") or "").strip():
                    narrative = str(res["narrative"]).strip()
            except Exception:
                pass

        return PeerReviewResult(
            narrative=narrative,
            risk_level=risk_assessment.risk_level,
            risk_score=risk_assessment.risk_score,
            fraud_indicators=list(risk_assessment.fraud_indicators),
            requires_peer_review=risk_assessment.requires_peer_review,
        )


def self_test() -> None:
    agent = PeerReviewAgent()
    risk = RiskAssessment(
        risk_level="high",
        risk_score=0.91,
        crosswalk_match=False,
        is_duplicate=True,
        provider_verified=False,
        provider_sanctioned=True,
        abnormal_labs=[],
        fraud_indicators=["Crosswalk mismatch", "Sanctioned provider"],
        requires_peer_review=True,
        reasoning="High risk detected.",
    )
    out = agent.synthesize(risk)
    assert out.risk_level == risk.risk_level
    assert out.risk_score == risk.risk_score
    assert out.fraud_indicators == risk.fraud_indicators
    assert out.requires_peer_review == risk.requires_peer_review
    assert len(out.narrative) > 10
    print("[PASS] PeerReviewAgent self_test")


if __name__ == "__main__":
    self_test()
