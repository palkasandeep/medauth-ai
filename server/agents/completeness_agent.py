from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from server.schemas import PatientCase, ValidDecision
from server.utils.config import get_rules_config


class CompletenessResult(BaseModel):
    is_complete: bool
    missing_docs: List[str]
    missing_fields: List[str]
    field_discrepancies: List[Dict[str, Any]]
    recommended_decision: Optional[ValidDecision] = None
    reason: str


class CompletenessAgent:
    def check(self, patient_case: PatientCase) -> CompletenessResult:
        rules = get_rules_config()
        present = set(patient_case.document_types_present)

        missing_docs = [d for d in rules["mandatory_docs"] if d not in present]
        missing_fields = [
            f
            for f in rules["required_case_fields"]
            if not getattr(patient_case, f, None)
            or str(getattr(patient_case, f)).strip() == ""
        ]

        discrepancies: List[Dict[str, Any]] = []
        for field in rules["required_case_fields"]:
            vals = {
                str(getattr(doc, field)).strip().upper()
                for doc in patient_case.documents
                if getattr(doc, field, None) and str(getattr(doc, field)).strip()
            }
            if len(vals) > 1:
                discrepancies.append({"field": field, "values": sorted(vals)})

        is_complete = len(missing_docs) == 0 and len(missing_fields) == 0
        reasons: List[str] = []
        if missing_docs:
            reasons.append(f"Missing mandatory documents: {', '.join(missing_docs)}")
        if missing_fields:
            reasons.append(f"Missing required fields: {', '.join(missing_fields)}")
        if discrepancies:
            desc = "; ".join(
                f"{d['field']} ({' vs '.join(d['values'])})" for d in discrepancies
            )
            reasons.append(f"Cross-document field discrepancies detected: {desc}")

        return CompletenessResult(
            is_complete=is_complete,
            missing_docs=missing_docs,
            missing_fields=missing_fields,
            field_discrepancies=discrepancies,
            recommended_decision=None if is_complete else "PENDED_ADDITIONAL_INFO",
            reason=". ".join(reasons)
            if reasons
            else "All 8 mandatory document types and 5 required case fields verified.",
        )


def self_test() -> None:
    agent = CompletenessAgent()
    incomplete = PatientCase(
        case_id="C1",
        patient_id="PAT-1004",
        member_id="MEM-MC-9004",
        diagnosis_code="I50.9",
        procedure_code="93306",
        requested_service="Echocardiogram",
        document_types_present=[
            "pa_request_form",
            "clinical_notes",
            "physician_order",
            "insurance_card",
            "referral_letter",
        ],
    )
    res1 = agent.check(incomplete)
    assert res1.is_complete is False
    assert len(res1.missing_docs) == 3
    assert res1.recommended_decision == "PENDED_ADDITIONAL_INFO"

    complete = incomplete.model_copy(
        update={"document_types_present": get_rules_config()["mandatory_docs"]}
    )
    res2 = agent.check(complete)
    assert res2.is_complete is True
    assert res2.recommended_decision is None
    print("[PASS] CompletenessAgent self_test")


if __name__ == "__main__":
    self_test()
