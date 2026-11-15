from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.agents.document_agent import ParsedDocument
from server.schemas import (
    DocumentRecord,
    MANDATORY_DOC_TYPES,
    PatientCase,
)
from server.utils.config import get_rules_config
from server.utils.gemini_client import generate_json_with_config

HEADER_RULES = [
    (re.compile(r"PRIOR\s+AUTHORIZATION\s+REQUEST", re.I), "pa_request_form"),
    (re.compile(r"CLINICAL\s+NOTES?", re.I), "clinical_notes"),
    (re.compile(r"PHYSICIAN\s+ORDER", re.I), "physician_order"),
    (re.compile(r"INSURANCE\s+CARD", re.I), "insurance_card"),
    (re.compile(r"IMAGING\s+REPORT", re.I), "imaging_report"),
    (re.compile(r"REFERRAL\s+LETTER", re.I), "referral_letter"),
    (re.compile(r"PATIENT\s+PHOTO\s+ID", re.I), "photo_id"),
    (re.compile(r"LAB(?:ORATORY)?\s+RESULTS?", re.I), "lab_results"),
]

SOURCE_PRIORITY = {
    "pa_request_form": 1,
    "physician_order": 2,
    "clinical_notes": 3,
    "insurance_card": 4,
    "referral_letter": 5,
    "imaging_report": 6,
    "lab_results": 7,
    "photo_id": 8,
    "unknown": 9,
}


class InformationExtractionAgent:
    def __init__(self) -> None:
        self.supported_extensions = {
            e.lower() for e in get_rules_config()["supported_extensions"]
        }

    def _validate_extension(self, filename_or_ext: str) -> str:
        ext = (
            filename_or_ext.lower()
            if filename_or_ext.startswith(".")
            else Path(filename_or_ext).suffix.lower()
        )
        if not ext or ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: '{filename_or_ext}'")
        return ext

    def classify(
        self,
        text: str,
        filename_or_ext: str,
        parsed_json: Optional[Dict[str, Any]] = None,
    ) -> str:
        ext = self._validate_extension(filename_or_ext)
        if ext == ".json":
            obj = parsed_json if parsed_json is not None else json.loads(text)
            dt = obj.get("doc_type")
            if isinstance(dt, str) and dt in MANDATORY_DOC_TYPES:
                return dt
            return "unknown"

        for pattern, doc_type in HEADER_RULES:
            if pattern.search(text):
                return doc_type
        return "unknown"

    def extract(self, doc: ParsedDocument) -> DocumentRecord:
        ext = self._validate_extension(doc.file_ext or doc.filename)
        doc_type = self.classify(
            doc.text_for_classification,
            doc.file_ext or doc.filename,
            doc.parsed_json,
        )

        if ext == ".json" and doc.parsed_json is not None:
            labs = doc.parsed_json.get("results") or doc.parsed_json.get("lab_values") or {}
            return DocumentRecord(
                doc_id=str(uuid.uuid4()),
                filename=doc.filename,
                file_ext=ext,
                doc_type=doc_type,
                patient_id=doc.parsed_json.get("patient_id"),
                member_id=doc.parsed_json.get("member_id"),
                patient_name=doc.parsed_json.get("patient_name"),
                lab_values=labs,
                source_language=doc.source_language,
                was_translated=doc.was_translated,
                raw_text=doc.raw_text,
                metadata=doc.parsed_json,
            )

        prompt = (
            f'Extract structured prior-authorization fields from this "{doc_type}" document.\n'
            "Return JSON with keys: patient_id, member_id, patient_name, diagnosis_code, "
            "procedure_code, requested_service, provider_npi, ordering_physician, urgency, "
            "clinical_justification, plan_name, lab_values, imaging_findings, referral_details.\n"
            "Use null for missing fields. Preserve exact ICD-10, CPT, NPI, patient_id, member_id.\n\n"
            f"DOCUMENT TEXT:\n{doc.text_for_classification}"
        )
        extracted = generate_json_with_config(
            "information_extraction",
            prompt,
            "You are a precise clinical data extraction agent. Return strict JSON.",
        )

        return DocumentRecord(
            doc_id=str(uuid.uuid4()),
            filename=doc.filename,
            file_ext=ext,
            doc_type=doc_type,
            patient_id=extracted.get("patient_id"),
            member_id=extracted.get("member_id"),
            patient_name=extracted.get("patient_name"),
            diagnosis_code=extracted.get("diagnosis_code"),
            procedure_code=extracted.get("procedure_code"),
            requested_service=extracted.get("requested_service"),
            provider_npi=extracted.get("provider_npi"),
            ordering_physician=extracted.get("ordering_physician"),
            urgency=extracted.get("urgency"),
            clinical_justification=extracted.get("clinical_justification"),
            plan_name=extracted.get("plan_name"),
            lab_values=extracted.get("lab_values"),
            imaging_findings=extracted.get("imaging_findings"),
            referral_details=extracted.get("referral_details"),
            source_language=doc.source_language,
            was_translated=doc.was_translated,
            raw_text=doc.raw_text,
            translated_text=doc.text_for_classification if doc.was_translated else None,
        )

    def merge_documents(
        self, records: List[DocumentRecord], case_id: Optional[str] = None
    ) -> PatientCase:
        ordered = sorted(records, key=lambda r: SOURCE_PRIORITY.get(r.doc_type, 99))

        def pick(attr: str) -> Optional[str]:
            for rec in ordered:
                val = getattr(rec, attr, None)
                if val is not None and str(val).strip() != "":
                    return str(val).strip()
            return None

        merged_labs: Dict[str, Any] = {}
        for rec in records:
            if rec.lab_values:
                merged_labs.update(rec.lab_values)

        justifications = [
            rec.clinical_justification.strip()
            for rec in ordered
            if rec.clinical_justification and rec.clinical_justification.strip()
        ]
        imaging_parts = [
            rec.imaging_findings.strip()
            for rec in records
            if rec.imaging_findings and rec.imaging_findings.strip()
        ]
        referral_parts = [
            rec.referral_details.strip()
            for rec in records
            if rec.referral_details and rec.referral_details.strip()
        ]
        translated_langs = sorted(
            {rec.source_language for rec in records if rec.was_translated}
        )
        doc_types_present = sorted(
            {rec.doc_type for rec in records if rec.doc_type != "unknown"}
        )

        return PatientCase(
            case_id=case_id or f"CASE-{str(uuid.uuid4())[:8]}",
            patient_id=pick("patient_id"),
            member_id=pick("member_id"),
            patient_name=pick("patient_name"),
            diagnosis_code=pick("diagnosis_code"),
            procedure_code=pick("procedure_code"),
            requested_service=pick("requested_service"),
            provider_npi=pick("provider_npi"),
            ordering_physician=pick("ordering_physician"),
            urgency=pick("urgency") or "routine",
            clinical_justification=" | ".join(justifications) if justifications else None,
            plan_name=pick("plan_name"),
            lab_values=merged_labs,
            imaging_findings=" | ".join(imaging_parts) if imaging_parts else None,
            referral_details=" | ".join(referral_parts) if referral_parts else None,
            has_translated_documents=any(rec.was_translated for rec in records),
            translated_languages=translated_langs,
            document_types_present=doc_types_present,
            documents=records,
        )


def self_test() -> None:
    agent = InformationExtractionAgent()
    assert agent.classify("PRIOR AUTHORIZATION REQUEST\nPatient: Elena", "pa.txt") == "pa_request_form"
    assert agent.classify("IMAGING REPORT\nFindings: narrowing", ".txt") == "imaging_report"
    assert agent.classify('{"doc_type":"lab_results","results":{"wbc":6.8}}', "labs.json") == "lab_results"
    try:
        agent.classify("text", "bad.exe")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    merged = agent.merge_documents(
        [
            DocumentRecord(
                doc_id="1",
                filename="pa.txt",
                file_ext=".txt",
                doc_type="pa_request_form",
                patient_id="PAT-1001",
                member_id="MEM-MC-9001",
                diagnosis_code="M54.5",
                procedure_code="72148",
                requested_service="MRI Lumbar Spine",
                clinical_justification="6 weeks failed PT",
            ),
            DocumentRecord(
                doc_id="2",
                filename="labs.json",
                file_ext=".json",
                doc_type="lab_results",
                lab_values={"wbc": 6.8},
            ),
            DocumentRecord(
                doc_id="3",
                filename="ref_es.txt",
                file_ext=".txt",
                doc_type="referral_letter",
                referral_details="Referred for MRI",
                source_language="es",
                was_translated=True,
            ),
        ]
    )
    assert merged.patient_id == "PAT-1001"
    assert merged.lab_values == {"wbc": 6.8}
    assert merged.has_translated_documents is True
    assert merged.translated_languages == ["es"]
    print("[PASS] InformationExtractionAgent self_test")


if __name__ == "__main__":
    self_test()
