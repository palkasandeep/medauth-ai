from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

MANDATORY_DOC_TYPES = (
    "pa_request_form",
    "clinical_notes",
    "physician_order",
    "insurance_card",
    "lab_results",
    "imaging_report",
    "referral_letter",
    "photo_id",
)

VALID_DECISIONS = (
    "APPROVED",
    "PARTIALLY_APPROVED",
    "PENDED_ADDITIONAL_INFO",
    "REQUIRES_PEER_REVIEW",
    "DENIED",
)

MandatoryDocType = Literal[
    "pa_request_form",
    "clinical_notes",
    "physician_order",
    "insurance_card",
    "lab_results",
    "imaging_report",
    "referral_letter",
    "photo_id",
]

ValidDecision = Literal[
    "APPROVED",
    "PARTIALLY_APPROVED",
    "PENDED_ADDITIONAL_INFO",
    "REQUIRES_PEER_REVIEW",
    "DENIED",
]


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    file_ext: str
    doc_type: Union[MandatoryDocType, Literal["unknown"]]
    patient_id: Optional[str] = None
    member_id: Optional[str] = None
    patient_name: Optional[str] = None
    diagnosis_code: Optional[str] = None
    procedure_code: Optional[str] = None
    requested_service: Optional[str] = None
    provider_npi: Optional[str] = None
    ordering_physician: Optional[str] = None
    urgency: Optional[str] = None
    clinical_justification: Optional[str] = None
    plan_name: Optional[str] = None
    lab_values: Optional[Dict[str, Union[float, int, str]]] = None
    imaging_findings: Optional[str] = None
    referral_details: Optional[str] = None
    source_language: str = "en"
    was_translated: bool = False
    raw_text: str = ""
    translated_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PatientCase(BaseModel):
    case_id: str
    patient_id: Optional[str] = None
    member_id: Optional[str] = None
    patient_name: Optional[str] = None
    diagnosis_code: Optional[str] = None
    procedure_code: Optional[str] = None
    requested_service: Optional[str] = None
    provider_npi: Optional[str] = None
    ordering_physician: Optional[str] = None
    urgency: Optional[str] = "routine"
    clinical_justification: Optional[str] = None
    plan_name: Optional[str] = None
    lab_values: Dict[str, Union[float, int, str]] = Field(default_factory=dict)
    imaging_findings: Optional[str] = None
    referral_details: Optional[str] = None
    has_translated_documents: bool = False
    translated_languages: List[str] = Field(default_factory=list)
    document_types_present: List[str] = Field(default_factory=list)
    documents: List[DocumentRecord] = Field(default_factory=list)


class EligibilityResult(BaseModel):
    eligible: bool
    eligibility_reason: str = Field(min_length=1)
    coverage_percentage: float = Field(ge=0, le=100)
    procedure_cost: float = Field(ge=0, default=0.0)
    patient_responsibility: float = Field(ge=0)
    plan_name: str = "Unknown"
    member_status: str = "UNKNOWN"


class VerificationResult(BaseModel):
    plausible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class AbnormalLabFinding(BaseModel):
    test_name: str
    value: float
    reference_min: float
    reference_max: float
    severity: str


class RiskAssessment(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    risk_score: float = Field(ge=0.0, le=1.0)
    crosswalk_match: bool
    is_duplicate: bool
    provider_verified: bool
    provider_sanctioned: bool = False
    abnormal_labs: List[AbnormalLabFinding] = Field(default_factory=list)
    fraud_indicators: List[str] = Field(default_factory=list)
    requires_peer_review: bool
    reasoning: str


class PeerReviewResult(BaseModel):
    narrative: str
    risk_level: Literal["low", "medium", "high"]
    risk_score: float = Field(ge=0.0, le=1.0)
    fraud_indicators: List[str] = Field(default_factory=list)
    requires_peer_review: bool


class RuleResult(BaseModel):
    rule_id: str
    rule_text: Optional[str] = None
    passed: bool
    reason: str


class DecisionRouting(BaseModel):
    escalate_to_human: bool
    reason: str


class Decision(BaseModel):
    decision: ValidDecision
    confidence_score: float = Field(ge=0.0, le=1.0)
    key_findings: List[str]
    rule_results: List[RuleResult]
    reasoning: str
    routing: DecisionRouting


class DecisionReflection(BaseModel):
    groundedness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    confidence_alignment: float = Field(ge=0.0, le=1.0)
    passed: bool
    feedback: str


class Report(BaseModel):
    report_id: str
    case_id: str
    patient_id: str
    patient_name: str
    requested_service: str
    generated_at: str
    eligibility: Optional[EligibilityResult] = None
    verification: Optional[VerificationResult] = None
    risk: Optional[RiskAssessment] = None
    peer_review: Optional[PeerReviewResult] = None
    rules_applied: List[RuleResult]
    decision: Decision
    decision_reflection: Optional[DecisionReflection] = None
    human_review_required: bool
    status: str
    reviewer_decision: Optional[str] = None
    reviewer_comments: Optional[str] = None
    json_path: Optional[str] = None
    html_path: Optional[str] = None


class RagTriad(BaseModel):
    answer_relevance: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    context_relevance: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    feedback: Optional[str] = None
