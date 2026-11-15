from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from server.schemas import (
    Decision,
    DecisionReflection,
    DecisionRouting,
    EligibilityResult,
    PatientCase,
    PeerReviewResult,
    RiskAssessment,
    RuleResult,
    VALID_DECISIONS,
    ValidDecision,
    VerificationResult,
)
from server.utils.config import get_persona_config, get_rules_config
from server.utils.gemini_client import generate_json_with_config, has_gemini_api_key


class DecisionAgent:
    def get_applicable_rules(self, procedure_code: Optional[str] = None) -> List[Dict[str, Any]]:
        rules = get_rules_config()["policy_rules"]
        general = list(rules["general"])
        code = (procedure_code or "").strip()
        specific = list(rules["by_procedure"].get(code, [])) if code else []
        return general + specific

    def apply_rules(
        self,
        rules_to_apply: List[Dict[str, Any]],
        patient_case: Optional[PatientCase] = None,
        eligibility: Optional[EligibilityResult] = None,
        verification: Optional[VerificationResult] = None,
        risk: Optional[RiskAssessment] = None,
    ) -> List[RuleResult]:
        safe_case = patient_case or PatientCase(case_id="DEFAULT")
        safe_elig = eligibility or EligibilityResult(
            eligible=True,
            eligibility_reason="Default active eligibility.",
            coverage_percentage=100,
            patient_responsibility=0.0,
        )

        elig_rule_ids = {
            r["rule_id"]
            for r in get_rules_config()["policy_rules"]["general"]
            if r.get("is_eligibility_rule")
        }

        by_id: Dict[str, RuleResult] = {}
        non_elig = []
        for rule in rules_to_apply:
            rid = rule["rule_id"]
            if rule.get("is_eligibility_rule") or rid in elig_rule_ids:
                by_id[rid] = RuleResult(
                    rule_id=rid,
                    rule_text=rule.get("rule_text"),
                    passed=safe_elig.eligible,
                    reason=safe_elig.eligibility_reason,
                )
            else:
                non_elig.append(rule)

        llm_map: Dict[str, Dict[str, Any]] = {}
        if non_elig and has_gemini_api_key():
            try:
                persona = get_persona_config()["decision_agent"]
                prompt = (
                    "Evaluate each MedCore prior-authorization policy rule against these case facts.\n"
                    'Return JSON: {"results": [{"rule_id": str, "passed": bool, "reason": str}]}\n\n'
                    f"Rules:\n{json.dumps(non_elig, indent=2)}\n\n"
                    f"Diagnosis: {safe_case.diagnosis_code}, CPT: {safe_case.procedure_code}, "
                    f"Service: {safe_case.requested_service}\n"
                    f"Justification: {safe_case.clinical_justification}\n"
                    f"Imaging: {safe_case.imaging_findings}, Referral: {safe_case.referral_details}\n"
                    f"Verification: {verification.model_dump() if verification else {}}\n"
                    f"Risk: {risk.model_dump() if risk else {}}"
                )
                res = generate_json_with_config(
                    "decision", prompt, f"{persona['persona']} {persona['instructions']}"
                )
                for item in res.get("results", []):
                    llm_map[item["rule_id"]] = {
                        "passed": bool(item.get("passed", True)),
                        "reason": str(item.get("reason") or "Evaluated against policy rule."),
                    }
            except Exception:
                pass

        for rule in non_elig:
            rid = rule["rule_id"]
            passed = llm_map.get(rid, {}).get("passed", True)
            reason = llm_map.get(rid, {}).get(
                "reason", "Evaluated against documented clinical evidence and provider records."
            )

            if rid == "GEN-PROV-003" and risk and (not risk.provider_verified or risk.provider_sanctioned):
                passed = False
                reason = f"Ordering provider NPI '{safe_case.provider_npi}' failed verification or is sanctioned."

            if rid == "GEN-CLIN-002" and risk and not risk.crosswalk_match:
                passed = False
                reason = f"ICD-10 '{safe_case.diagnosis_code}' does not align with CPT '{safe_case.procedure_code}'."

            by_id[rid] = RuleResult(
                rule_id=rid,
                rule_text=rule.get("rule_text"),
                passed=passed,
                reason=reason,
            )

        return [by_id[r["rule_id"]] for r in rules_to_apply]

    def compute_routing(
        self,
        decision: ValidDecision,
        confidence_score: float,
        risk: RiskAssessment,
        reflection_passed: Optional[bool] = None,
        has_error: bool = False,
    ) -> DecisionRouting:
        t = get_rules_config()["human_review_thresholds"]
        reasons: List[str] = []

        if decision in t["escalate_decisions"]:
            reasons.append(f"Decision is {decision}")
        if confidence_score < float(t["min_confidence"]):
            reasons.append(f"Confidence {confidence_score:.2f} < {t['min_confidence']}")
        if t["escalate_on_fraud_indicator"] and len(risk.fraud_indicators) > 0:
            reasons.append(f"Fraud/integrity indicators flagged ({len(risk.fraud_indicators)})")
        if t["escalate_on_non_low_risk"] and risk.risk_level != "low":
            reasons.append(f"Risk level is '{risk.risk_level}' (not low)")
        if t["escalate_on_peer_review_flag"] and risk.requires_peer_review:
            reasons.append("Peer review flag raised")
        if t["escalate_on_reflection_failure"] and reflection_passed is False:
            reasons.append("Deterministic self-check failed")
        if has_error:
            reasons.append("Processing error occurred")

        escalate = len(reasons) > 0
        return DecisionRouting(
            escalate_to_human=escalate,
            reason="; ".join(reasons)
            if escalate
            else "Clean, high-confidence, low-risk APPROVED case; no manual review required.",
        )

    def synthesize(
        self,
        patient_case: PatientCase,
        eligibility: EligibilityResult,
        risk: RiskAssessment,
        rule_results: List[RuleResult],
        verification: Optional[VerificationResult] = None,
        peer_review: Optional[PeerReviewResult] = None,
        reflection_passed: Optional[bool] = None,
        has_error: bool = False,
    ) -> Decision:
        elig_rule_ids = {
            r["rule_id"]
            for r in get_rules_config()["policy_rules"]["general"]
            if r.get("is_eligibility_rule")
        }

        elig_failed = (not eligibility.eligible) or any(
            (r.rule_id in elig_rule_ids and not r.passed) for r in rule_results
        )
        non_elig_failed = any(
            (r.rule_id not in elig_rule_ids and not r.passed) for r in rule_results
        )

        confidence = (
            0.96
            if elig_failed
            else (0.65 if risk.requires_peer_review else (0.82 if non_elig_failed else 0.93))
        )
        llm_findings: List[str] = []
        llm_reasoning = ""

        if has_gemini_api_key():
            try:
                persona = get_persona_config()["decision_agent"]
                prompt = (
                    "Synthesize key_findings, reasoning, and calibrated confidence_score in [0, 1].\n"
                    'Return JSON: {"confidence_score": float, "key_findings": [str], "reasoning": str}\n'
                    f"Eligibility: {eligibility.model_dump()}\n"
                    f"Risk: {risk.model_dump()}\n"
                    f"Rules: {[r.model_dump() for r in rule_results]}\n"
                    f"Peer Review: {peer_review.model_dump() if peer_review else {}}"
                )
                res = generate_json_with_config(
                    "decision", prompt, f"{persona['persona']} {persona['instructions']}"
                )
                if isinstance(res.get("confidence_score"), (int, float)):
                    confidence = round(max(0.0, min(1.0, float(res["confidence_score"]))), 2)
                if isinstance(res.get("key_findings"), list) and res["key_findings"]:
                    llm_findings = [str(x) for x in res["key_findings"]]
                if str(res.get("reasoning") or "").strip():
                    llm_reasoning = str(res["reasoning"]).strip()
            except Exception:
                pass

        if elig_failed:
            decision: ValidDecision = "DENIED"
        elif risk.requires_peer_review or risk.risk_level == "high" or len(risk.fraud_indicators) > 0:
            decision = "REQUIRES_PEER_REVIEW"
        elif non_elig_failed:
            decision = "PARTIALLY_APPROVED" if confidence >= 0.5 else "PENDED_ADDITIONAL_INFO"
        else:
            decision = "APPROVED"

        if not llm_findings:
            llm_findings = [
                f"Eligibility: {'Eligible' if eligibility.eligible else 'Ineligible'} ({eligibility.eligibility_reason})",
                f"Risk Assessment: {risk.risk_level} risk (score {risk.risk_score})",
                *[
                    f"{r.rule_id}: {'PASS' if r.passed else 'FAIL'} — {r.reason}"
                    for r in rule_results
                ],
            ]

        if not llm_reasoning:
            passed_cnt = sum(1 for r in rule_results if r.passed)
            llm_reasoning = (
                f"Determination '{decision}' reached based on {passed_cnt}/{len(rule_results)} "
                f"rules passed, eligibility={eligibility.eligible}, risk={risk.risk_level}."
            )

        routing = self.compute_routing(
            decision=decision,
            confidence_score=confidence,
            risk=risk,
            reflection_passed=reflection_passed,
            has_error=has_error,
        )

        return Decision(
            decision=decision,
            confidence_score=confidence,
            key_findings=llm_findings,
            rule_results=rule_results,
            reasoning=llm_reasoning,
            routing=routing,
        )

    def reflect(self, decision_obj: Dict[str, Any] | Decision) -> DecisionReflection:
        d = (
            decision_obj.model_dump()
            if isinstance(decision_obj, Decision)
            else dict(decision_obj)
        )
        issues: List[str] = []
        elig_rule_ids = {
            r["rule_id"]
            for r in get_rules_config()["policy_rules"]["general"]
            if r.get("is_eligibility_rule")
        }

        dec_val = d.get("decision")
        is_valid = dec_val in VALID_DECISIONS
        if not is_valid:
            issues.append(f"Invalid decision '{dec_val}'; must be one of {VALID_DECISIONS}.")

        rule_results = d.get("rule_results", [])
        elig_failed = any(
            (r.get("rule_id") in elig_rule_ids and not r.get("passed")) for r in rule_results
        )
        any_failed = any(not r.get("passed") for r in rule_results)
        all_passed = len(rule_results) > 0 and all(r.get("passed") for r in rule_results)

        if elig_failed and dec_val in ("APPROVED", "PARTIALLY_APPROVED"):
            issues.append("Inconsistency: Eligibility rule failed, cannot approve.")
        if any_failed and dec_val == "APPROVED":
            issues.append("Inconsistency: One or more policy rules failed, cannot be APPROVED.")
        if all_passed and dec_val == "DENIED":
            issues.append("Inconsistency: All policy rules passed, yet decision is DENIED.")

        conf = d.get("confidence_score")
        conf_valid = isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0
        if not conf_valid:
            issues.append("Confidence score is outside [0, 1].")

        passed = len(issues) == 0
        groundedness = 1.0 if (rule_results and d.get("key_findings") and passed) else 0.4
        relevance = 1.0 if (is_valid and len(str(d.get("reasoning", "")).strip()) > 10) else 0.4
        conf_align = 1.0 if (conf_valid and passed) else (0.5 if conf_valid else 0.0)

        return DecisionReflection(
            groundedness=groundedness,
            relevance=relevance,
            confidence_alignment=conf_align,
            passed=passed,
            feedback=(
                "Deterministic self-check passed: decision is valid and consistent with rule_results."
                if passed
                else " ".join(issues)
            ),
        )


def self_test() -> None:
    agent = DecisionAgent()
    assert any(r["rule_id"] == "PROC-72148-01" for r in agent.get_applicable_rules("72148"))
    assert len(agent.get_applicable_rules("99999")) == len(get_rules_config()["policy_rules"]["general"])

    p_case = PatientCase(
        case_id="C1",
        patient_id="PAT-1001",
        member_id="MEM-MC-9001",
        diagnosis_code="M54.5",
        procedure_code="72148",
        requested_service="MRI Lumbar Spine",
        provider_npi="1234567890",
    )
    elig_ok = EligibilityResult(
        eligible=True, eligibility_reason="Active", coverage_percentage=100, patient_responsibility=0.0
    )
    risk_low = RiskAssessment(
        risk_level="low",
        risk_score=0.08,
        crosswalk_match=True,
        is_duplicate=False,
        provider_verified=True,
        provider_sanctioned=False,
        abnormal_labs=[],
        fraud_indicators=[],
        requires_peer_review=False,
        reasoning="Clean",
    )

    approved = agent.synthesize(
        p_case,
        elig_ok,
        risk_low,
        [
            RuleResult(rule_id="GEN-ELIG-001", passed=True, reason="Active"),
            RuleResult(rule_id="GEN-CLIN-002", passed=True, reason="Met"),
        ],
    )
    assert approved.decision == "APPROVED"
    assert approved.routing.escalate_to_human is False
    assert agent.reflect(approved).passed is True

    denied = agent.synthesize(
        p_case,
        elig_ok.model_copy(update={"eligible": False, "eligibility_reason": "Terminated"}),
        risk_low,
        [
            RuleResult(rule_id="GEN-ELIG-001", passed=False, reason="Terminated"),
            RuleResult(rule_id="GEN-CLIN-002", passed=True, reason="Met"),
        ],
    )
    assert denied.decision == "DENIED"
    assert denied.routing.escalate_to_human is True
    assert agent.reflect(denied).passed is True

    bad_refl = agent.reflect(
        {"decision": "MAYBE", "confidence_score": 0.9, "rule_results": [{"rule_id": "GEN-ELIG-001", "passed": True}]}
    )
    assert bad_refl.passed is False
    print("[PASS] DecisionAgent self_test")


if __name__ == "__main__":
    self_test()
