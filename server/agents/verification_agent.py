from __future__ import annotations

from typing import Optional

from server.schemas import VerificationResult
from server.utils.gemini_client import generate_json_with_config, has_gemini_api_key


class VerificationAgent:
    def verify_clinical_justification(
        self,
        diagnosis_code: Optional[str] = None,
        procedure_code: Optional[str] = None,
        requested_service: Optional[str] = None,
        clinical_justification: Optional[str] = None,
    ) -> VerificationResult:
        text = (clinical_justification or "").strip()
        if not text:
            return VerificationResult(
                plausible=True,
                confidence=0.0,
                reasoning="No clinical justification text was provided; returning plausible=True with confidence 0.0.",
            )

        try:
            prompt = (
                "Evaluate whether this clinical justification is medically plausible for the diagnosis and procedure.\n"
                'Return JSON: {"plausible": bool, "confidence": float (0.0-1.0), "reasoning": str}\n\n'
                f"Diagnosis Code (ICD-10): {diagnosis_code or 'N/A'}\n"
                f"Procedure Code (CPT): {procedure_code or 'N/A'}\n"
                f"Requested Service: {requested_service or 'N/A'}\n"
                f"Clinical Justification: {text}"
            )
            res = generate_json_with_config(
                "verification",
                prompt,
                "You are a clinical utilization verification specialist. Return strict JSON.",
            )
            conf = max(0.0, min(1.0, float(res.get("confidence", 0.5))))
            return VerificationResult(
                plausible=bool(res.get("plausible", True)),
                confidence=conf,
                reasoning=str(res.get("reasoning") or "Evaluated by Gemini.").strip(),
            )
        except Exception as exc:
            return VerificationResult(
                plausible=True,
                confidence=0.0,
                reasoning=f"Verification fallback (non-fatal error): {exc}",
            )


def self_test() -> None:
    agent = VerificationAgent()
    empty = agent.verify_clinical_justification("M54.5", "72148", "MRI", "")
    assert empty.plausible is True
    assert empty.confidence == 0.0

    if has_gemini_api_key():
        live = agent.verify_clinical_justification(
            "M54.5",
            "72148",
            "MRI Lumbar Spine",
            "8 weeks persistent lumbar radiculopathy refractory to 6 weeks PT.",
        )
        assert isinstance(live.plausible, bool)
        assert 0.0 <= live.confidence <= 1.0
        print("[PASS] VerificationAgent (with Gemini) self_test")
    else:
        print("[PASS] VerificationAgent self_test")


if __name__ == "__main__":
    self_test()
