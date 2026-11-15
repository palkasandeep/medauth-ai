from __future__ import annotations

import json
from typing import Any, Dict, List
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from server.utils.config import get_rules_config, resolve_project_path

router = APIRouter(prefix="/payer", tags=["MockPayer"])


def _load_mock_json(filename: str) -> List[Dict[str, Any]]:
    rules = get_rules_config()
    payer_dir = resolve_project_path(rules["reporting"]["payer_mock_dir"])
    return json.loads((payer_dir / filename).read_text(encoding="utf-8"))


@router.get("/eligibility/{lookup_id}")
def get_eligibility(lookup_id: str):
    members = _load_mock_json("members.json")
    for m in members:
        if m.get("member_id") == lookup_id or m.get("patient_id") == lookup_id:
            return m
    return JSONResponse(
        status_code=404,
        content={
            "eligible": False,
            "member_id": lookup_id,
            "status": "NOT_FOUND",
            "coverage_percentage": 0,
            "plan_name": "Unknown",
            "reason": f"No MedCore member record found for identifier '{lookup_id}'.",
        },
    )


@router.get("/prior-claims/{lookup_id}")
def get_prior_claims(lookup_id: str):
    claims = _load_mock_json("prior_claims.json")
    matching = [
        c
        for c in claims
        if c.get("patient_id") == lookup_id or c.get("member_id") == lookup_id
    ]
    return {"id": lookup_id, "claims": matching}


@router.get("/provider-verification/{npi}")
def verify_provider(npi: str):
    providers = _load_mock_json("providers.json")
    for p in providers:
        if p.get("npi") == npi:
            return p
    return {
        "npi": npi,
        "provider_name": "Unknown Provider",
        "verified": False,
        "sanctioned": False,
        "license_status": "UNVERIFIED",
        "network_status": "OUT_OF_NETWORK",
    }


@router.get("/procedure-cost/{code}")
def get_procedure_cost(code: str):
    costs = _load_mock_json("procedure_costs.json")
    for c in costs:
        if c.get("procedure_code") == code:
            return c
    return JSONResponse(
        status_code=404,
        content={
            "procedure_code": code,
            "description": "Unlisted Procedure",
            "allowed_amount": 0,
        },
    )
