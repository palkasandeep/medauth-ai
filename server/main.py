from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.agents.rag.indexing_agent import IndexingAgent
from server.mockPayer.router import router as mock_payer_router
from server.schemas import Report
from server.utils.config import ROOT_DIR, get_rules_config, resolve_project_path
from server.utils.gemini_client import has_gemini_api_key
from server.workflows.medauth_workflow import (
    resume_case_human_review,
    run_case,
    run_case_chat,
)

load_dotenv()

PORT = int(os.environ.get("PORT", "4000"))


def _ensure_dirs() -> None:
    rep = get_rules_config()["reporting"]
    for rel in [rep["incoming_dir"], rep["json_dir"], rep["html_dir"]]:
        resolve_project_path(rel).mkdir(parents=True, exist_ok=True)
    resolve_project_path(rep["audit_log_path"]).parent.mkdir(parents=True, exist_ok=True)


def _warm_index_saved_reports() -> None:
    json_dir = resolve_project_path(get_rules_config()["reporting"]["json_dir"])
    if not json_dir.exists():
        return
    indexer = IndexingAgent(
        None if has_gemini_api_key() else (lambda txt: [float(len(txt) % 17), 1.0, 0.25])
    )
    for f in sorted(json_dir.glob("*.json")):
        try:
            rpt = Report.model_validate_json(f.read_text(encoding="utf-8"))
            indexer.index_report(rpt)
        except Exception:
            continue


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _ensure_dirs()
    _warm_index_saved_reports()
    yield


app = FastAPI(title="MedAuth AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mock_payer_router)


class RunCaseRequest(BaseModel):
    sampleId: Optional[str] = None
    filePaths: Optional[List[str]] = None


class ReviewRequest(BaseModel):
    reviewerDecision: str
    comments: str = ""


class ChatRequest(BaseModel):
    patientId: str = ""
    query: str = ""
    history: List[Dict[str, str]] = []


@app.get("/api/health")
def health():
    rules = get_rules_config()
    return {
        "status": "ok",
        "service": "MedAuth AI",
        "insurer": rules["insurer_name"],
        "geminiConfigured": has_gemini_api_key(),
    }


@app.get("/api/samples")
def list_samples():
    samples_dir = resolve_project_path(get_rules_config()["reporting"]["sample_bundles_dir"])
    if not samples_dir.exists():
        return {"samples": []}
    samples = []
    for d in sorted(samples_dir.iterdir()):
        if not d.is_dir():
            continue
        files = [f.name for f in sorted(d.iterdir()) if f.is_file() and not f.name.startswith(".")]
        samples.append({"id": d.name, "name": d.name, "fileCount": len(files), "files": files})
    return {"samples": samples}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    incoming_dir = resolve_project_path(get_rules_config()["reporting"]["incoming_dir"])
    incoming_dir.mkdir(parents=True, exist_ok=True)
    uploaded = []
    for f in files:
        dest_name = f"{int(time.time() * 1000)}-{Path(f.filename or 'doc.txt').name}"
        dest = incoming_dir / dest_name
        content = await f.read()
        dest.write_bytes(content)
        uploaded.append({"filename": dest_name, "originalName": f.filename, "path": str(dest)})
    return {"uploaded": uploaded}


@app.get("/api/reports")
def list_reports():
    json_dir = resolve_project_path(get_rules_config()["reporting"]["json_dir"])
    if not json_dir.exists():
        return {"reports": []}
    reports = []
    for f in sorted(json_dir.glob("*.json"), reverse=True):
        try:
            reports.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return {"reports": reports}


@app.get("/api/audit-log")
def read_audit_log():
    log_path = resolve_project_path(get_rules_config()["reporting"]["audit_log_path"])
    if not log_path.exists():
        return {"entries": []}
    lines = [l.strip() for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"entries": list(reversed(lines[-200:]))}


@app.post("/api/cases/run")
def api_run_case(req: RunCaseRequest):
    resolved_files: List[str] = []
    if req.filePaths:
        resolved_files = req.filePaths
    elif req.sampleId:
        sample_dir = (
            resolve_project_path(get_rules_config()["reporting"]["sample_bundles_dir"])
            / Path(req.sampleId).name
        )
        if not sample_dir.exists():
            raise HTTPException(status_code=404, detail=f"Sample bundle '{req.sampleId}' not found.")
        resolved_files = [
            str(f) for f in sorted(sample_dir.iterdir()) if f.is_file() and not f.name.startswith(".")
        ]
    else:
        raise HTTPException(status_code=400, detail="Provide either sampleId or filePaths.")

    return run_case(
        resolved_files,
        mock_payer_base_url=f"http://127.0.0.1:{PORT}/payer",
    )


@app.post("/api/cases/{thread_id}/review")
def api_submit_review(thread_id: str, req: ReviewRequest):
    if not req.reviewerDecision:
        raise HTTPException(status_code=400, detail="reviewerDecision is required.")
    return resume_case_human_review(
        thread_id=thread_id,
        reviewer_decision=req.reviewerDecision,
        comments=req.comments,
        mock_payer_base_url=f"http://127.0.0.1:{PORT}/payer",
    )


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    state = run_case_chat(
        patient_id=req.patientId,
        query=req.query,
        history=req.history,
    )
    return {
        "chat_answer": state.get("chat_answer", ""),
        "citations": state.get("chat_citations", []),
        "tools_used": state.get("chat_tools_used", []),
        "chat_reflection": state.get("chat_reflection"),
    }


client_dist = ROOT_DIR / "client" / "dist"
if client_dist.exists():
    assets_dir = client_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("payer/"):
            raise HTTPException(status_code=404, detail="Not found")
        target = client_dist / full_path
        if full_path and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(client_dist / "index.html"))
