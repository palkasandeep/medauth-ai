from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from server.schemas import Report
from server.utils.config import get_models_config, get_rules_config, resolve_project_path


@dataclass
class IndexedChunk:
    chunk_id: str
    collection: str
    patient_id: str
    report_id: str
    text: str
    embedding: List[float]


class InMemoryCaseVectorStore:
    def __init__(self) -> None:
        self._chunks_by_collection: Dict[str, List[IndexedChunk]] = {}
        self._reports_by_patient: Dict[str, Report] = {}

    @property
    def collection_name(self) -> str:
        return get_models_config()["rag"]["collection_name"]

    def set_chunks_for_report(
        self, patient_id: str, report_id: str, chunks: List[IndexedChunk]
    ) -> None:
        col = self.collection_name
        existing = self._chunks_by_collection.get(col, [])
        filtered = [
            c
            for c in existing
            if not (c.patient_id == patient_id and c.report_id == report_id)
        ]
        self._chunks_by_collection[col] = filtered + chunks

    def get_chunks_for_patient(self, patient_id: str) -> List[IndexedChunk]:
        all_chunks = self._chunks_by_collection.get(self.collection_name, [])
        return [c for c in all_chunks if c.patient_id == patient_id]

    def save_report(self, report: Report) -> None:
        self._reports_by_patient[report.patient_id] = report

    def get_report(self, patient_id: str) -> Optional[Report]:
        if patient_id in self._reports_by_patient:
            return self._reports_by_patient[patient_id]

        json_dir = resolve_project_path(get_rules_config()["reporting"]["json_dir"])
        if not json_dir.exists():
            return None

        for f in sorted(json_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                rpt = Report.model_validate(data)
                if rpt.patient_id == patient_id:
                    self._reports_by_patient[patient_id] = rpt
                    return rpt
            except Exception:
                continue
        return None

    def clear(self) -> None:
        self._chunks_by_collection.clear()
        self._reports_by_patient.clear()


case_vector_store = InMemoryCaseVectorStore()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    mag_a = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    mag_b = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    denom = mag_a * mag_b
    return 0.0 if denom == 0 else dot / denom
