from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from docx import Document as DocxDocument
from PIL import Image
from pypdf import PdfReader
import pytesseract

from server.utils.config import get_rules_config
from server.utils.gemini_client import generate_json_with_config, has_gemini_api_key


@dataclass
class TranslationResult:
    text: str
    source_language: str
    was_translated: bool


@dataclass
class ParsedDocument:
    filename: str
    file_ext: str
    raw_text: str
    text_for_classification: str
    parsed_json: Optional[Dict[str, Any]]
    source_language: str
    was_translated: bool


class TranslationAgent:
    def detect_language(self, text: str) -> str:
        if not text or len(text.strip()) < 3:
            return "en"
        prompt = (
            'Detect the primary ISO 639-1 two-letter language code of this clinical document '
            '(e.g. "en", "es", "fr"). Return JSON: {"language_code": "<code>"}.\n\n'
            f"TEXT:\n{text[:1500]}"
        )
        res = generate_json_with_config(
            "translation",
            prompt,
            "You are a precise clinical language detector. Return strict JSON.",
        )
        return str(res.get("language_code", "en")).lower().strip()

    def translate(
        self, text: str, explicit_source_lang: Optional[str] = None
    ) -> TranslationResult:
        lang = (
            explicit_source_lang.lower().strip()
            if explicit_source_lang
            else self.detect_language(text)
        )
        if lang == "en" or lang.startswith("en-"):
            return TranslationResult(text=text, source_language="en", was_translated=False)

        prompt = (
            f'Translate the following clinical document from "{lang}" to English.\n'
            "CRITICAL REQUIREMENTS:\n"
            "1. Preserve EVERY ICD-10 diagnosis code (e.g. M54.5), CPT code (e.g. 72148), "
            "NPI number, patient ID, member ID, date, and numeric measurement EXACTLY.\n"
            '2. If the document is a referral letter (e.g. "CARTA DE REFERENCIA MÉDICA"), '
            'begin with the exact header "REFERRAL LETTER".\n'
            '3. Return JSON: {"translated_text": "..."}.\n\n'
            f"SOURCE TEXT:\n{text}"
        )
        res = generate_json_with_config(
            "translation",
            prompt,
            "You are a medical translator preserving clinical codes and numbers verbatim.",
        )
        translated = str(res.get("translated_text") or text).strip()
        return TranslationResult(
            text=translated or text,
            source_language=lang,
            was_translated=True,
        )


class DocumentAgent:
    def __init__(self, translation_agent: Optional[TranslationAgent] = None) -> None:
        self.translation_agent = translation_agent or TranslationAgent()
        self.supported_extensions = {
            e.lower() for e in get_rules_config()["supported_extensions"]
        }

    def parse_file(self, file_path: str | Path) -> Dict[str, Any]:
        p = Path(file_path)
        ext = p.suffix.lower()
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {ext}")

        if ext == ".txt":
            return {"raw_text": p.read_text(encoding="utf-8"), "file_ext": ext}

        if ext == ".json":
            raw = p.read_text(encoding="utf-8")
            return {
                "raw_text": raw,
                "parsed_json": json.loads(raw),
                "file_ext": ext,
            }

        if ext == ".pdf":
            reader = PdfReader(str(p))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return {"raw_text": text, "file_ext": ext}

        if ext == ".docx":
            doc = DocxDocument(str(p))
            text = "\n".join(para.text for para in doc.paragraphs)
            return {"raw_text": text, "file_ext": ext}

        if ext == ".png":
            img = Image.open(p)
            text = pytesseract.image_to_string(img)
            return {"raw_text": text, "file_ext": ext}

        raise ValueError(f"Unsupported file extension: {ext}")

    def process_file(self, file_path: str | Path) -> ParsedDocument:
        p = Path(file_path)
        parsed = self.parse_file(p)
        ext = parsed["file_ext"]

        if ext == ".json":
            return ParsedDocument(
                filename=p.name,
                file_ext=ext,
                raw_text=parsed["raw_text"],
                text_for_classification=parsed["raw_text"],
                parsed_json=parsed.get("parsed_json"),
                source_language="en",
                was_translated=False,
            )

        tr = self.translation_agent.translate(parsed["raw_text"])
        return ParsedDocument(
            filename=p.name,
            file_ext=ext,
            raw_text=parsed["raw_text"],
            text_for_classification=tr.text,
            parsed_json=None,
            source_language=tr.source_language,
            was_translated=tr.was_translated,
        )

    def process_bundle(self, file_paths: List[str | Path]) -> List[ParsedDocument]:
        return [self.process_file(fp) for fp in file_paths]


def self_test() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="medauth-doc-"))
    try:
        json_path = tmp_dir / "labs.json"
        payload = {"doc_type": "lab_results", "patient_id": "PAT-1001", "results": {"wbc": 6.5}}
        json_path.write_text(json.dumps(payload), encoding="utf-8")

        txt_path = tmp_dir / "note.txt"
        txt_path.write_text("CLINICAL NOTES\nPatient PAT-1001 stable.", encoding="utf-8")

        bad_path = tmp_dir / "bad.csv"
        bad_path.write_text("a,b,c", encoding="utf-8")

        agent = DocumentAgent()
        res_json = agent.process_file(json_path)
        assert res_json.parsed_json == payload
        assert res_json.was_translated is False

        try:
            agent.parse_file(bad_path)
            raise AssertionError("Expected ValueError for unsupported extension")
        except ValueError:
            pass

        if has_gemini_api_key():
            en_doc = agent.process_file(txt_path)
            assert en_doc.was_translated is False
            es_tr = TranslationAgent().translate(
                "CARTA DE REFERENCIA MÉDICA\nDiagnóstico M54.5, CPT 72148."
            )
            assert es_tr.was_translated is True
            assert "M54.5" in es_tr.text and "72148" in es_tr.text
            print("[PASS] DocumentAgent + TranslationAgent (with Gemini) self_test")
        else:
            parsed_txt = agent.parse_file(txt_path)
            assert "CLINICAL NOTES" in parsed_txt["raw_text"]
            print("[PASS] DocumentAgent self_test")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    self_test()
