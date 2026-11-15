from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
SERVER_DIR = ROOT_DIR / "server"
CONFIGS_DIR = SERVER_DIR / "configs"


def _read_json(filename: str) -> Dict[str, Any]:
    path = CONFIGS_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def get_models_config() -> Dict[str, Any]:
    return _read_json("models.json")


def get_rules_config() -> Dict[str, Any]:
    return _read_json("rules.json")


def get_persona_config() -> Dict[str, Any]:
    return _read_json("persona.json")


def get_workflow_dev_config() -> Dict[str, Any]:
    return _read_json("workflow_dev.json")


def resolve_agent_model(agent_key: str) -> Tuple[str, float]:
    cfg = get_models_config()
    agent_cfg = cfg["agents"].get(
        agent_key, {"model_tier": "reasoning", "temperature": 0.1}
    )
    model_name = cfg["models"][agent_cfg["model_tier"]]
    temperature = float(agent_cfg["temperature"])
    return model_name, temperature


def resolve_project_path(rel_path: str) -> Path:
    p = Path(rel_path)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()
