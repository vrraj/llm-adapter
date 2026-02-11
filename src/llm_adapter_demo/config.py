from typing import Any, Dict
import os

from pathlib import Path

from llm_adapter import model_registry

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_project_root / ".env")

def get_model_options() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, mi in (model_registry.REGISTRY or {}).items():
        provider = str(getattr(mi, "provider", "") or "")
        if not provider:
            continue
        out[key] = {
            "key": key,
            "provider": provider,
            "model": str(getattr(mi, "model", "") or ""),
            "endpoint": str(getattr(mi, "endpoint", "") or ""),
            "capabilities": getattr(mi, "capabilities", {}) or {},
        }
    return out


def is_provider_enabled(provider: str) -> bool:
    provider = provider.lower()
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    return False
