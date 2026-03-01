from typing import Any, Dict
import os

from pathlib import Path

from llm_adapter import model_registry

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_project_root / ".env")

# Custom registry configuration
CUSTOM_REGISTRY_PATH = _project_root / "examples" / "custom_registry.py"

def get_model_options(adapter=None) -> Dict[str, Any]:
    # Use provided adapter's registry or default registry
    registry = getattr(adapter, 'model_registry', model_registry.REGISTRY) if adapter else model_registry.REGISTRY
    
    out: Dict[str, Any] = {}
    for key, mi in (registry or {}).items():
        provider = str(getattr(mi, "provider", "") or "")
        if not provider:
            continue
        
        # Convert ModelInfo to dict with all fields (matching dataclass order)
        model_data = {
            "key": key,  # Use registry dictionary key
            "provider": provider,
            "model": str(getattr(mi, "model", "") or ""),
            "endpoint": str(getattr(mi, "endpoint", "") or ""),
            "pricing": getattr(mi, "pricing", None),
            "param_policy": getattr(mi, "param_policy", {}) or {},
            "limits": getattr(mi, "limits", {}) or {},
            "reasoning_policy": getattr(mi, "reasoning_policy", {}) or {},
            "thinking_tax": getattr(mi, "thinking_tax", {}) or {},
            "reasoning_parameter": getattr(mi, "reasoning_parameter", None),
            "capabilities": getattr(mi, "capabilities", {}) or {},
        }
        
        # Convert Pricing object to dict if present
        if model_data["pricing"] is not None:
            pricing = model_data["pricing"]
            model_data["pricing"] = {
                "input_per_mm": getattr(pricing, "input_per_mm", 0.0),
                "output_per_mm": getattr(pricing, "output_per_mm", 0.0),
                "cached_input_per_mm": getattr(pricing, "cached_input_per_mm", 0.0),
            }
        
        out[key] = model_data
    return out


def is_provider_enabled(provider: str) -> bool:
    # Ensure environment variables are loaded (defensive programming)
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        
        # Reload dotenv to ensure latest environment variables
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        env_path = project_root / ".env"
        
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
    except Exception:
        pass  # Ignore errors, just use existing environment
    
    provider = provider.lower()
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    return False
