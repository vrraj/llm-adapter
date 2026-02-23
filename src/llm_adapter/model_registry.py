from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Literal

Provider = Literal["openai", "gemini"]
Endpoint = Literal["responses", "chat_completions", "embeddings", "gemini_sdk", "embed_content"]


@dataclass(frozen=True)
class Pricing:
    # USD per 1,000,000 tokens
    input_per_mm: float
    output_per_mm: float
    cached_input_per_mm: float = 0.0


@dataclass(frozen=True)
class ModelInfo:
    provider: Provider
    model: str
    endpoint: Endpoint
    pricing: Optional[Pricing]
    param_policy: Dict[str, Any] = field(default_factory=dict)
    limits: Dict[str, Any] = field(default_factory=dict)
    reasoning_policy: Dict[str, Any] = field(default_factory=dict)
    thinking_tax: Dict[str, Any] = field(default_factory=dict)
    reasoning_parameter: Optional[Tuple[str, Any]] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)


REGISTRY: Dict[str, ModelInfo] = {
    "openai:embed_small": ModelInfo(
        provider="openai",
        model="text-embedding-3-small",
        endpoint="embeddings",
        pricing=Pricing(input_per_mm=0.02, output_per_mm=0.0),
        param_policy={
            "allowed": {"normalize_embedding", "dimensions"},
            "disabled": {"include_thoughts", "output_dimensionality"}
        },
        capabilities={
            "dimensions": 1536,
            "assistant_role": "assistant", # Model Response Role  - will be used to send in Request for  conversation
        },
    ),
    "openai:embed_large": ModelInfo(
        provider="openai",
        model="text-embedding-3-large",
        endpoint="embeddings",
        pricing=Pricing(input_per_mm=0.13, output_per_mm=0.0),
        param_policy={
            "allowed": {"normalize_embedding", "dimensions"},
            "disabled": {"include_thoughts", "output_dimensionality"}
        },
        capabilities={
            "dimensions": 3072,
            "assistant_role": "assistant", # Model Response Role  - will be used to send in Request for  conversation
        },
    ),
    "openai:gpt-4o-mini": ModelInfo(
        provider="openai",
        model="gpt-4o-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=0.15, output_per_mm=0.60, cached_input_per_mm=0.075),
        limits={
            "max_output_tokens": 2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": {"reasoning_effort", "stream", "include_thoughts"}
        },
        capabilities={
            "assistant_role": "assistant", # Model Response Role  - will be used to send in Request for  conversation
        },
    ),
    "openai:gpt-4o": ModelInfo(
        provider="openai",
        model="gpt-4o",
        endpoint="responses",
        pricing=Pricing(input_per_mm=2.50, output_per_mm=10.00, cached_input_per_mm=1.25),
        limits={
            "max_output_tokens":  2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": {"reasoning_effort", "stream", "include_thoughts"}
        },
        capabilities={
            "assistant_role": "assistant", # Model Response Role  - will be used to send in Request for  conversation
        },
    ),
    "openai:chat_gpt-4o-mini": ModelInfo(
        provider="openai",
        model="gpt-4o-mini",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.15, output_per_mm=0.60, cached_input_per_mm=0.075),
        limits={
            "max_output_tokens": 2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": {"reasoning_effort", "stream", "include_thoughts"}
        },
        capabilities={
            "assistant_role": "assistant", # Model Response Role  - will be used to send in Request for  conversation
        },
    ),
    "openai:chat_gpt-4o": ModelInfo(
        provider="openai",
        model="gpt-4o",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=2.50, output_per_mm=10.00, cached_input_per_mm=1.25),
        limits={
            "max_output_tokens": 2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": {"reasoning_effort", "stream", "include_thoughts"}
        },
        capabilities={
            "assistant_role": "assistant", # Model Response Role  - will be used to send in Request for  conversation
        },
    ),
    "openai:reasoning_o3-mini": ModelInfo(
        provider="openai",
        model="o3-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=1.10, output_per_mm=4.40),
        limits={
            "max_output_tokens": 2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "reasoning_effort", "reasoning", "tools", "tool_choice"},
            "disabled": {"stream", "temperature", "top_p", "include_thoughts"}
        },
        reasoning_policy={
            "mode": "openai_effort",
            "default": "low",
        },
        reasoning_parameter=("reasoning_effort", "low"),
        capabilities={
            "assistant_role": "assistant", # Model Response Role  - will be used to send in Request for  conversation
            "reasoning_effort": True,
        },
    ),
    "openai:reasoning_gpt-5-mini": ModelInfo(
        provider="openai",
        model="gpt-5-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=0.25, output_per_mm=2.00),
        limits={
            "max_output_tokens": 2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "reasoning_effort", "reasoning", "tools", "tool_choice"},
            "disabled": {"stream", "temperature", "top_p", "include_thoughts"}
        },
        reasoning_policy={
            "mode": "openai_effort",
            "default": "minimal",
        },
        reasoning_parameter=("reasoning_effort", "minimal"),
        capabilities={
            "assistant_role": "assistant", # Model Response Role  - will be used to send in Request for  conversation
            "reasoning_effort": True,
        },
    ),
    "gemini:native-embed": ModelInfo(
        provider="gemini",
        model="gemini-embedding-001",
        endpoint="embed_content",
        pricing=Pricing(input_per_mm=0.10, output_per_mm=0.0),
        param_policy={
            "allowed": {"normalize_embedding", "dimensions", "task_type", "output_dimensionality"},
            "disabled": set()
        },
        capabilities={
            "assistant_role": "model", # Model Response Role  - will be used to send in Request for  conversation
            "dimensions": 1536,
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 1536,
        },
    ),
    "gemini:openai-2.5-flash-lite": ModelInfo(
        provider="gemini",
        model="models/gemini-2.5-flash-lite",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.20, output_per_mm=0.80),
        limits={
            "max_output_tokens": 2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": set()
        },
        thinking_tax={
            "effort_map": {
                "none": {"reserve_ratio": 0.0},
                "low": {"reserve_ratio": 0.25},
                "medium": {"reserve_ratio": 0.50},
                "high": {"reserve_ratio": 0.80},
            },
            "kind": "budget",
        },
        capabilities={
            "assistant_role": "model", # Model Response Role  - will be used to send in Request for  conversation
        },
    ),
    "gemini:openai-3-flash-preview": ModelInfo(
        provider="gemini",
        model="models/gemini-3-flash-preview",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.50, output_per_mm=3.00),
        limits={
            "max_output_tokens": 2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "reasoning_effort", "include_thoughts", "thinking_level", "thinking_budget", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": set()
        },
        reasoning_policy={
            "mode": "gemini_level",
            "param": "thinking_level",
            "default": "minimal",
            "map": {
                "none": "minimal",
                "minimal": "minimal",
                "low": "low",
                "medium": "medium",
                "high": "high",
            },
            "reserve_ratio": {
                "none": 0.0,
                "minimal": 0.25,
                "low": 0.30,
                "medium": 0.50,
                "high": 0.80,
            },
            "counts_against_output": True,
        },
        reasoning_parameter=("thinking_level", "minimal"),
        thinking_tax={
            "effort_map": {
                "none": {"reserve_ratio": 0.0},
                "minimal": {"reserve_ratio": 0.25},
                "low": {"reserve_ratio": 0.30},
                "medium": {"reserve_ratio": 0.50},
                "high": {"reserve_ratio": 0.80},
            },
            "param_map": {
                "none": "minimal",
                "minimal": "minimal",
                "low": "low",
                "medium": "medium",
                "high": "high",
            },
            "kind": "level",
        },
        capabilities={
            "assistant_role": "model", # Model Response Role  - will be used to send in Request for  conversation
            "reasoning_effort": True,
        },
    ),
    "gemini:native-sdk-3-flash-preview": ModelInfo(
        provider="gemini",
        model="models/gemini-3-flash-preview",
        endpoint="gemini_sdk",
        pricing=Pricing(input_per_mm=0.50, output_per_mm=3.00),
        capabilities={
            "assistant_role": "model", # Model Response Role  - will be used to send in Request for  conversation
            "reasoning_effort": True,
        },
        limits={
            "max_output_tokens": 2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "reasoning_effort", "include_thoughts", "thinking_level", "thinking_budget", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": set()
        },
        reasoning_policy={
            "mode": "gemini_level",
            "param": "thinking_level",
            "default": "low",
            "map": {
                "none": "minimal",
                "minimal": "minimal",
                "low": "low",
                "medium": "medium",
                "high": "high",
            },
            "reserve_ratio": {
                "none": 0.0,
                "minimal": 0.25,
                "low": 0.30,
                "medium": 0.50,
                "high": 0.80,
            },
            "counts_against_output": True,
        },
        reasoning_parameter=("thinking_level", "low"),
        thinking_tax={
            "effort_map": {
                "none": {"reserve_ratio": 0.0},
                "minimal": {"reserve_ratio": 0.25},
                "low": {"reserve_ratio": 0.30},
                "medium": {"reserve_ratio": 0.50},
                "high": {"reserve_ratio": 0.80},
            },
            "param_map": {
                "none": "minimal",
                "minimal": "minimal",
                "low": "low",
                "medium": "medium",
                "high": "high",
            },
            "kind": "level",
        },
    ),
    "gemini:openai-reasoning-2.5-flash": ModelInfo(
        provider="gemini",
        model="models/gemini-2.5-flash",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.30, output_per_mm=2.50),
        capabilities={
            "assistant_role": "model", # Model Response Role  - will be used to send in Request for  conversation
            "allow_tools": True,
            "temperature": True,
            "reasoning_effort": True,
            "top_p": True,
        },
        limits={
            "max_output_tokens": 2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "reasoning_effort", "include_thoughts", "thinking_level", "thinking_budget", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": set()
        },
        reasoning_policy={
            "mode": "gemini_budget",
            "param": "thinking_budget",
            "default": "low",
            "budget_map": {
                "none": 0,
                "minimal": 500,
                "low": 1000,
                "medium": 2000,
                "high": 5000,
            },
            "counts_against_output": True,
        },
        reasoning_parameter=("thinking_budget", 1000),
        thinking_tax={
            "effort_map": {
                "none": {"reserve_ratio": 0.0},
                "low": {"reserve_ratio": 0.25},
                "medium": {"reserve_ratio": 0.50},
                "high": {"reserve_ratio": 0.80},
            },
            "kind": "budget",
        },
    ),
    "gemini:native-sdk-reasoning-2.5-flash": ModelInfo(
        provider="gemini",
        model="models/gemini-2.5-flash",
        endpoint="gemini_sdk",
        pricing=Pricing(input_per_mm=0.30, output_per_mm=2.50),
        capabilities={
            "assistant_role": "model", # Model Response Role  - will be used to send in Request for  conversation
            "reasoning_effort": True,
        },
        limits={
            "max_output_tokens": 2000
        },
        param_policy={
            "allowed": {"max_output_tokens", "reasoning_effort", "include_thoughts", "thinking_level", "thinking_budget", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": set()
        },
        reasoning_policy={
            "mode": "gemini_budget",
            "param": "thinking_budget",
            "default": "low",
            "budget_map": {
                "none": 0,
                "minimal": 500,
                "low": 1000,
                "medium": 2000,
                "high": 5000,
            },
            "counts_against_output": True,
        },
        reasoning_parameter=("thinking_budget", 1000),
        thinking_tax={
            "effort_map": {
                "none": {"reserve_ratio": 0.0},
                "low": {"reserve_ratio": 0.25},
                "medium": {"reserve_ratio": 0.50},
                "high": {"reserve_ratio": 0.80},
            },
            "kind": "budget",
        },
    ),
}


def get_model_info(key: str) -> ModelInfo:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key: {key}")
    return REGISTRY[key]

# ------------------------------
# Registry validation helpers
# ------------------------------

def validate_registry(registry: Dict[str, ModelInfo], *, strict: bool = True) -> None:
    """Validate a model registry mapping.

    This is intended for users who supply a custom registry to `LLMAdapter(model_registry=...)`.

    Checks:
    - registry is a non-empty dict
    - provider and endpoint are valid
    - pricing values are non-negative numbers (when pricing is present)
    - limits/capabilities/param_policy/reasoning_policy/thinking_tax are dicts

    If `strict=True`, raises ValueError on the first error set.
    If `strict=False`, raises ValueError with aggregated errors (still raises, but includes all).
    """

    errors: list[str] = []

    if not isinstance(registry, dict) or not registry:
        raise ValueError("REGISTRY must be a non-empty dict[str, ModelInfo]")

    allowed_providers = {"openai", "gemini"}
    allowed_endpoints = {"responses", "chat_completions", "embeddings", "gemini_sdk", "embed_content"}

    def _err(msg: str) -> None:
        if strict:
            raise ValueError(f"Registry validation failed: {msg}")
        errors.append(msg)

    for k, mi in registry.items():
        if not isinstance(k, str) or not k.strip():
            _err(f"Invalid registry key {k!r}: must be a non-empty string")
            continue

        if not isinstance(mi, ModelInfo):
            _err(f"[{k}] value must be a ModelInfo, got {type(mi).__name__}")
            continue

        # provider / endpoint validation
        prov = str(getattr(mi, "provider", "")).strip().lower()
        if prov not in allowed_providers:
            _err(f"[{k}] provider {prov!r} not in {sorted(allowed_providers)}")

        ep = str(getattr(mi, "endpoint", "")).strip()
        if ep not in allowed_endpoints:
            _err(f"[{k}] endpoint {ep!r} not in {sorted(allowed_endpoints)}")

        model_name = str(getattr(mi, "model", "")).strip()
        if not model_name:
            _err(f"[{k}] model must be a non-empty string")

        # pricing validation
        pr = getattr(mi, "pricing", None)
        if pr is not None:
            for field_name in ("input_per_mm", "output_per_mm", "cached_input_per_mm"):
                val = getattr(pr, field_name, None)
                if val is None:
                    continue
                try:
                    f = float(val)
                    if f < 0:
                        _err(f"[{k}] pricing.{field_name} must be >= 0, got {val!r}")
                except Exception:
                    _err(f"[{k}] pricing.{field_name} must be numeric, got {val!r}")

        # dict-shaped fields
        for fname in ("capabilities", "param_policy", "limits", "reasoning_policy", "thinking_tax"):
            v = getattr(mi, fname, None)
            if v is not None and not isinstance(v, dict):
                _err(f"[{k}] {fname} must be a dict if set, got {type(v).__name__}")

        # reasoning_parameter shape
        rp = getattr(mi, "reasoning_parameter", None)
        if rp is not None:
            if not isinstance(rp, tuple) or len(rp) != 2 or not isinstance(rp[0], str):
                _err(f"[{k}] reasoning_parameter must be a (str, Any) tuple, got {rp!r}")

    if errors:
        raise ValueError("Registry validation failed:\n- " + "\n- ".join(errors))
