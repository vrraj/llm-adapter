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
    key: str
    provider: Provider
    model: str
    endpoint: Endpoint
    pricing: Optional[Pricing]
    capabilities: Dict[str, Any] = field(default_factory=dict)
    max_tokens_parameter: str = "max_output_tokens"
    thinking_tax: Dict[str, Any] = field(default_factory=dict)
    reasoning_parameter: Optional[Tuple[str, Any]] = None


REGISTRY: Dict[str, ModelInfo] = {
    "openai:embed_small": ModelInfo(
        key="openai:embed_small",
        provider="openai",
        model="text-embedding-3-small",
        endpoint="embeddings",
        pricing=Pricing(input_per_mm=0.02, output_per_mm=0.0),
        capabilities={"dimensions": 1536},
        max_tokens_parameter="max_output_tokens",
    ),
    "openai:embed_large": ModelInfo(
        key="openai:embed_large",
        provider="openai",
        model="text-embedding-3-large",
        endpoint="embeddings",
        pricing=Pricing(input_per_mm=0.13, output_per_mm=0.0),
        capabilities={"dimensions": 3072},
        max_tokens_parameter="max_output_tokens",
    ),
    "openai:fast": ModelInfo(
        key="openai:fast",
        provider="openai",
        model="gpt-4o-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=0.15, output_per_mm=0.60, cached_input_per_mm=0.075),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": False,
            "top_p": True,
        },
        max_tokens_parameter="max_output_tokens",
    ),
    "openai:best": ModelInfo(
        key="openai:best",
        provider="openai",
        model="gpt-4o",
        endpoint="responses",
        pricing=Pricing(input_per_mm=2.50, output_per_mm=10.00, cached_input_per_mm=1.25),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": False,
            "top_p": True,
        },
        max_tokens_parameter="max_output_tokens",
    ),
    "openai:chat_fast": ModelInfo(
        key="openai:chat_fast",
        provider="openai",
        model="gpt-4o-mini",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.15, output_per_mm=0.60, cached_input_per_mm=0.075),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": False,
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",
    ),
    "openai:chat_best": ModelInfo(
        key="openai:chat_best",
        provider="openai",
        model="gpt-4o",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=2.50, output_per_mm=10.00, cached_input_per_mm=1.25),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": False,
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",
    ),
    "openai:reasoning_mini": ModelInfo(
        key="openai:reasoning_mini",
        provider="openai",
        model="o3-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=1.10, output_per_mm=4.40),
        capabilities={
            "tools": True,
            "stream": False,
            "temperature": False,
            "reasoning_effort": True,
            "top_p": False,
        },
        max_tokens_parameter="max_output_tokens",
        reasoning_parameter=("reasoning_effort", "low"),
    ),
    "openai:reasoning_mini_small": ModelInfo(
        key="openai:reasoning_mini_small",
        provider="openai",
        model="gpt-5-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=0.25, output_per_mm=2.00),
        capabilities={
            "tools": True,
            "stream": False,
            "temperature": False,
            "reasoning_effort": True,
            "top_p": False,
        },
        max_tokens_parameter="max_output_tokens",
        reasoning_parameter=("reasoning_effort", "minimal"),
    ),
    "gemini:native-embed": ModelInfo(
        key="gemini:native-embed",
        provider="gemini",
        model="gemini-embedding-001",
        endpoint="embed_content",
        pricing=Pricing(input_per_mm=0.10, output_per_mm=0.0),
        capabilities={
            "dimensions": 1536,
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 1536,
            "normalize_embedding": False,
        },
        max_tokens_parameter="max_tokens",
    ),
    "gemini:openai-2.5-flash-lite": ModelInfo(
        key="gemini:openai-2.5-flash-lite",
        provider="gemini",
        model="models/gemini-2.5-flash-lite",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.20, output_per_mm=0.80),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": False,
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",
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
    "gemini:openai-3-flash-preview": ModelInfo(
        key="gemini:openai-3-flash-preview",
        provider="gemini",
        model="models/gemini-3-flash-preview",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.50, output_per_mm=3.00),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": True,
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",
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
    ),
    "gemini:native-sdk-3-flash-preview": ModelInfo(
        key="gemini:native-sdk-3-flash-preview",
        provider="gemini",
        model="models/gemini-3-flash-preview",
        endpoint="gemini_sdk",
        pricing=Pricing(input_per_mm=0.50, output_per_mm=3.00),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": True,
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",
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
        key="gemini:openai-reasoning-2.5-flash",
        provider="gemini",
        model="models/gemini-2.5-flash",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.30, output_per_mm=2.50),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": True,
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",
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
        key="gemini:native-sdk-reasoning-2.5-flash",
        provider="gemini",
        model="models/gemini-2.5-flash",
        endpoint="gemini_sdk",
        pricing=Pricing(input_per_mm=0.30, output_per_mm=2.50),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": True,
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",
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


def resolve_model(
    provider: str | None,
    model: str | None,
    model_key: str | None = None,
) -> Optional[ModelInfo]:
    try:
        reg = REGISTRY or {}
        if not reg:
            return None

        mk = str(model_key).strip() if model_key else ""
        if mk and mk in reg:
            return reg.get(mk)

        m = str(model or "").strip()
        if not m:
            return None
        if m in reg:
            return reg.get(m)

        p = str(provider or "").strip().lower()
        for _k, _v in reg.items():
            try:
                if not _v:
                    continue
                if str(getattr(_v, "model", "")) != m:
                    continue
                if p and str(getattr(_v, "provider", "")).lower() != p:
                    continue
                return _v
            except Exception:
                continue
    except Exception:
        return None
    return None
