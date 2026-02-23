from openai import OpenAI
import openai
import os
import logging
import time
from typing import Any, Dict, Optional, Iterator, TypedDict, List, Callable, Set, Iterable

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, rely on environment variables
    pass

logger = logging.getLogger(__name__)

# ModelSpec import: for this standalone package we import locally.
from .ModelSpec import ModelSpec

# Model registry for parameter mapping
from . import model_registry as _model_registry


class LLMError(Exception):
    """Structured error raised for provider or configuration failures.

    This intentionally subclasses Exception so existing callers that catch
    generic exceptions remain backward-compatible. Callers that want richer
    handling can catch LLMError explicitly and inspect its attributes.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: Optional[str] = None,
        kind: str = "llm_error",
        code: Optional[Any] = None,
        message: str = "",
        retry_after: Optional[float] = None,
    ) -> None:
        self.provider = (provider or "").lower()
        self.model = model
        self.kind = kind  # e.g. "rate_limit", "auth", "config", "model_not_found", "request"
        self.code = code
        self.retry_after = retry_after
        super().__init__(message or kind)


class AdapterResponse:
    """Minimal OpenAI Responses-compatible response shim.

    Your existing tooling can read `output_text` (and optionally `usage`).
    `raw` preserves the provider-native response for debugging.
    """

    def __init__(
        self,
        *,
        output_text: str,
        model: str,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        adapter_response: Any | None = None,
        model_response: Any | None = None,
        status: Optional[str] = None,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ):
        self.output_text = output_text
        self.model = model
        self.usage = usage
        self.metadata = metadata or {}

        self.adapter_response = adapter_response
        self.model_response = model_response

        self.status = status
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls or []


class AdapterEvent:
    """Minimal OpenAI Responses-compatible streaming event shim."""

    def __init__(self, event_type: str, delta: Optional[str] = None):
        self.type = event_type
        self.delta = delta


class EmbeddingResponse:
    """Normalized embedding response structure for all providers.
    
    Provides consistent interface across OpenAI, Gemini, and other providers.
    Always returns direct list of embedding vectors in `data` field.
    """
    
    def __init__(
        self,
        data: List[List[float]],
        usage: Any,
        provider: str,
        model: str,
        normalized: Optional[bool] = None,
        vector_dim: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw: Optional[Any] = None,
    ):
        self.data = data  # Always direct list of embedding vectors
        self.usage = usage  # Standardized usage object
        self.provider = provider  # Provider identifier
        self.model = model  # Model used
        self.normalized = normalized  # Was normalization applied
        self.vector_dim = vector_dim  # Convenience field
        self.metadata = metadata or {}  # Rich metadata
        self.raw = raw  # Original response for debugging


class EmbeddingUsage:
    """Standardized usage information for embedding responses."""
    
    def __init__(self, prompt_tokens: int = 0, total_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens
    
    def __repr__(self) -> str:
        return f"EmbeddingUsage(prompt_tokens={self.prompt_tokens}, total_tokens={self.total_tokens})"


class LLMAdapter:
    ENDPOINT_RESPONSES = "responses"
    ENDPOINT_CHAT_COMPLETIONS = "chat_completions"
    ENDPOINT_GEMINI_SDK = "gemini_sdk"

    def _safe_get(self, obj: Any, name: str) -> Any:
        """Safely read an attribute/key from either an object or dict."""
        try:
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(name)
            return getattr(obj, name, None)
        except Exception:
            return None

    def _canonicalize_usage(self, raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        """Normalize provider usage dict into a canonical schema.

        Always returns canonical keys:
          input_tokens, output_tokens, total_tokens, cached_tokens, reasoning_tokens
        And keeps legacy aliases for backward-compat:
          prompt_tokens, completion_tokens

        This helper is intentionally provider-agnostic; provider-specific nuances
        (e.g., Gemini hidden-thought deltas) should be applied before calling it.
        """
        try:
            if not isinstance(raw, dict) or not raw:
                return None

            # Accept either canonical or legacy keys.
            it = raw.get("input_tokens")
            if it is None:
                it = raw.get("prompt_tokens")

            ot = raw.get("output_tokens")
            if ot is None:
                ot = raw.get("completion_tokens")

            tt = raw.get("total_tokens")

            ct = raw.get("cached_tokens")
            if ct is None:
                ct = raw.get("cached_content_tokens")

            rt = raw.get("reasoning_tokens")

            at = raw.get("answer_tokens")

            in_i = int(it or 0)
            out_i = int(ot or 0)
            cached_i = int(ct or 0)
            reason_i = int(rt or 0)
            answer_i = int(at or 0)

            # Total tokens: prefer explicit value; otherwise default to input+output.
            if tt is None:
                total_i = int(in_i + out_i)
            else:
                total_i = int(tt or 0)

            # Clamp reasoning to non-negative.
            # NOTE: For some providers (e.g., Gemini native SDK), reasoning/thought tokens are
            # reported separately and may exceed visible output tokens, so do not clamp to output.
            if reason_i < 0:
                reason_i = 0

            out: Dict[str, int] = {
                # Canonical
                "input_tokens": in_i,
                "output_tokens": out_i,
                "total_tokens": total_i,
                "cached_tokens": cached_i,
                "reasoning_tokens": reason_i,
                "answer_tokens": answer_i,
                # Legacy aliases
                "prompt_tokens": in_i,
                "completion_tokens": out_i,
            }
            return out
        except Exception:
            return None
    """Unified LLM interface supporting multiple providers."""

    def __init__(
        self,
        *,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        gemini_base_url: Optional[str] = None,
        model_registry: Optional[Dict[str, Any]] = None,
        allowed_model_keys: Optional[Iterable[str]] = None,
        openai_client: Any = None,
        gemini_client: Any = None,
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.gemini_base_url = gemini_base_url or os.getenv("GEMINI_OPENAI_BASE_URL")

        # Registry dict:
        # - Defaults come from package registry (`llm_adapter.model_registry.REGISTRY`).
        # - Callers may provide their own registry mapping to override individual keys.
        #   Whole-object override semantics: user entry replaces the default entry for that key.
        defaults = getattr(_model_registry, "REGISTRY", {})
        if model_registry is None:
            # Keep a per-instance copy so caller code can't accidentally mutate package defaults.
            self.model_registry = dict(defaults)
        else:
            # Merge defaults + user overrides (user wins on key collisions).
            self.model_registry = {**dict(defaults), **dict(model_registry)}

        # Allowlist policy (optional): restrict which registry keys may be used.
        # Precedence:
        # 1) allowed_model_keys passed to ctor (explicit wins)
        # 2) env var LLM_ADAPTER_ALLOWED_MODELS (comma-separated)
        # 3) None (no restriction)
        self.allowed_model_keys: Optional[Set[str]] = None
        if allowed_model_keys is not None:
            try:
                self.allowed_model_keys = {str(k).strip() for k in allowed_model_keys if str(k).strip()}
            except Exception:
                # Best-effort fallback
                try:
                    self.allowed_model_keys = set([str(allowed_model_keys).strip()]) if str(allowed_model_keys).strip() else None
                except Exception:
                    self.allowed_model_keys = None
        else:
            env_val = os.getenv("LLM_ADAPTER_ALLOWED_MODELS")
            if isinstance(env_val, str) and env_val.strip():
                self.allowed_model_keys = {k.strip() for k in env_val.split(",") if k.strip()}

        # Optional injected clients (mirrors chat-with-rag handler ctor)
        self._openai = openai_client
        self._gemini = gemini_client
        self._gemini_native = None
        self.metadata_hook: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None
        self.responses = _ResponsesFacade(self)
        self.embeddings = _EmbeddingsFacade(self)

    class LLMUsage(TypedDict, total=False):
        input_tokens: int
        cached_tokens: int
        output_tokens: int
        reasoning_tokens: int
        answer_tokens: int
        completion_tokens: int
        total_tokens: int

    class LLMToolCall(TypedDict, total=False):
        name: str
        args: Any
        id: Optional[str]

    class LLMResult(TypedDict, total=False):
        provider: str
        model: str
        id: Optional[str]
        created_at: Optional[float]
        text: str
        reasoning: Optional[str]
        role: str
        status: str
        finish_reason: Optional[str]
        usage: "LLMAdapter.LLMUsage"
        tool_calls: List["LLMAdapter.LLMToolCall"]
        metadata: Optional[Dict[str, Any]]
        raw: Any

    def _extract_finish_reason(self, resp: Any) -> Optional[str]:
        try:
            choices = getattr(resp, "choices", None)
            if isinstance(choices, list) and choices:
                return getattr(choices[0], "finish_reason", None)
        except Exception:
            pass
        return None

    def _map_completion_status_from_finish_reason(self, finish_reason: Optional[str]) -> str:
        fr = (str(finish_reason or "").strip() or "").lower()
        if fr in ("length", "max_output_tokens", "max_tokens", "max_tokens_exceeded", "max_tokens_reached", "max_tokens_limit"):
            return "incomplete"
        if fr:
            return "completed"
        return "completed"

    def _map_gemini_native_status_from_finish_reason(self, finish_reason: Optional[str]) -> str:
        fr_raw = str(finish_reason or "").strip()
        fr = fr_raw.upper()
        if fr.endswith("MAX_TOKENS") or fr == "MAX_TOKENS":
            return "incomplete"
        if fr.endswith("STOP") or fr == "STOP":
            return "completed"
        return "completed"

    def _was_normalization_applied(self, provider: str, **kwargs: Any) -> bool:
        """Determine if normalization was applied based on provider and parameters."""
        
        # Check explicit user request first
        if "normalize_embedding" in kwargs:
            return kwargs["normalize_embedding"]
        
        # Provider-specific defaults - match actual implementation
        provider_defaults = {
            "openai": False,        # Never normalizes by default
            "gemini_native": False,  # Only normalizes if explicitly requested
            "gemini": False,        # Depends on endpoint
            "anthropic": False,       # Assume no normalization
        }
        
        return provider_defaults.get(provider, False)

    # ----------------------------
    # Pricing helpers (registry-driven)
    # ----------------------------
    def get_pricing_for_model(self, model: str) -> Optional[Dict[str, Any]]:
        """Return pricing metadata for a model identifier, if present in registry.

        Accepts either:
        - a registry model key (direct dict key), or
        - a provider-native model name (matched against candidate.model).
        """
        try:
            if not model:
                return None

            # Lookup supports both registry keys and provider-native model names.
            mi = self._lookup_model_info_from_registry(model)

            if mi is None:
                return None
            pricing = getattr(mi, "pricing", None)
            if pricing is None:
                return None

            # If registry stores pricing as a dict, return it directly.
            if isinstance(pricing, dict):
                return pricing

            # If registry stores pricing as a typed object (e.g., Pricing dataclass), convert to dict.
            try:
                # dataclasses.asdict support
                import dataclasses
                if dataclasses.is_dataclass(pricing):
                    as_d = dataclasses.asdict(pricing)
                    return as_d if isinstance(as_d, dict) else None
            except Exception:
                pass

            # Common case: simple object with attributes
            try:
                d = dict(getattr(pricing, "__dict__", {}) or {})
                if d:
                    return d
            except Exception:
                pass

            # Optional: support a to_dict() method if provided
            try:
                to_d = getattr(pricing, "to_dict", None)
                if callable(to_d):
                    out = to_d()
                    return out if isinstance(out, dict) else None
            except Exception:
                pass

            return None
        except Exception:
            return None

    def build_llm_result_from_response(
        self,
        resp: Any, # AdapterResponse from llm_adapter.create(...)
        *,
        provider: Optional[str] = None,
        model_key: Optional[str] = None,
    ) -> LLMResult:
        """Build a normalized LLMResult from a non-streaming response."""

        # Fast-path: if caller passed an AdapterResponse from this adapter, do not re-parse
        # provider-native responses here (prevents drift). Only do lightweight normalization.
        if isinstance(resp, AdapterResponse):
            ar = resp
            meta: Dict[str, Any] = ar.metadata if isinstance(ar.metadata, dict) else {}

            provider_norm = str(meta.get("provider") or (provider or "")).strip().lower() or "openai"

            # Prefer adapter-resolved model name.
            try:
                model = str(ar.model or "").strip()
            except Exception:
                model = ""

            rid = meta.get("provider_response_id")
            created_at = meta.get("provider_created_at")

            # Fallbacks (best-effort) if metadata is missing.
            if rid is None:
                try:
                    rid = getattr(ar.model_response, "id", None)
                except Exception:
                    rid = None
            if created_at is None:
                try:
                    created_at = getattr(ar.model_response, "created_at", None)
                except Exception:
                    created_at = None
                if created_at is None:
                    try:
                        created_at = getattr(ar.model_response, "created", None)
                    except Exception:
                        created_at = None

            status = None
            try:
                if isinstance(ar.status, str) and ar.status:
                    status = ar.status
            except Exception:
                status = None
            if not status:
                status = "completed"

            finish_reason = None
            try:
                if isinstance(ar.finish_reason, str) and ar.finish_reason:
                    finish_reason = ar.finish_reason
            except Exception:
                finish_reason = None

            # Usage: AdapterResponse.usage is canonicalized at source (provider wrappers).
            usage: LLMAdapter.LLMUsage = {
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "answer_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            try:
                u = ar.usage if isinstance(ar.usage, dict) else None
                if isinstance(u, dict) and u:
                    it = u.get("input_tokens")
                    ot = u.get("output_tokens")
                    rt = u.get("reasoning_tokens")
                    ct = u.get("cached_tokens")
                    tt = u.get("total_tokens")
                    comp = u.get("completion_tokens")
                    ans = u.get("answer_tokens")

                    usage["input_tokens"] = int(it or 0)
                    usage["output_tokens"] = int(ot or 0)
                    usage["reasoning_tokens"] = int(rt or 0)
                    usage["cached_tokens"] = int(ct or 0)

                    if tt is None:
                        usage["total_tokens"] = int(usage["input_tokens"] + usage["output_tokens"])
                    else:
                        usage["total_tokens"] = int(tt or 0)

                    # completion_tokens is a legacy alias for output_tokens (NOT answer-only).
                    try:
                        if comp is not None:
                            usage["completion_tokens"] = int(comp or 0)
                        else:
                            usage["completion_tokens"] = int(usage["output_tokens"])
                    except Exception:
                        usage["completion_tokens"] = int(usage["output_tokens"])

                    # answer_tokens is the visible/non-reasoning portion of the output.
                    try:
                        if ans is not None:
                            usage["answer_tokens"] = int(ans or 0)
                        else:
                            usage["answer_tokens"] = max(int(usage["output_tokens"]) - int(usage["reasoning_tokens"]), 0)
                    except Exception:
                        usage["answer_tokens"] = max(int(usage["output_tokens"]) - int(usage["reasoning_tokens"]), 0)
            except Exception:
                pass

            # Text + reasoning extraction and stripping thought block from output.
            try:
                best_text = (ar.output_text or "").strip()
            except Exception:
                best_text = ""

            reasoning_text = None
            answer_text = best_text
            try:
                if isinstance(best_text, str):
                    start_tag = "<thought>"
                    end_tag = "</thought>"
                    start = best_text.find(start_tag)
                    end = best_text.find(end_tag)
                    if start != -1 and end != -1 and end > start:
                        inner = best_text[start + len(start_tag) : end]
                        if inner and isinstance(inner, str):
                            reasoning_text = inner.strip()
                        # Strip the thought block from displayed text; keep only answer after </thought>
                        answer_text = best_text[end + len(end_tag) :].strip()
            except Exception:
                pass

            if not isinstance(answer_text, str):
                answer_text = best_text

            # Tool calls: take directly from AdapterResponse.tool_calls (already normalized).
            tool_calls: list[LLMAdapter.LLMToolCall] = []
            try:
                if isinstance(ar.tool_calls, list):
                    tool_calls = [
                        {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                        for tc in ar.tool_calls
                        if isinstance(tc, dict) and isinstance(tc.get("name"), str) and tc.get("name")
                    ]
            except Exception:
                tool_calls = []

            raw_out: Any = None
            try:
                raw_out = ar.model_response or ar.adapter_response
            except Exception:
                raw_out = ar

            return {
                "provider": provider_norm,
                "model": model,
                "id": rid,
                "created_at": created_at,
                "text": answer_text or "",
                "reasoning": reasoning_text,
                "role": "assistant",
                "status": str(status),
                "finish_reason": finish_reason,
                "metadata": meta if meta else None,
                "usage": usage,
                "tool_calls": tool_calls,
                "raw": raw_out,
            }

        raise LLMError(
            provider=str(provider or "unknown"),
            model=None,
            kind="request",
            code="adapter_response_required",
            message="build_llm_result_from_response expects an AdapterResponse produced by LLMAdapter.create()",
        )

    
    def _get_openai(self) -> OpenAI:
        if self._openai is not None:
            return self._openai
        if not self.openai_api_key:
            raise LLMError(
                provider="openai",
                kind="config",
                code="missing_api_key",
                message="OpenAI API key not provided",
            )
        self._openai = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
        return self._openai

    def _get_gemini(self) -> OpenAI:
        """Get Gemini client via OpenAI-compatible interface."""
        if self._gemini is not None:
            return self._gemini
        if not self.gemini_api_key:
            raise LLMError(
                provider="gemini",
                kind="config",
                code="missing_api_key",
                message="Gemini API key not provided",
            )
        base_url = self.gemini_base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        self._gemini = OpenAI(api_key=self.gemini_api_key, base_url=base_url)
        return self._gemini

    def _get_gemini_native(self):
        if self._gemini_native is None:
            try:
                from google import genai  # type: ignore
            except Exception as e:
                raise LLMError(
                    provider="gemini",
                    kind="config",
                    code="missing_dependency",
                    message="Gemini native SDK not available. Install optional dependency: pip install google-genai",
                ) from e

            api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise LLMError(
                    provider="gemini",
                    kind="config",
                    code="missing_api_key",
                    message="Gemini native SDK not configured: GEMINI_API_KEY is not set",
                )

            try:
                self._gemini_native = genai.Client(api_key=api_key)
            except Exception as e:
                raise LLMError(
                    provider="gemini",
                    kind="config",
                    code="native_client_init_failed",
                    message=f"Gemini native SDK client init failed: {e}",
                ) from e

        return self._gemini_native

    def _resolve_provider_model_name(self, model: str) -> str:
        """Resolve model alias/registry key to actual model name."""
        try:
            model_info = self.model_registry.get(model)
            if model_info:
                return str(getattr(model_info, "model", model) or model)
        except Exception:
            pass
        return model

    def _lookup_model_info_from_registry(self, model: str) -> Any | None:
        """Resolve a model identifier to a ModelInfo entry.

        Resolution order:
        1) Direct registry-key lookup (preferred)
        2) Provider-native model name scan (only when allowlist is not enabled)

        If an allowlist is enabled (`self.allowed_model_keys`), callers must use
        registry keys (e.g. "openai:gpt-4o-mini"). Provider-native model names
        (e.g. "gpt-4o-mini") are rejected to avoid ambiguity.
        """
        if not model:
            return None

        def _infer_provider_from_key(k: str) -> str:
            try:
                if isinstance(k, str) and ":" in k:
                    return k.split(":", 1)[0].strip().lower() or "unknown"
            except Exception:
                pass
            return "unknown"

        try:
            # 1) Direct registry-key lookup
            info = self.model_registry.get(model)
            if info is not None:
                if self.allowed_model_keys is not None and model not in self.allowed_model_keys:
                    raise LLMError(
                        provider=_infer_provider_from_key(model),
                        model=model,
                        kind="config",
                        code="model_not_allowed",
                        message=f"Model '{model}' is not in allowed_model_keys",
                    )
                return info

            # If allowlist is enabled, do not allow provider-native name lookup.
            if self.allowed_model_keys is not None:
                raise LLMError(
                    provider=_infer_provider_from_key(str(model)),
                    model=str(model),
                    kind="config",
                    code="model_key_required",
                    message=(
                        "Allowlist is enabled; model must be referenced by registry key "
                        "(e.g. 'openai:gpt-4o-mini'), not a provider-native model name."
                    ),
                )

            # 2) Provider-native model name scan (best-effort, may be ambiguous)
            for candidate_key, candidate in self.model_registry.items():
                if getattr(candidate, "model", None) == model:
                    return candidate
        except LLMError:
            raise
        except Exception:
            return None
        return None


    def _build_adapter_response_metadata(
        self,
        *,
        provider: str,
        model_key: str,
        resolved_model: str,
        dropped_kwargs: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "provider": provider,
            "model": resolved_model,
            "model_key": model_key,
        }
        if isinstance(dropped_kwargs, dict) and dropped_kwargs:
            out["adapter"] = {"dropped_kwargs": dict(dropped_kwargs)}
        return out

    class _GeminiSDKResponsesWrapper:
        def __init__(
            self,
            *,
            output_text: str,
            output: Any,
            usage: Optional[Dict[str, int]],
            model: str,
            model_response: Any,
            finish_reason: Optional[str] = None,
        ) -> None:
            self.output_text = output_text
            self.output = output
            self.usage = usage
            self.model = model
            self.model_response = model_response
            self.finish_reason = finish_reason

    def _extract_native_text(self, resp: Any) -> str:
        try:
            if hasattr(resp, "text") and isinstance(resp.text, str):
                return resp.text
        except Exception:
            pass
        try:
            candidates = getattr(resp, "candidates", None)
            if isinstance(candidates, list) and candidates:
                cand = candidates[0]
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                if isinstance(parts, list) and parts:
                    for p in parts:
                        if isinstance(p, str) and p.strip():
                            return p
                        txt = getattr(p, "text", None)
                        if isinstance(txt, str) and txt.strip():
                            return txt
        except Exception:
            pass
        try:
            return str(resp)
        except Exception:
            return ""

    def _extract_native_text_with_collapsed_thoughts(self, resp: Any) -> str:
        try:
            candidates = getattr(resp, "candidates", None)
            if not isinstance(candidates, list) or not candidates:
                return self._extract_native_text(resp)
            cand = candidates[0]
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not isinstance(parts, list) or not parts:
                return self._extract_native_text(resp)

            thought_texts: list[str] = []
            answer_texts: list[str] = []
            for p in parts:
                if isinstance(p, str):
                    if p.strip():
                        answer_texts.append(p)
                    continue
                txt = getattr(p, "text", None)
                if not isinstance(txt, str) or not txt.strip():
                    continue
                is_thought = False
                try:
                    is_thought = (getattr(p, "thought", None) is True)
                except Exception:
                    is_thought = False
                if is_thought:
                    thought_texts.append(txt)
                else:
                    answer_texts.append(txt)

            if thought_texts:
                thought_block = "\n\n".join([t.strip() for t in thought_texts if t.strip()])
                answer_block = "\n\n".join([t.strip() for t in answer_texts if t.strip()])
                if answer_block:
                    return f"<thought>\n{thought_block}\n</thought>\n\n{answer_block}".strip()
                return f"<thought>\n{thought_block}\n</thought>".strip()

            if answer_texts:
                return "\n\n".join([t.strip() for t in answer_texts if t.strip()]).strip()
            return self._extract_native_text(resp)
        except Exception:
            return self._extract_native_text(resp)

    def _clean_schema(self, obj: Any) -> Any:
        if isinstance(obj, list):
            return [self._clean_schema(x) for x in obj]
        if not isinstance(obj, dict):
            return obj
        forbidden = {"default", "title", "$schema", "additionalProperties", "additional_properties"}
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in forbidden:
                continue
            out[k] = self._clean_schema(v)
        return out

    def _assemble_adapter_response_metadata(
        self,
        *,
        provider: str,
        model_key: str,
        resolved_model: str,
        endpoint: Optional[str] = None,
        raw_response: Any = None,
        dropped_kwargs: Optional[Dict[str, str]] = None,
        now_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        meta = self._build_adapter_response_metadata(
            provider=provider,
            model_key=model_key,
            resolved_model=resolved_model,
            dropped_kwargs=dropped_kwargs,
        )

        if endpoint:
            meta["endpoint"] = str(endpoint)

        def _safe_get(obj: Any, name: str) -> Any:
            try:
                if obj is None:
                    return None
                if isinstance(obj, dict):
                    return obj.get(name)
                return getattr(obj, name, None)
            except Exception:
                return None

        def _safe_get_nested(obj: Any, name: str) -> Any:
            v = _safe_get(obj, name)
            if v is not None:
                return v
            if isinstance(obj, dict):
                for k in ("response", "data", "result"):
                    inner = obj.get(k)
                    if isinstance(inner, dict):
                        vv = inner.get(name)
                        if vv is not None:
                            return vv
            return None

        raw = raw_response

        # ---- provider_response_id ----
        rid = _safe_get_nested(raw, "id")
        if rid is None:
            rid = _safe_get_nested(raw, "response_id")
        if rid is None:
            rid = _safe_get_nested(raw, "responseId")

        if rid is not None:
            meta["provider_response_id"] = str(rid)

        # ---- provider_created_at ----
        created = _safe_get_nested(raw, "created_at")
        if created is None:
            created = _safe_get_nested(raw, "created")

        if created is None and str(provider).lower() == "gemini" and str(endpoint or "").lower() == self.ENDPOINT_GEMINI_SDK:
            import time
            created = int(time.time())

        if created is not None:
            try:
                meta["provider_created_at"] = int(created)
            except Exception:
                meta["provider_created_at"] = created

        # optional hook
        hook = getattr(self, "metadata_hook", None)
        if callable(hook):
            updated = hook(dict(meta), raw)
            if isinstance(updated, dict) and updated:
                meta = updated

        return meta

    def _extract_openai_response_usage(self, resp: Any, endpoint: str) -> Optional[Dict[str, int]]:
        if endpoint == self.ENDPOINT_RESPONSES:
            try:
                u = getattr(resp, "usage", None)
                if u is None:
                    return None

                it = getattr(u, "input_tokens", None)
                ot = getattr(u, "output_tokens", None)
                tt = getattr(u, "total_tokens", None)

                # Cached tokens are a subset of input tokens (billed at a separate rate).
                cached_i = 0
                try:
                    in_details = getattr(u, "input_tokens_details", None)
                    if in_details is not None:
                        cached_i = int(getattr(in_details, "cached_tokens", 0) or 0)
                except Exception:
                    cached_i = 0

                # Parse ints
                try:
                    input_total_i = int(it or 0)
                except Exception:
                    input_total_i = 0

                if cached_i < 0:
                    cached_i = 0
                if cached_i > input_total_i:
                    cached_i = input_total_i

                prompt_uncached_i = input_total_i - cached_i
                if prompt_uncached_i < 0:
                    prompt_uncached_i = 0

                try:
                    output_i = int(ot or 0)
                except Exception:
                    output_i = 0
                if output_i < 0:
                    output_i = 0

                # Reasoning tokens (reported only; do not infer if missing)
                reasoning_i = 0
                reasoning_reported = False
                try:
                    out_details = getattr(u, "output_tokens_details", None)
                    if out_details is not None:
                        r = getattr(out_details, "reasoning_tokens", None)
                        if r is not None:
                            reasoning_i = int(r or 0)
                            if reasoning_i < 0:
                                reasoning_i = 0
                            reasoning_reported = True
                except Exception:
                    reasoning_i = 0
                    reasoning_reported = False

                # Visible answer tokens
                if reasoning_reported:
                    answer_i = output_i - reasoning_i
                    if answer_i < 0:
                        answer_i = 0
                else:
                    answer_i = output_i

                # Total tokens
                try:
                    tt_i = int(tt or 0)
                except Exception:
                    tt_i = 0
                if tt_i <= 0:
                    tt_i = int(input_total_i + output_i)

                # Return final schema fields PLUS temporary backward-compat fields (so nothing breaks yet)
                return {
                    # Final canonical fields
                    "prompt_tokens": int(prompt_uncached_i),
                    "cached_tokens": int(cached_i),
                    "output_tokens": int(output_i),
                    "reasoning_tokens": int(reasoning_i),
                    "answer_tokens": int(answer_i),
                    "total_tokens": int(tt_i),
                    # Temporary backward-compat fields until other paths are refactored
                    "input_tokens": int(input_total_i),
                    "completion_tokens": int(output_i),
                }
            except Exception:
                return None

        elif endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
            try:
                u = getattr(resp, "usage", None)
                if u is None:
                    return None
                pt = getattr(u, "prompt_tokens", None)
                ct = getattr(u, "completion_tokens", None)
                tt2 = getattr(u, "total_tokens", None)

                # ---- Billing-safe normalization (OpenAI chat.completions) ----
                # Provider reports:
                #   prompt_tokens = total prompt tokens (may include cached)
                #   completion_tokens = billed output tokens (answer + reasoning)
                #   total_tokens = total billed (prompt + completion)
                # completion_tokens_details.reasoning_tokens (if present) is a subset of completion_tokens.

                # Extract cached tokens from prompt_tokens_details if present
                cached_details = getattr(u, "prompt_tokens_details", None)
                cc = None
                if cached_details is not None:
                    cc = getattr(cached_details, "cached_tokens", None)

                # Parse ints
                try:
                    pt_total_i = int(pt or 0)
                except Exception:
                    pt_total_i = 0

                try:
                    cached_i = int(cc or 0)
                except Exception:
                    cached_i = 0
                if cached_i < 0:
                    cached_i = 0
                if cached_i > pt_total_i:
                    cached_i = pt_total_i

                prompt_uncached_i = pt_total_i - cached_i
                if prompt_uncached_i < 0:
                    prompt_uncached_i = 0

                try:
                    ct_total_i = int(ct or 0)
                except Exception:
                    ct_total_i = 0
                if ct_total_i < 0:
                    ct_total_i = 0

                # Total tokens
                try:
                    tt_i = int(tt2 or 0)
                except Exception:
                    tt_i = 0

                # Reasoning tokens (reported only; do not infer if missing)
                reasoning_i = 0
                reasoning_reported = False
                try:
                    completion_tokens_details = getattr(u, "completion_tokens_details", None)
                    if completion_tokens_details is not None:
                        r = getattr(completion_tokens_details, "reasoning_tokens", None)
                        if r is not None:
                            reasoning_i = int(r or 0)
                            if reasoning_i < 0:
                                reasoning_i = 0
                            reasoning_reported = True
                except Exception:
                    reasoning_i = 0
                    reasoning_reported = False

                # Visible answer tokens
                if reasoning_reported:
                    answer_i = ct_total_i - reasoning_i
                    if answer_i < 0:
                        answer_i = 0
                else:
                    answer_i = ct_total_i

                # Billed output tokens: OpenAI completion_tokens already matches billed output.
                output_i = ct_total_i

                # If total_tokens missing, compute it.
                if tt_i <= 0:
                    tt_i = int(pt_total_i + output_i)

                # Return final schema fields PLUS temporary backward-compat fields (so nothing breaks yet)
                return {
                    # Final canonical fields
                    "prompt_tokens": int(prompt_uncached_i),
                    "cached_tokens": int(cached_i),
                    "output_tokens": int(output_i),
                    "reasoning_tokens": int(reasoning_i),
                    "answer_tokens": int(answer_i),
                    "total_tokens": int(tt_i),
                    # Temporary backward-compat fields until other paths are refactored
                    "input_tokens": int(pt_total_i),
                    "completion_tokens": int(output_i),
                }
            except Exception:
                return None
        return None

    def _extract_gemini_response_usage(self, resp: Any, endpoint: str) -> Optional[Dict[str, int]]:
        """Extract usage information from Gemini responses.
        
        Handles both Gemini chat completions (OpenAI-compatible) and Gemini native SDK responses.
        """
        if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
            # Gemini OpenAI-compatible chat completions
            try:
                u = getattr(resp, "usage", None)
                if u is None:
                    return None
                pt = u.get("prompt_tokens") if isinstance(u, dict) else getattr(u, "prompt_tokens", None)
                ct = u.get("completion_tokens") if isinstance(u, dict) else getattr(u, "completion_tokens", None)
                tt = u.get("total_tokens") if isinstance(u, dict) else getattr(u, "total_tokens", None)

                # ---- Billing-safe normalization (Gemini OpenAI-compatible) ----
                # Provider typically reports:
                #   prompt_tokens = total prompt tokens (may include cached)
                #   completion_tokens = visible answer tokens
                #   total_tokens = total billed (prompt + answer + thoughts)
                # We normalize to final schema:
                #   prompt_tokens = non-cached prompt (normal-rate)
                #   cached_tokens = cached prompt (cached-rate)
                #   output_tokens = billed output (answer + reasoning) = total - prompt_total
                #   answer_tokens = visible answer
                #   reasoning_tokens = billed output - answer

                # Extract cached tokens from prompt_tokens_details if present
                cached_details = getattr(u, "prompt_tokens_details", None)
                cc = None
                if cached_details is not None:
                    cc = getattr(cached_details, "cached_tokens", None)

                # Parse ints
                try:
                    pt_total_i = int(pt or 0)
                except Exception:
                    pt_total_i = 0

                try:
                    cached_i = int(cc or 0)
                except Exception:
                    cached_i = 0
                if cached_i < 0:
                    cached_i = 0
                if cached_i > pt_total_i:
                    cached_i = pt_total_i

                prompt_uncached_i = pt_total_i - cached_i
                if prompt_uncached_i < 0:
                    prompt_uncached_i = 0

                try:
                    tt_i = int(tt or 0)
                except Exception:
                    tt_i = 0

                # Visible answer (Gemini's completion_tokens)
                try:
                    answer_i = int(ct or 0)
                except Exception:
                    answer_i = 0
                if answer_i < 0:
                    answer_i = 0

                # Billed output tokens
                output_i = tt_i - pt_total_i
                if output_i < 0:
                    output_i = 0

                # Reasoning/thought tokens are the remainder of billed output after visible answer.
                reasoning_calc_i = output_i - answer_i
                if reasoning_calc_i < 0:
                    reasoning_calc_i = 0

                # If provider ever explicitly reports reasoning inside completion_tokens_details (rare),
                # prefer it ONLY if it is <= answer tokens (subset-of-answer semantics).
                reasoning_final_i = reasoning_calc_i
                try:
                    completion_tokens_details = getattr(u, "completion_tokens_details", None)
                    if completion_tokens_details is not None:
                        r = getattr(completion_tokens_details, "reasoning_tokens", None)
                        if r is not None:
                            r_i = int(r or 0)
                            if 0 <= r_i <= answer_i:
                                # In this (OpenAI-like) case, completion includes reasoning; adjust answer accordingly.
                                reasoning_final_i = r_i
                                answer_i = max(answer_i - reasoning_final_i, 0)
                except Exception:
                    pass

                # If total_tokens missing, compute it
                if tt_i <= 0:
                    tt_i = int(pt_total_i + output_i)

                # Return final schema fields PLUS temporary backward-compat fields (so nothing breaks yet)
                return {
                    # Final canonical fields
                    "prompt_tokens": int(prompt_uncached_i),
                    "cached_tokens": int(cached_i),
                    "output_tokens": int(output_i),
                    "reasoning_tokens": int(reasoning_final_i),
                    "answer_tokens": int(answer_i),
                    "total_tokens": int(tt_i),
                    # Temporary backward-compat fields until other paths are refactored
                    "input_tokens": int(pt_total_i),
                    "completion_tokens": int(answer_i),
                }
            except Exception:
                return None
                
        elif endpoint == self.ENDPOINT_GEMINI_SDK:
            # Native SDK (google-genai) fields
            # We normalize to final schema:
            #   prompt_tokens = non-cached prompt (normal-rate)
            #   cached_tokens = cached prompt (cached-rate)
            #   answer_tokens = candidatesTokenCount
            #   reasoning_tokens = thoughtsTokenCount
            #   output_tokens = answer + reasoning
            #   total_tokens = totalTokenCount

            try:
                # Unwrap AdapterResponse if caller passed it
                raw = resp
                try:
                    if isinstance(resp, AdapterResponse):
                        raw = resp.model_response or resp.adapter_response or resp
                except Exception:
                    raw = resp

                # Find usage metadata (object or dict)
                um = None
                if isinstance(raw, dict):
                    um = raw.get("usage_metadata") or raw.get("usageMetadata")
                    # Some wrappers nest the response
                    if um is None and isinstance(raw.get("response"), dict):
                        um = raw["response"].get("usage_metadata") or raw["response"].get("usageMetadata")
                else:
                    um = getattr(raw, "usage_metadata", None)
                    if um is None:
                        um = getattr(raw, "usageMetadata", None)

                if um is None:
                    return None

                def _um_get(name_snake: str, name_camel: str) -> Any:
                    if um is None:
                        return None
                    if isinstance(um, dict):
                        return um.get(name_snake) if name_snake in um else um.get(name_camel)
                    v = getattr(um, name_snake, None)
                    if v is None:
                        v = getattr(um, name_camel, None)
                    return v

                pt_total = _um_get("prompt_token_count", "promptTokenCount")
                ans = _um_get("candidates_token_count", "candidatesTokenCount")
                tt = _um_get("total_token_count", "totalTokenCount")
                reasoning = _um_get("thoughts_token_count", "thoughtsTokenCount")
                cc = _um_get("cached_content_token_count", "cachedContentTokenCount")

                # Parse ints safely
                try:
                    pt_total_i = int(pt_total or 0)
                except Exception:
                    pt_total_i = 0

                try:
                    cached_i = int(cc or 0)
                except Exception:
                    cached_i = 0
                if cached_i < 0:
                    cached_i = 0
                if cached_i > pt_total_i:
                    cached_i = pt_total_i

                prompt_uncached_i = pt_total_i - cached_i
                if prompt_uncached_i < 0:
                    prompt_uncached_i = 0

                try:
                    answer_i = int(ans or 0)
                except Exception:
                    answer_i = 0
                if answer_i < 0:
                    answer_i = 0

                try:
                    reasoning_i = int(reasoning or 0)
                except Exception:
                    reasoning_i = 0
                if reasoning_i < 0:
                    reasoning_i = 0

                # Billed output tokens
                output_i = answer_i + reasoning_i

                # Total tokens
                try:
                    tt_i = int(tt or 0)
                except Exception:
                    tt_i = 0
                if tt_i <= 0:
                    tt_i = int(pt_total_i + output_i)

                # If reported total is inconsistent, recompute (best-effort)
                if tt_i != int(pt_total_i + output_i):
                    tt_i = int(pt_total_i + output_i)

                # Return final schema fields PLUS temporary backward-compat fields (so nothing breaks yet)
                return {
                    # Final canonical fields
                    "prompt_tokens": int(prompt_uncached_i),
                    "cached_tokens": int(cached_i),
                    "output_tokens": int(output_i),
                    "reasoning_tokens": int(reasoning_i),
                    "answer_tokens": int(answer_i),
                    "total_tokens": int(tt_i),
                    # Temporary backward-compat fields until other paths are refactored
                    "input_tokens": int(pt_total_i),
                    "completion_tokens": int(output_i),
                }
            except Exception:
                return None
                
        return None

    def _extract_openai_chatcompletion_text(self, resp: Any) -> str:
        try:
            choices = getattr(resp, "choices", None)
            if isinstance(choices, list) and choices:
                msg = getattr(choices[0], "message", None)
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
        except Exception:
            return ""
        return ""

    def _extract_openai_chatcompletion_finish_reason(self, resp: Any) -> Optional[str]:
        try:
            choices = getattr(resp, "choices", None)
            if isinstance(choices, list) and choices:
                fr = getattr(choices[0], "finish_reason", None)
                if isinstance(fr, str) and fr:
                    return fr
        except Exception:
            return None
        return None

    def _extract_chatcompletion_tool_calls(self, resp: Any) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        try:
            choices = getattr(resp, "choices", None)
            if isinstance(choices, list) and choices:
                msg = getattr(choices[0], "message", None)
                tc_list = getattr(msg, "tool_calls", None)
                if isinstance(tc_list, list):
                    for t in tc_list:
                        t_type = t.get("type") if isinstance(t, dict) else getattr(t, "type", None)
                        if t_type not in ("function", "tool_call"):
                            continue
                        func = (t.get("function") if isinstance(t, dict) else getattr(t, "function", None)) or t
                        name = func.get("name") if isinstance(func, dict) else getattr(func, "name", None)
                        args = func.get("arguments") if isinstance(func, dict) else getattr(func, "arguments", None)
                        cid = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
                        if isinstance(name, str) and name:
                            tool_calls.append({"name": name, "args": args, "id": cid})
        except Exception:
            return []
        return tool_calls

    def _extract_responses_tool_calls(self, resp: Any) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []

        try:
            output = self._safe_get(resp, "output")
            if isinstance(output, list):
                for item in output:
                    it_type = self._safe_get(item, "type")
                    if it_type in ("function_call", "tool_use", "tool_call"):
                        name = self._safe_get(item, "name")
                        args = self._safe_get(item, "arguments")
                        cid = self._safe_get(item, "call_id") or self._safe_get(item, "id")
                        if isinstance(name, str) and name:
                            tool_calls.append({"name": name, "args": args, "id": cid})
                        continue

                    content = self._safe_get(item, "content")
                    if isinstance(content, list):
                        for c in content:
                            c_type = self._safe_get(c, "type")
                            if c_type == "tool_call":
                                name = self._safe_get(c, "name")
                                args = self._safe_get(c, "arguments")
                                cid = self._safe_get(c, "id")
                                if isinstance(name, str) and name:
                                    tool_calls.append({"name": name, "args": args, "id": cid})
        except Exception:
            return []

        return tool_calls

    def _extract_gemini_sdk_tool_calls(self, resp: Any) -> List[Dict[str, Any]]:
        """Extract tool/function calls from Gemini *native* SDK responses (google-genai).

        Expected common shape (Gemini 3 SDK):
          resp.candidates[0].content.parts = [
            {"text": "..."},
            {"function_call": {"name": "...", "args": {...}}},
            ...
          ]

        Returns canonical list items: {"name": <str>, "args": <any>, "id": <optional str>}.
        """
        tool_calls: List[Dict[str, Any]] = []

        # Unwrap AdapterResponse if caller passed it
        raw = resp
        try:
            if isinstance(resp, AdapterResponse):
                raw = resp.model_response or resp.adapter_response or resp
        except Exception:
            raw = resp

        try:
            candidates = getattr(raw, "candidates", None)
            if not isinstance(candidates, list) or not candidates:
                return []

            cand0 = candidates[0]
            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not isinstance(parts, list) or not parts:
                return []

            for p in parts:
                # Most common: parts are dicts
                if isinstance(p, dict):
                    fc = p.get("function_call") or p.get("functionCall")
                    if isinstance(fc, dict):
                        name = fc.get("name")
                        args = fc.get("args") if "args" in fc else fc.get("arguments")
                        cid = p.get("id") or p.get("call_id") or p.get("callId")
                        if isinstance(name, str) and name:
                            tool_calls.append({"name": name, "args": args, "id": cid})
                    continue

                # Fallback: Part objects
                fc = getattr(p, "function_call", None)
                if fc is None:
                    fc = getattr(p, "functionCall", None)
                if fc is None:
                    continue

                name = getattr(fc, "name", None)
                args = getattr(fc, "args", None)
                if args is None:
                    args = getattr(fc, "arguments", None)

                cid = getattr(p, "id", None)
                if cid is None:
                    cid = getattr(p, "call_id", None)
                if cid is None:
                    cid = getattr(p, "callId", None)

                if isinstance(name, str) and name:
                    tool_calls.append({"name": name, "args": args, "id": cid})

        except Exception:
            return []

        return tool_calls
    
    def _get_model_capabilities(self, model: str) -> Dict[str, Any]:
        try:
            mi = self._lookup_model_info_from_registry(model)
            if mi is not None:
                return getattr(mi, "capabilities", {}) or {}
        except Exception:
            pass
        return {}

    def _get_model_param_policy(self, model: str) -> Dict[str, Any]:
        try:
            mi = self._lookup_model_info_from_registry(model)
            if mi is not None:
                pp = getattr(mi, "param_policy", None)
                if isinstance(pp, dict):
                    return pp
        except Exception:
            pass
        return {}

    def _get_model_reasoning_policy(self, model: str) -> Dict[str, Any]:
        try:
            mi = self._lookup_model_info_from_registry(model)
            if mi is not None:
                rp = getattr(mi, "reasoning_policy", None)
                if isinstance(rp, dict):
                    return rp
        except Exception:
            pass
        return {}

    def _apply_openai_reasoning_policy(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(kwargs, dict):
            return kwargs
        policy = self._get_model_reasoning_policy(model)
        if not isinstance(policy, dict) or not policy:
            return kwargs
        if str(policy.get("mode") or "").strip().lower() != "openai_effort":
            return kwargs

        out = dict(kwargs)

        # Respect an explicit provider-native `reasoning` block if caller already passed it.
        if isinstance(out.get("reasoning"), dict) and out.get("reasoning"):
            return out

        effort = out.pop("reasoning_effort", None)
        if effort is None:
            effort = policy.get("default")
        effort_name = self._normalize_effort_name(effort)

        # Treat "none" as no reasoning block.
        if effort_name == "none":
            return out

        out["reasoning"] = {"effort": effort_name}
        return out

    def _apply_gemini_reasoning_policy(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(kwargs, dict):
            return kwargs

        model_info = self._lookup_model_info_from_registry(model)
        if model_info is None or getattr(model_info, "provider", None) != "gemini":
            return kwargs

        policy = self._get_model_reasoning_policy(model)
        if not isinstance(policy, dict) or not policy:
            return kwargs

        mode = str(policy.get("mode") or "").strip().lower()
        if mode not in ("gemini_level", "gemini_budget"):
            return kwargs

        out = dict(kwargs)

        # Determine canonical requested effort (public knob).
        effort = out.pop("reasoning_effort", None)
        if effort is None:
            effort = policy.get("default")
        effort_name = self._normalize_effort_name(effort)

        # Canonical output token key used by adapter.
        max_param = "max_output_tokens"
        base_max = out.get(max_param)
        base_max_i = None
        try:
            if base_max is not None:
                base_max_i = int(base_max)
        except Exception:
            base_max_i = None

        if mode == "gemini_level":
            # Map effort -> thinking_level
            p_name = str(policy.get("param") or "thinking_level")
            mapping = policy.get("map") if isinstance(policy.get("map"), dict) else {}
            mapped_level = mapping.get(effort_name, mapping.get("medium", effort_name))
            out[p_name] = str(mapped_level)

            # Inflate max_output_tokens to reserve room for thoughts (Gemini nuance)
            reserve = policy.get("reserve_ratio") if isinstance(policy.get("reserve_ratio"), dict) else {}
            ratio = reserve.get(effort_name)
            if ratio is None:
                ratio = reserve.get("medium", 0.0)
            try:
                ratio_f = float(ratio)
            except Exception:
                ratio_f = 0.0

            if base_max_i is not None and base_max_i > 0 and ratio_f > 0.0:
                inflated = int(round(base_max_i * (1.0 + ratio_f)))
                if inflated > base_max_i:
                    out[max_param] = inflated

            return out

        # mode == gemini_budget
        p_name = str(policy.get("param") or "thinking_budget")
        budget_map = policy.get("budget_map") if isinstance(policy.get("budget_map"), dict) else {}
        budget = budget_map.get(effort_name)
        if budget is None:
            budget = budget_map.get("medium", budget_map.get("low", None))

        try:
            budget_i = int(budget) if budget is not None else None
        except Exception:
            budget_i = None

        if budget_i is None:
            return out

        # Clamp budget so it cannot consume the entire output cap.
        if base_max_i is not None and base_max_i > 0:
            # Keep at least 100 tokens for visible answer.
            cap = max(base_max_i - 100, 0)
            budget_i = min(budget_i, cap)

        out[p_name] = int(budget_i)
        return out

    def _filter_kwargs_by_capabilities(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        capabilities = self._get_model_capabilities(model)
        if not isinstance(kwargs, dict) or not kwargs:
            return kwargs
        kwargs = dict(kwargs)   # <-- shallow copy so all pop() calls below don't affect caller kwargs dict

        policy = self._get_model_param_policy(model)

        # Rename parameters (public input -> model/provider-specific param name)
        try:
            rename_map = policy.get("rename") if isinstance(policy, dict) else None
            if isinstance(rename_map, dict) and rename_map:
                for src, dst in rename_map.items():
                    try:
                        src_s = str(src)
                        dst_s = str(dst)
                    except Exception:
                        continue
                    if not src_s or not dst_s:
                        continue
                    if src_s in kwargs:
                        val = kwargs.pop(src_s)
                        if dst_s not in kwargs:
                            kwargs[dst_s] = val
        except Exception:
            pass

        # Remove disabled parameters (from registry) before sending to provider
        try:
            disabled = policy.get("disabled") if isinstance(policy, dict) else None
            if isinstance(disabled, (set, list, tuple)):
                for p in disabled:
                    try:
                        p_s = str(p)
                    except Exception:
                        continue
                    if not p_s:
                        continue
                    if p_s in kwargs:
                        kwargs.pop(p_s, None)
        except Exception:
            pass

        token_params = {"max_output_tokens"}

        filtered: Dict[str, Any] = {}
        for param, value in kwargs.items():
            if param in token_params:
                filtered[param] = value
                continue

            if param in capabilities:
                if bool(capabilities.get(param)):
                    filtered[param] = value
                continue

            filtered[param] = value

        return filtered

    def _convert_reasoning_value(self, model: str, value: Any) -> Any:
        mi = self._lookup_model_info_from_registry(model)
        if mi is None:
            return value
        rp = getattr(mi, "reasoning_parameter", None)
        if not (isinstance(rp, (tuple, list)) and len(rp) >= 2):
            return value
        default_value = rp[1]

        if isinstance(default_value, (int, float)):
            mapping = {"low": 1000, "medium": 2000, "high": 5000}
            return mapping.get(str(value).lower(), default_value)

        return str(value).lower()

    def _map_reasoning_parameter_with_default(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        mi = self._lookup_model_info_from_registry(model)
        if mi is None:
            return kwargs
        rp = getattr(mi, "reasoning_parameter", None)
        if not (isinstance(rp, (tuple, list)) and len(rp) >= 2):
            return kwargs

        param_name, default_value = rp[0], rp[1]
        mapped_kwargs = kwargs.copy()

        if "reasoning_effort" in kwargs:
            reasoning_value = kwargs["reasoning_effort"]
            converted_value = self._convert_reasoning_value(model, reasoning_value)
            mapped_kwargs[param_name] = converted_value
            if param_name != "reasoning_effort":
                mapped_kwargs.pop("reasoning_effort", None)
        elif (
            default_value is not None
            and getattr(mi, "capabilities", {}).get("reasoning_effort", False)
            and kwargs.get(param_name) is None
        ):
            mapped_kwargs[param_name] = default_value

        return mapped_kwargs

    def _normalize_effort_name(self, effort: Any) -> str:
        if effort is None:
            return "medium"
        eff = str(effort).strip().lower()
        if eff in ("min", "minimal"):
            return "minimal"
        if eff in ("none", "off", "0"):
            return "none"
        return eff or "medium"

    def _get_requested_effort_from_kwargs(self, model_info: Any, kwargs: Dict[str, Any]) -> Any:
        if kwargs.get("reasoning_effort") is not None:
            return kwargs.get("reasoning_effort")
        try:
            rp = getattr(model_info, "reasoning_parameter", None)
            if isinstance(rp, tuple) and len(rp) >= 1:
                rp_name = rp[0]
                if rp_name and kwargs.get(rp_name) is not None:
                    return kwargs.get(rp_name)
        except Exception:
            pass
        return None

    def _extract_effort_map(self, model_info: Any, spec: Any | None) -> Dict[str, float] | None:
        thinking_tax = getattr(model_info, "thinking_tax", None)
        if isinstance(thinking_tax, dict) and thinking_tax:
            effort_map = thinking_tax.get("effort_map")
            if isinstance(effort_map, dict) and effort_map:
                out: Dict[str, float] = {}
                for k, v in effort_map.items():
                    key = str(k).strip().lower()
                    if isinstance(v, dict):
                        rr = v.get("reserve_ratio")
                    else:
                        rr = v
                    try:
                        out[key] = float(rr)
                    except Exception:
                        continue
                if out:
                    return out

        if spec is None:
            return None

        def _get_from_mapping(obj: Any) -> Any:
            if not isinstance(obj, dict):
                return None
            return obj.get("ratios") or obj.get("effort_ratios") or obj

        spec_map = getattr(spec, "effort_map", None) or getattr(spec, "thinking_tax", None)
        candidate = _get_from_mapping(spec_map)
        if isinstance(candidate, dict) and candidate:
            try:
                return {str(k).strip().lower(): float(v) for k, v in candidate.items()}
            except Exception:
                pass

        for attr_name in ("extra", "extras"):
            maybe = getattr(spec, attr_name, None)
            if isinstance(maybe, dict) and maybe:
                candidate = _get_from_mapping(maybe.get("effort_map") or maybe.get("thinking_tax"))
                if isinstance(candidate, dict) and candidate:
                    try:
                        return {str(k).strip().lower(): float(v) for k, v in candidate.items()}
                    except Exception:
                        pass

        if hasattr(spec, "to_kwargs"):
            try:
                d = spec.to_kwargs() or {}
                if isinstance(d, dict) and d:
                    candidate = _get_from_mapping(d.get("effort_map") or d.get("thinking_tax"))
                    if isinstance(candidate, dict) and candidate:
                        return {str(k).strip().lower(): float(v) for k, v in candidate.items()}
            except Exception:
                pass

        return None

    def _apply_gemini_thinking_tax(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # Deprecated: reasoning behavior is now implemented via `reasoning_policy`.
        return kwargs

    def _inject_gemini_thinking_config(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(kwargs, dict) or not kwargs:
            return kwargs

        model_info = self._lookup_model_info_from_registry(model)
        if model_info is None or getattr(model_info, "provider", None) != "gemini":
            return kwargs

        policy = self._get_model_reasoning_policy(model)
        if not isinstance(policy, dict) or not policy:
            return kwargs

        mode = str(policy.get("mode") or "").strip().lower()
        if mode not in ("gemini_level", "gemini_budget"):
            return kwargs

        out = dict(kwargs)
        existing = out.pop("extra_body", {})
        inner: Dict[str, Any] = {}
        if isinstance(existing, dict):
            inner = existing.get("extra_body", existing)

        inner.setdefault("google", {})
        tc = inner["google"].get("thinking_config")
        if not isinstance(tc, dict):
            tc = {}

        # `include_thoughts` may be set by `_prepare_gemini_adapter_kwargs`.
        if out.get("include_thoughts") is not None:
            tc["include_thoughts"] = bool(out.get("include_thoughts"))
            out.pop("include_thoughts", None)

        if mode == "gemini_level":
            # Prefer explicit thinking_level if present.
            if out.get("thinking_level") is not None:
                tc["thinking_level"] = str(out.get("thinking_level"))
                out.pop("thinking_level", None)

        if mode == "gemini_budget":
            if out.get("thinking_budget") is not None:
                try:
                    tc["thinking_budget"] = int(out.get("thinking_budget"))
                except Exception:
                    tc["thinking_budget"] = out.get("thinking_budget")
                out.pop("thinking_budget", None)

        if tc:
            inner["google"]["thinking_config"] = tc
            out["extra_body"] = {"extra_body": inner}
        else:
            # Preserve existing extra_body shape if nothing added.
            if existing:
                out["extra_body"] = existing

        return out

    def _prepare_gemini_adapter_kwargs(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        filtered_kwargs = self._filter_kwargs_by_capabilities(model, kwargs)
        prepared_kwargs = self._apply_gemini_reasoning_policy(model, filtered_kwargs)
        prepared_kwargs = self._inject_gemini_thinking_config(model, prepared_kwargs)

        # Best-effort: include thoughts when user requested reasoning effort (your UI requirement)
        try:
            include_thoughts = False
            if kwargs.get("reasoning_effort") is not None:
                include_thoughts = True
            if kwargs.get("debug_thoughts"):
                include_thoughts = True

            if include_thoughts:
                model_info = self._lookup_model_info_from_registry(model)
                if model_info is not None and getattr(model_info, "provider", None) == "gemini":
                    caps = getattr(model_info, "capabilities", {}) or {}
                    if bool(caps.get("reasoning_effort", False)):
                        existing = prepared_kwargs.get("extra_body")
                        inner: Dict[str, Any] = {}
                        if isinstance(existing, dict):
                            inner = existing.get("extra_body", existing)
                        inner.setdefault("google", {})
                        tc = inner["google"].get("thinking_config")
                        if not isinstance(tc, dict):
                            tc = {}
                        tc["include_thoughts"] = True
                        inner["google"]["thinking_config"] = tc
                        prepared_kwargs["extra_body"] = {"extra_body": inner}
        except Exception:
            pass

        return prepared_kwargs

    def _sanitize_tools_for_gemini_adapter(self, tools: Any) -> Any:
        if not isinstance(tools, list):
            return tools
        out = []
        for t in tools:
            if not isinstance(t, dict):
                out.append(t)
                continue
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                out.append(t)
                continue
            if "function" in t and isinstance(t.get("function"), dict):
                out.append({"type": "function", "function": t.get("function")})
                continue
            out.append(t)
        return out

    def _wrap_gemini_chatcompletion_as_responses(self, resp: Any, *, model_key: str, resolved_model: str) -> AdapterResponse:
        """Wrap a Gemini (OpenAI-compatible) chat.completions response into a minimal Responses-like shape.

        This mirrors the compatibility behavior in the original llm_handler, so callers can rely on:
        - AdapterResponse.output_text
        - AdapterResponse.output (Responses-style message/content)
        - AdapterResponse.usage (best-effort)
        - AdapterResponse.model_response (provider-native response)
        """

        # Best-effort text extraction from chat.completions shape
        text = ""
        try:
            choices = self._safe_get(resp, "choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                msg = self._safe_get(c0, "message")
                content = self._safe_get(msg, "content")
                if isinstance(content, str):
                    text = content
        except Exception:
            text = ""

        # Best-effort tool call extraction from chat.completions message.tool_calls
        tool_calls = self._extract_chatcompletion_tool_calls(resp)

        # Build a minimal Responses-style output list
        output_list: list[Dict[str, Any]] = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text or ""}],
            }
        ]
        if tool_calls:
            # Keep tool calls as additional content items
            output_list[0]["content"].extend(tool_calls)

        # Best-effort usage mapping to Responses-style keys
        usage_dict = self._extract_gemini_response_usage(resp, self.ENDPOINT_CHAT_COMPLETIONS)
        #print(f"[DEBUG _wrap_gemini_chatcompletion_as_responses] usage dict: {usage_dict}")
        metadata_dict = self._assemble_adapter_response_metadata(
            provider="gemini",
            model_key=model_key,
            resolved_model=resolved_model,
            endpoint=self.ENDPOINT_CHAT_COMPLETIONS,
            raw_response=resp,
        )
        
        # Wrap as AdapterResponse (Responses-like shim)
        finish_reason = self._extract_finish_reason(resp)
        return AdapterResponse(
            output_text=text or "",
            model=resolved_model,
            usage=usage_dict,
            metadata=metadata_dict,
            adapter_response={"output": output_list},
            model_response=resp,
            status=self._map_completion_status_from_finish_reason(finish_reason),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    def create(
        self,
        *,
        input: Any,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        spec: Optional[ModelSpec] = None,
        stream: bool = False,
        **kwargs: Any,
    ):
        if spec is not None:
            provider = spec.provider
            model = spec.model
            merged: Dict[str, Any] = {}
            merged.update(spec.to_kwargs())
            merged.update({k: v for k, v in kwargs.items() if v is not None})
            kwargs = merged
            kwargs["__model_spec"] = spec
        else:
            provider = (provider or "").strip().lower()
            if not model:
                raise ValueError("model is required when spec is not provided")

            # If provider not explicitly provided, infer it from the registry.
            # Prefer registry-key lookup (model as key) and fall back to scanning
            # for a matching provider-native model name.
            if not provider:
                try:
                    mi = self._lookup_model_info_from_registry(model)
                    inferred = getattr(mi, "provider", None) if mi is not None else None
                    if inferred:
                        provider = str(inferred).strip().lower()
                except Exception:
                    provider = ""

            if not provider:
                provider = "openai"

        # Never forward None-valued params to provider SDK calls.
        if isinstance(kwargs, dict) and kwargs:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Canonicalize output token budget across providers/endpoints.
        # Public canonical key: max_output_tokens
        mot = None
        if isinstance(kwargs, dict):
            mot = kwargs.pop("max_output_tokens", None) # max-output-tokens passed to the adapter

        # Clamp to per-model limit if provided in registry (limits may be missing).
        try:
            mi = self._lookup_model_info_from_registry(model) if model else None
            limit = (getattr(mi, "limits", None) or {}).get("max_output_tokens") if mi is not None else None
            if mot is not None and limit is not None:
                try:
                    mot_i = int(mot)
                    limit_i = int(limit)
                    if mot_i > 0 and limit_i > 0:
                        mot = min(mot_i, limit_i)
                except Exception:
                    pass
        except Exception:
            pass

        if mot is not None and isinstance(kwargs, dict):
            kwargs["max_output_tokens"] = mot

        if provider == "openai":
            kwargs.pop("__model_spec", None)
            return self._openai_call(model=model, input=input, stream=stream, **kwargs)
        if provider == "gemini":
            return self._gemini_call(model=model, input=input, stream=stream, **kwargs)

        raise LLMError(
            provider=str(provider or "unknown"),
            model=model,
            kind="config",
            code="unsupported_provider",
            message=f"Provider '{provider}' not supported",
        )

    def create_embedding(
        self,
        *,
        input: Any,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        spec: Optional[ModelSpec] = None,
        **kwargs: Any,
    ):
        if spec is not None:
            provider = spec.provider
            model = spec.model
            merged: Dict[str, Any] = {}
            merged.update(spec.to_kwargs())
            merged.update({k: v for k, v in kwargs.items() if v is not None})
            kwargs = merged
        else:
            provider = (provider or "").strip().lower() or None
            if not model:
                raise ValueError("model is required when spec is not provided")

        # Never forward None-valued params to provider SDK calls.
        if isinstance(kwargs, dict) and kwargs:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        mi = None
        endpoint = None
        try:
            if model:
                mi = self._lookup_model_info_from_registry(model)
                if mi is not None:
                    endpoint = getattr(mi, "endpoint", None)
                    if provider is None and getattr(mi, "provider", None):
                        provider = str(mi.provider)
        except Exception:
            mi = None
            endpoint = None

        if not provider:
            provider = "openai"

        if provider == "openai":
            return self._openai_embedding_call(model=model, input=input, **kwargs)

        if provider == "gemini":
            if endpoint == "embed_content":
                return self._gemini_native_embedding_call(model=model, input=input, **kwargs)
            return self._gemini_embedding_call(model=model, input=input, **kwargs)

        if provider == "gemini_native":
            return self._gemini_native_embedding_call(model=model, input=input, **kwargs)

        raise LLMError(
            provider=str(provider or "unknown"),
            model=model,
            kind="config",
            code="unsupported_provider_embeddings",
            message=f"Provider '{provider}' not supported for embeddings",
        )

    def _openai_call(self, *, model: str, input: Any, stream: bool, **kwargs: Any):
        client = self._get_openai()
        resolved_model = self._resolve_provider_model_name(model)
        filtered_kwargs = self._filter_kwargs_by_capabilities(model, kwargs)
        mapped_kwargs = self._apply_openai_reasoning_policy(model, filtered_kwargs)

        # Canonical output token budget (set by create()); map to endpoint-specific param name later.
        mot = None
        mot = mapped_kwargs.pop("max_output_tokens", None)

        endpoint = self.ENDPOINT_RESPONSES
        try:
            model_info = self._lookup_model_info_from_registry(model)
            if model_info is not None:
                endpoint = getattr(model_info, "endpoint", None)
            else:
                endpoint = self.ENDPOINT_CHAT_COMPLETIONS
        except Exception:
            endpoint = self.ENDPOINT_CHAT_COMPLETIONS

        # Map canonical max_output_tokens to the OpenAI field name for this endpoint.
        if mot is not None:
            try:
                mot_i = int(mot)
            except Exception:
                mot_i = mot
            if endpoint == self.ENDPOINT_RESPONSES:
                mapped_kwargs["max_output_tokens"] = mot_i
            elif endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
                mapped_kwargs["max_completion_tokens"] = mot_i

        if endpoint == self.ENDPOINT_RESPONSES:
            if stream:
                return client.responses.create(model=resolved_model, input=input, stream=True, **mapped_kwargs)

            resp = client.responses.create(model=resolved_model, input=input, stream=False, **mapped_kwargs)
            text = getattr(resp, "output_text", None)
            if not isinstance(text, str):
                text = ""

            status: Optional[str] = None
            finish_reason: Optional[str] = None
            try:
                st = getattr(resp, "status", None)
                if isinstance(st, str) and st:
                    status = st
            except Exception:
                status = None
            try:
                incomplete = getattr(resp, "incomplete_details", None)
                if incomplete is not None:
                    r = getattr(incomplete, "reason", None)
                    if r is not None:
                        finish_reason = str(r)
            except Exception:
                finish_reason = None

            if status:
                status_norm = status.strip().lower()
                if status_norm == "incomplete":
                    status = "incomplete"
                elif status_norm == "completed":
                    status = "completed"

            usage_dict = self._extract_openai_response_usage(resp, endpoint)
            
            metadata_dict = self._assemble_adapter_response_metadata(
                provider="openai",
                model_key=model,
                resolved_model=resolved_model,
                endpoint=endpoint,
                raw_response=resp,
            )
            
            tool_calls = self._extract_responses_tool_calls(resp)
            return AdapterResponse(
                output_text=text,
                model=resolved_model,
                usage=usage_dict,
                metadata=metadata_dict,
                adapter_response=resp,
                model_response=resp,
                status=status or ("incomplete" if finish_reason else "completed"),
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )

        if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
            if "tools" in mapped_kwargs:
                try:
                    mapped_kwargs = dict(mapped_kwargs)
                    mapped_kwargs["tools"] = self._sanitize_tools_for_gemini_adapter(mapped_kwargs["tools"])
                except Exception:
                    pass

            messages = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]
            if not stream:
                resp = client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,
                    stream=False,
                    **mapped_kwargs,
                )
                finish_reason = self._extract_openai_chatcompletion_finish_reason(resp)
                usage_dict = self._extract_openai_response_usage(resp, endpoint)
                
                metadata_dict = self._assemble_adapter_response_metadata(
                    provider="openai",
                    model_key=model,
                    resolved_model=resolved_model,
                    endpoint=endpoint,
                    raw_response=resp,
                )
                
                tool_calls = self._extract_chatcompletion_tool_calls(resp)
                return AdapterResponse(
                    output_text=self._extract_openai_chatcompletion_text(resp),
                    model=resolved_model,
                    usage=usage_dict,
                    metadata=metadata_dict,
                    adapter_response=resp,
                    model_response=resp,
                    status=self._map_completion_status_from_finish_reason(finish_reason),
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                )

            def event_gen() -> Iterator[AdapterEvent]:
                stream_obj = client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,
                    stream=True,
                    timeout=60,
                    **mapped_kwargs,
                )
                for chunk in stream_obj:
                    try:
                        if not getattr(chunk, "choices", None):
                            continue
                        delta_obj = getattr(chunk.choices[0], "delta", None)
                        delta_text = getattr(delta_obj, "content", None)
                        if delta_text:
                            yield AdapterEvent("response.output_text.delta", delta=delta_text)
                    except Exception:
                        continue
                yield AdapterEvent("response.output_text.done")

            return event_gen()

        return client.responses.create(model=resolved_model, input=input, stream=stream, **mapped_kwargs)

    def _openai_embedding_call(self, *, model: str, input: Any, **kwargs: Any):
        import time
        start_time = time.time()
        
        client = self._get_openai()
        resolved_model = self._resolve_provider_model_name(model)
        raw_response = client.embeddings.create(model=resolved_model, input=input, **kwargs)
        
        # Extract embedding vectors from OpenAI response
        vectors = []
        for embedding_obj in raw_response.data:
            vectors.append(embedding_obj.embedding)
        
        # Prepare metadata
        input_texts = input if isinstance(input, list) else [input]
        
        metadata = {
            # Response characteristics
            "dimensions": len(vectors[0]) if vectors else None,
            "vector_type": "dense",
            "precision": "float32",
            
            # Input context
            "input_count": len(input_texts),
            "processing_order": list(range(len(input_texts))),
            
            # Performance and debugging
            "processing_time": time.time() - start_time,
            "raw_response_id": getattr(raw_response, 'id', None),
            "cache_hit": False,
            "retry_count": 0,
            
            # Convenience fields
            "total_tokens_used": getattr(raw_response.usage, 'total_tokens', 0) if hasattr(raw_response, 'usage') else 0,
        }
        
        # Usage extraction (avoid precedence bugs in inline conditional/or expressions)
        u = getattr(raw_response, "usage", None)
        pt = 0
        tt = 0
        try:
            if u is not None:
                pt = getattr(u, "input_tokens", None)
                if pt is None:
                    pt = getattr(u, "prompt_tokens", None)
                pt = int(pt or 0)
                tt = int(getattr(u, "total_tokens", 0) or 0)
        except Exception:
            pt = 0
            tt = 0
        
        return EmbeddingResponse(
            data=vectors,  # Direct list of embedding vectors
            usage=EmbeddingUsage(
                prompt_tokens=pt,
                total_tokens=tt,
            ),
            provider="openai",
            model=resolved_model,
            normalized=self._was_normalization_applied("openai", **kwargs),
            vector_dim=len(vectors[0]) if vectors else None,
            metadata=metadata,
            raw=raw_response
        )

    def _gemini_call(self, *, model: str, input: Any, stream: bool, skip_reasoning: bool = False, **kwargs: Any):
        resolved_model = self._resolve_provider_model_name(model)
        working_kwargs = self._prepare_gemini_adapter_kwargs(model, kwargs)

        # Extract canonical output token budget (create() writes max_output_tokens).
        mot = None
        mot = working_kwargs.pop("max_output_tokens", None)

        endpoint = self.ENDPOINT_CHAT_COMPLETIONS
        try:
            model_info = self._lookup_model_info_from_registry(model)
            if model_info is not None and hasattr(model_info, "endpoint"):
                endpoint = getattr(model_info, "endpoint")
        except Exception:
            pass

        if endpoint == self.ENDPOINT_GEMINI_SDK:
            client = self._get_gemini_native()
        else:
            client = self._get_gemini()

        if endpoint == self.ENDPOINT_GEMINI_SDK:
            native_client = client
  
            # Build contents + config for google-genai.
            # Start from already-prepared kwargs so canonical token handling and thinking config stay consistent.
            sdk_kwargs: Dict[str, Any] = dict(working_kwargs or {}) if isinstance(working_kwargs, dict) else {}
            try:
                sdk_kwargs = self._filter_kwargs_by_capabilities(model, sdk_kwargs)
                sdk_kwargs = self._apply_gemini_reasoning_policy(model, sdk_kwargs)
            except Exception:
                pass

            cfg: Dict[str, Any] = {}
            if sdk_kwargs.get("temperature") is not None:
                cfg["temperature"] = sdk_kwargs.get("temperature")
            if sdk_kwargs.get("top_p") is not None:
                cfg["top_p"] = sdk_kwargs.get("top_p")

            # Token limit mapping: canonical max_output_tokens.
            if mot is not None:
                cfg["max_output_tokens"] = mot

            contents: Any = input
            if isinstance(input, list):
                # Validate and preserve multi-turn structure.
                system_texts: list[str] = []
                non_system: list[Dict[str, Any]] = []
                for m in input:
                    if not isinstance(m, dict):
                        raise LLMError(
                            provider="gemini",
                            model=model,
                            kind="input",
                            code="invalid_message",
                            message="Gemini SDK expects message items to be dicts with role/content",
                        )
                    role = (m.get("role") or "").strip().lower()
                    if role not in ("system", "user", "assistant"):
                        raise LLMError(
                            provider="gemini",
                            model=model,
                            kind="input",
                            code="invalid_role",
                            message=f"Invalid role '{role}'. Allowed roles: system, user, assistant",
                        )
                    if role == "system":
                        system_texts.append(str(m.get("content", "") or ""))
                    else:
                        non_system.append(m)

                if system_texts:
                    cfg["system_instruction"] = "\n".join([t for t in system_texts if t])

                # Build google-genai Contents in order.
                try:
                    from google.genai import types as _types  # type: ignore

                    contents_list: list[Any] = []
                    for m in non_system:
                        role = (m.get("role") or "").strip().lower()
                        mapped_role = "user" if role == "user" else "model"  # assistant -> model
                        text = str(m.get("content", "") or "")
                        if not text:
                            continue
                        part = _types.Part.from_text(text=text)
                        contents_list.append(_types.Content(role=mapped_role, parts=[part]))

                    contents = contents_list if contents_list else ""
                except Exception:
                    # Fallback: collapse transcript as plain text.
                    lines: list[str] = []
                    for m in non_system:
                        role = (m.get("role") or "").strip().lower()
                        label = "User" if role == "user" else "Assistant"
                        text = str(m.get("content", "") or "")
                        if text:
                            lines.append(f"{label}: {text}")
                    contents = "\n".join(lines) if lines else ""

            # Thinking config
            try:
                tc_kwargs: Dict[str, Any] = {}
                if sdk_kwargs.get("include_thoughts") is not None:
                    tc_kwargs["include_thoughts"] = bool(sdk_kwargs.get("include_thoughts"))
                if sdk_kwargs.get("thinking_budget") is not None:
                    tc_kwargs["thinking_budget"] = sdk_kwargs.get("thinking_budget")
                if sdk_kwargs.get("thinking_level") is not None:
                    tc_kwargs["thinking_level"] = sdk_kwargs.get("thinking_level")
                if "thinking_budget" in tc_kwargs and "thinking_level" in tc_kwargs:
                    tc_kwargs.pop("thinking_budget", None)
                if tc_kwargs:
                    from google.genai import types as _types  # type: ignore

                    cfg["thinking_config"] = _types.ThinkingConfig(**tc_kwargs)
            except Exception:
                pass

            # Best-effort: map OpenAI-style tools to google-genai tool declarations.
            try:
                raw_tools = sdk_kwargs.get("tools") if isinstance(sdk_kwargs, dict) else None
                if isinstance(raw_tools, list) and raw_tools:
                    from google.genai import types as _types  # type: ignore

                    
                    fdecls: list[Any] = []
                    for t in raw_tools:
                        if not isinstance(t, dict):
                            continue
                        func = t.get("function") if isinstance(t.get("function"), dict) else t
                        name = func.get("name") if isinstance(func, dict) else None
                        if not name:
                            continue
                        desc = (func.get("description") or "") if isinstance(func, dict) else ""
                        params = (func.get("parameters") or {"type": "OBJECT", "properties": {}}) if isinstance(func, dict) else {"type": "OBJECT", "properties": {}}
                        params = self._clean_schema(params)
                        try:
                            if isinstance(params, dict) and isinstance(params.get("type"), str):
                                params["type"] = str(params.get("type")).strip().upper()
                        except Exception:
                            pass
                        try:
                            schema = _types.Schema(**params) if isinstance(params, dict) else _types.Schema(type="OBJECT")
                        except Exception:
                            schema = _types.Schema(type="OBJECT")
                        try:
                            fdecls.append(
                                _types.FunctionDeclaration(
                                    name=str(name),
                                    description=str(desc),
                                    parameters=schema,
                                )
                            )
                        except Exception:
                            continue
                    if fdecls:
                        cfg["tools"] = [_types.Tool(function_declarations=fdecls)]
            except Exception:
                pass

            # Disable automatic function calling when supported.
            try:
                from google.genai import types as _types  # type: ignore

                if hasattr(_types, "AutomaticFunctionCallingConfig"):
                    cfg["automatic_function_calling"] = _types.AutomaticFunctionCallingConfig(disable=True)
            except Exception:
                pass

            config_obj: Any = None
            if cfg:
                try:
                    from google.genai import types as _types  # type: ignore

                    config_obj = _types.GenerateContentConfig(**cfg)
                except Exception:
                    config_obj = cfg

            if not stream:
                try:
                    resp = native_client.models.generate_content(
                        model=resolved_model,
                        contents=contents,
                        config=config_obj,
                    )
                except Exception as e:
                    raise LLMError(
                        provider="gemini",
                        model=model,
                        kind="provider_error",
                        code="native_generate_failed",
                        message=str(e),
                    ) from e

                text = self._extract_native_text_with_collapsed_thoughts(resp)
                usage_dict = self._extract_gemini_response_usage(resp, self.ENDPOINT_GEMINI_SDK)
                print(f"[DEBUG _gemini_call Native SDK] usage dict for AdapterResponse: {usage_dict}")

                
                output_list = [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    }
                ]

                wrapper = self._GeminiSDKResponsesWrapper(
                    output_text=text,
                    output=output_list,
                    usage=usage_dict,
                    model=resolved_model,
                    model_response=resp,
                    finish_reason=None,
                )

                finish_reason: Optional[str] = None
                try:
                    candidates = getattr(resp, "candidates", None)
                    if isinstance(candidates, list) and candidates:
                        cand0 = candidates[0]
                        fr = getattr(cand0, "finish_reason", None)
                        if fr is None:
                            fr = getattr(cand0, "finishReason", None)
                        if fr is not None:
                            finish_reason = str(fr)
                except Exception:
                    finish_reason = None
                metadata_dict = self._assemble_adapter_response_metadata(
                    provider="gemini",
                    model_key=model,
                    resolved_model=resolved_model,
                    endpoint=endpoint, #gemini_sdk
                    raw_response=resp,
                )
                tool_calls = self._extract_gemini_sdk_tool_calls(resp)
                return AdapterResponse(
                    output_text=text,
                    model=resolved_model,
                    usage=usage_dict,
                    metadata=metadata_dict,
                    adapter_response=wrapper,
                    model_response=resp,
                    status=self._map_gemini_native_status_from_finish_reason(finish_reason),
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                )

            def event_gen() -> Iterator[AdapterEvent]:
                try:
                    resp_stream = native_client.models.generate_content_stream(
                        model=resolved_model,
                        contents=contents,
                        config=config_obj,
                    )
                    for chunk in resp_stream:
                        try:
                            txt = self._extract_native_text(chunk)
                            if txt:
                                yield AdapterEvent("response.output_text.delta", delta=txt)
                        except Exception:
                            continue
                    yield AdapterEvent("response.output_text.done")
                except Exception as e:
                    raise LLMError(
                        provider="gemini",
                        model=model,
                        kind="generation_error",
                        code="stream_error",
                        message=str(e),
                        retry_after=None,
                    ) from e

            return event_gen()
        if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
            if "tools" in working_kwargs:
                try:
                    working_kwargs["tools"] = self._sanitize_tools_for_gemini_adapter(working_kwargs["tools"])
                except Exception:
                    pass

            if mot is not None:
                # Gemini OpenAI-compatible chat.completions uses max_completion_tokens.
                working_kwargs["max_completion_tokens"] = mot

            messages = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]
            resp = client.chat.completions.create(model=resolved_model, messages=messages, stream=stream, **working_kwargs)

            # For non-streaming calls, wrap Gemini chat.completions into a minimal Responses-like shim
            # so callers can consistently access output_text/output/usage across providers.
            if not stream:
                return self._wrap_gemini_chatcompletion_as_responses(resp, model_key=model, resolved_model=resolved_model)

            return resp

    def _gemini_embedding_call(self, *, model: str, input: Any, **kwargs: Any):
        """Gemini embedding call via the OpenAI-compatible adapter client."""
        import time
        start_time = time.time()
        
        client = self._get_gemini()
        resolved_model = self._resolve_provider_model_name(model)

        # Set default dimensions if not provided (matches common registry defaults).
        if "dimensions" not in kwargs:
            kwargs["dimensions"] = 1536

        normalize_embedding = bool(kwargs.pop("normalize_embedding", False))
        raw_response = client.embeddings.create(model=resolved_model, input=input, **kwargs)

        # Extract embedding vectors from Gemini OpenAI-compatible response
        vectors = []
        magnitudes = []
        
        if normalize_embedding and hasattr(raw_response, "data"):
            try:
                import math as _math

                for item in raw_response.data:
                    vec = getattr(item, "embedding", None)
                    if isinstance(vec, list) and vec:
                        s = 0.0
                        for x in vec:
                            try:
                                fx = float(x)
                            except Exception:
                                fx = 0.0
                            s += fx * fx
                        n = _math.sqrt(s)
                        magnitudes.append(n)
                        if n > 0.0:
                            item.embedding = [float(x) / n for x in vec]
                            setattr(item, "magnitude", n)
                            setattr(item, "normalized", True)
                            setattr(item, "provider", "gemini_adapter")
            except Exception:
                pass
        
        # Extract vectors for normalized response
        for item in raw_response.data:
            vec = getattr(item, "embedding", None)
            if isinstance(vec, list) and vec:
                vectors.append(vec)
            elif not normalize_embedding:
                # If not normalized, extract the original vector
                vectors.append(vec if isinstance(vec, list) else [])
        
        # Calculate magnitudes if not already calculated
        if not magnitudes and vectors:
            try:
                import numpy as _np
                magnitudes = [float(_np.linalg.norm(_np.asarray(v, dtype="float32"))) for v in vectors]
            except Exception:
                magnitudes = [1.0] * len(vectors)

        # Prepare metadata
        input_texts = input if isinstance(input, list) else [input]
        
        metadata = {
            # Response characteristics
            "dimensions": len(vectors[0]) if vectors else None,
            "vector_type": "dense",
            "precision": "float32",
            
            # Input context
            "input_count": len(input_texts),
            "processing_order": list(range(len(input_texts))),
            
            # Provider-specific
            "dimensions_requested": kwargs.get("dimensions", 1536),
            
            # Magnitude information
            "magnitudes": magnitudes,
            
            # Performance and debugging
            "processing_time": time.time() - start_time,
            "cache_hit": False,
            "retry_count": 0,
            
            # Convenience fields
            "total_tokens_used": getattr(raw_response.usage, 'total_tokens', 0) if hasattr(raw_response, 'usage') else 0,
        }

        return EmbeddingResponse(
            data=vectors,  # Direct list of embedding vectors
            usage=EmbeddingUsage(
                prompt_tokens=(
                    getattr(raw_response.usage, 'input_tokens', 0) if hasattr(raw_response, 'usage') else 0 or
                    getattr(raw_response.usage, 'prompt_tokens', 0) if hasattr(raw_response, 'usage') else 0
                ),
                total_tokens=getattr(raw_response.usage, 'total_tokens', 0) if hasattr(raw_response, 'usage') else 0
            ),
            provider="gemini",
            model=resolved_model,
            normalized=normalize_embedding,
            vector_dim=len(vectors[0]) if vectors else None,
            metadata=metadata,
            raw=raw_response
        )

    def _gemini_native_embedding_call(self, *, model: str, input: Any, **kwargs: Any):
        """Gemini embedding call via the native google-genai SDK."""
        import time
        start_time = time.time()
        
        client = self._get_gemini_native()
        resolved_model = self._resolve_provider_model_name(model)

        contents = input if isinstance(input, list) else str(input)

        # Get model capabilities for filtering
        model_caps = {}
        try:
            mi = self._lookup_model_info_from_registry(model)
            model_caps = getattr(mi, "capabilities", {}) or {}
            model_caps = dict(model_caps)  # Make a mutable copy
        except Exception:
            model_caps = {}

        # Extract adapter-level parameters before capability filtering
        normalize_embedding = bool(kwargs.pop("normalize_embedding", False))
        
        # Apply capability filtering logic
        filtered_kwargs = {}
        
        # For each capability in model registry:
        for cap_name, cap_value in model_caps.items():
            if cap_value is False:
                # If capability is False, drop it from kwargs even if user passed it
                kwargs.pop(cap_name, None)
            elif cap_value not in [True, False]:
                # If capability has a non-boolean value, use as default unless overridden
                filtered_kwargs[cap_name] = kwargs.pop(cap_name, cap_value)
        
        # Merge filtered capabilities with remaining kwargs
        kwargs = {**filtered_kwargs, **kwargs}
        
        task_type = kwargs.pop("task_type", model_caps.get("task_type"))
        output_dim = kwargs.pop("output_dimensionality", model_caps.get("output_dimensionality"))
        cfg = None
        if task_type is not None or output_dim is not None:
            try:
                from google.genai import types as _types  # type: ignore

                cfg = _types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=output_dim,
                )
            except Exception:
                cfg = None
        
        try:
            raw_response = client.models.embed_content(
                model=resolved_model,
                contents=contents,
                config=cfg,
            )
        except Exception as e:
            raise LLMError(
                provider="gemini",
                model=model,
                kind="provider_error",
                code="native_embed_failed",
                message=str(e),
            ) from e

        # Extract embedding vectors from Gemini response
        vectors: list[list[float]] = []
        try:
            emb_list = getattr(raw_response, "embeddings", None)
            if isinstance(emb_list, list):
                for emb in emb_list:
                    vals = getattr(emb, "values", None)
                    if vals is not None:
                        vectors.append(list(vals))
        except Exception:
            vectors = []

        # Calculate magnitudes
        magnitudes: list[float] = []
        if normalize_embedding and vectors:
            
            try:
                import numpy as _np  # type: ignore

                normalized: list[list[float]] = []
                for v in vectors:
                    arr = _np.asarray(v, dtype="float32")
                    n = float(_np.linalg.norm(arr))
                    if n > 0.0:
                        arr = arr / n
                        magnitudes.append(1.0)  # Normalized magnitude is always 1.0
                    else:
                        magnitudes.append(0.0)  # Zero vector stays zero
                    normalized.append(arr.tolist())
                vectors = normalized
            except Exception:
                magnitudes = [1.0] * len(vectors)
        else:
            try:
                import numpy as _np  # type: ignore

                magnitudes = [float(_np.linalg.norm(_np.asarray(v, dtype="float32"))) for v in vectors]
            except Exception:
                magnitudes = [1.0] * len(vectors)

        # Extract usage information
        usage = None
        try:
            um = getattr(raw_response, "usage_metadata", None)
            if um is not None:
                pt = getattr(um, "prompt_token_count", 0) or 0
                tt = getattr(um, "total_token_count", 0) or 0
                usage = EmbeddingUsage(prompt_tokens=int(pt), total_tokens=int(tt or pt))
        except Exception:
            usage = None

        if usage is None:
            try:
                if isinstance(contents, list):
                    total_text = " ".join(str(c) for c in contents)
                else:
                    total_text = str(contents)
                approx = self._estimate_embedding_tokens(total_text)
                usage = EmbeddingUsage(prompt_tokens=approx, total_tokens=approx)
            except Exception:
                usage = EmbeddingUsage(prompt_tokens=0, total_tokens=0)

        # Prepare metadata
        metadata = {
            # Response characteristics
            "dimensions": len(vectors[0]) if vectors else None,
            "vector_type": "dense",
            "precision": "float32",
            
            # Input context
            "input_count": len(input) if isinstance(input, list) else 1,
            "processing_order": list(range(len(input) if isinstance(input, list) else 1)),
            
            # Provider-specific
            "task_type": task_type,
            "output_dimensionality": output_dim,
            "google_project": getattr(client, '_project_id', None),
            
            # Magnitude information
            "magnitudes": magnitudes,
            
            # Performance and debugging
            "processing_time": time.time() - start_time,
            "cache_hit": False,
            "retry_count": 0,
            
            # Convenience fields
            "total_tokens_used": getattr(usage, 'total_tokens', 0) if usage else 0,
        }

        return EmbeddingResponse(
            data=vectors,  # Direct list of embedding vectors
            usage=usage,
            provider="gemini_native",
            model=resolved_model,
            normalized=normalize_embedding,
            vector_dim=len(vectors[0]) if vectors else None,
            metadata=metadata,
            raw=raw_response
        )

    def _estimate_embedding_tokens(self, text: Any) -> int:
        """Best-effort token estimate for embeddings when provider usage is missing."""
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(str(text), disallowed_special=()))
        except Exception:
            s = str(text) if text is not None else ""
            return max(1, len(s) // 4)


class _ResponsesFacade:
    """Drop-in facade that mimics `client.responses.create(...)`.

    This lets call sites use `llm_adapter.responses.create(...)` while routing
    through the unified `LLMAdapter.create(...)` implementation.
    """

    def __init__(self, handler: LLMAdapter):
        self._handler = handler

    def create(self, **kwargs: Any) -> Any:
        stream = bool(kwargs.pop("stream", False))
        return self._handler.create(**kwargs)



class _EmbeddingsFacade:
    """Drop-in facade that mimics `client.embeddings.create(...)`."""

    def __init__(self, handler: LLMAdapter):
        self._handler = handler

    def create(self, **kwargs: Any):
        return self._handler.create_embedding(**kwargs)


# Default, convenience instance (mirrors prior llm_handler usage pattern).
# Uses environment variables for API keys/base URLs by default.
llm_adapter = LLMAdapter()
