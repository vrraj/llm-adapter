from openai import OpenAI
import openai
import os
import logging
from typing import Any, Dict, Optional, Iterator, TypedDict, List

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
    ):
        self.output_text = output_text
        self.model = model
        self.usage = usage
        self.metadata = metadata or {}

        self.adapter_response = adapter_response
        self.model_response = model_response

        self.status = status
        self.finish_reason = finish_reason


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
    """Unified LLM interface supporting multiple providers."""

    def __init__(
        self,
        *,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        gemini_base_url: Optional[str] = None,
        model_registry: Optional[Dict[str, Any]] = None,
        openai_client: Any = None,
        gemini_client: Any = None,
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.gemini_base_url = gemini_base_url or os.getenv("GEMINI_OPENAI_BASE_URL")

        # Registry dict: prefer REGISTRY if present; otherwise accept injected mapping.
        self.model_registry = model_registry or getattr(_model_registry, "REGISTRY", {})

        # Optional injected clients (mirrors chat-with-rag handler ctor)
        self._openai = openai_client
        self._gemini = gemini_client
        self._gemini_native = None

        self.responses = _ResponsesFacade(self)
        self.embeddings = _EmbeddingsFacade(self)

    class LLMUsage(TypedDict, total=False):
        input_tokens: int
        cached_tokens: int
        output_tokens: int
        reasoning_tokens: int
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
    def get_pricing_for_model_key(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Return pricing metadata for a registry model key, if present.

        This adapter does not compute costs; it only exposes any pricing metadata
        stored in the model registry (if you choose to include it there).
        """
        try:
            if not model_key:
                return None
            mi = self.model_registry.get(model_key)
            if mi is None:
                return None
            pricing = getattr(mi, "pricing", None)
            return pricing if isinstance(pricing, dict) else None
        except Exception:
            return None

    def get_pricing_for_model(self, model: str) -> Optional[Dict[str, Any]]:
        """Return pricing metadata for a provider model name, if present in registry."""
        try:
            if not model:
                return None
            mi = self._lookup_model_info_from_registry(model)
            if mi is None:
                return None
            pricing = getattr(mi, "pricing", None)
            return pricing if isinstance(pricing, dict) else None
        except Exception:
            return None

    def build_llm_result_from_response(
        self,
        resp: Any,
        *,
        provider: Optional[str] = None,
        model_key: Optional[str] = None,
    ) -> LLMResult:
        """Build a normalized LLMResult from a non-streaming response."""

        # If caller passed an AdapterResponse (from this adapter), prefer its already-derived
        # fields (output_text/usage/status/finish_reason/metadata) and only fall back to
        # provider-native parsing when missing.
        adapter_usage: Optional[Dict[str, int]] = None
        adapter_output_text: Optional[str] = None
        adapter_status: Optional[str] = None
        adapter_finish_reason: Optional[str] = None
        adapter_metadata: Optional[Dict[str, Any]] = None
        raw_resp: Any = resp
        if isinstance(resp, AdapterResponse):
            try:
                raw_resp = resp.model_response or resp.adapter_response or resp
            except Exception:
                raw_resp = resp
            try:
                if isinstance(resp.usage, dict) and resp.usage:
                    adapter_usage = dict(resp.usage)
            except Exception:
                adapter_usage = None
            try:
                if isinstance(resp.output_text, str) and resp.output_text.strip():
                    adapter_output_text = resp.output_text.strip()
            except Exception:
                adapter_output_text = None
            try:
                if isinstance(resp.status, str) and resp.status:
                    adapter_status = resp.status
            except Exception:
                adapter_status = None
            try:
                if isinstance(resp.finish_reason, str) and resp.finish_reason:
                    adapter_finish_reason = resp.finish_reason
            except Exception:
                adapter_finish_reason = None
            try:
                if isinstance(resp.metadata, dict) and resp.metadata:
                    adapter_metadata = dict(resp.metadata)
            except Exception:
                adapter_metadata = None

        # Use raw_resp for any provider-native parsing below.
        resp = raw_resp

        # Provider inference (optional): callers may pass provider and/or model_key.
        # If provider is not provided, infer it from:
        #   1) registry key (model_key), then
        #   2) provider-native model name on the response (resp.model), then
        #   3) default to "openai" for backward compatibility.
        provider_norm = (provider or "").strip().lower() or None
        if not provider_norm:
            try:
                if model_key:
                    mi = self.model_registry.get(model_key)
                else:
                    mi = None
                if mi is None:
                    try:
                        resp_model = getattr(resp, "model", None)
                    except Exception:
                        resp_model = None
                    if resp_model:
                        mi = self._lookup_model_info_from_registry(str(resp_model))
                inferred = getattr(mi, "provider", None) if mi is not None else None
                if inferred:
                    provider_norm = str(inferred).strip().lower()
            except Exception:
                provider_norm = None
        if not provider_norm:
            provider_norm = "openai"

        try:
            model = getattr(resp, "model", None) or ""
        except Exception:
            model = ""

        try:
            rid = getattr(resp, "id", None)
        except Exception:
            rid = None

        try:
            created_at = getattr(resp, "created_at", None)
        except Exception:
            created_at = None

        try:
            st_attr = getattr(resp, "status", None)
        except Exception:
            st_attr = None

        status: Optional[str] = None
        if st_attr is not None:
            try:
                status = str(st_attr).strip() or None
            except Exception:
                status = None
        if adapter_status is not None:
            try:
                status = str(adapter_status).strip() or status
            except Exception:
                pass
        if not status:
            status = "completed"

        finish_reason: Optional[str] = None
        try:
            fr_attr = getattr(resp, "finish_reason", None)
            if fr_attr is not None:
                finish_reason = str(fr_attr).strip() or None
        except Exception:
            finish_reason = None
        if adapter_finish_reason is not None and not finish_reason:
            try:
                finish_reason = str(adapter_finish_reason).strip() or finish_reason
            except Exception:
                pass

        if finish_reason is None:
            fr2 = self._extract_finish_reason(resp)
            if fr2 is not None:
                try:
                    finish_reason = str(fr2).strip() or None
                except Exception:
                    finish_reason = None
        if finish_reason is None:
            try:
                incomplete = getattr(resp, "incomplete_details", None)
                if incomplete is not None:
                    r = getattr(incomplete, "reason", None)
                    if r is not None:
                        finish_reason = str(r).strip() or None
            except Exception:
                finish_reason = None

        if finish_reason is None and provider_norm == "gemini":
            try:
                candidates = getattr(resp, "candidates", None)
                if isinstance(candidates, list) and candidates:
                    cand0 = candidates[0]
                    fr = getattr(cand0, "finish_reason", None)
                    if fr is None:
                        fr = getattr(cand0, "finishReason", None)
                    if fr is not None:
                        finish_reason = str(fr).strip() or None
            except Exception:
                finish_reason = None

        # Derive status from finish_reason when status is missing or not meaningful.
        try:
            status_norm = (status or "").strip().lower()
        except Exception:
            status_norm = ""

        if status_norm not in ("completed", "incomplete"):
            status_norm = "completed"

        if status_norm == "completed" and finish_reason:
            try:
                if provider_norm == "gemini" and isinstance(finish_reason, str):
                    # Normalize enum-ish string values like "FinishReason.MAX_TOKENS" -> "MAX_TOKENS".
                    if "." in finish_reason:
                        finish_reason = finish_reason.split(".")[-1]
            except Exception:
                pass
            if provider_norm == "gemini":
                status_norm = self._map_gemini_native_status_from_finish_reason(finish_reason)
            else:
                status_norm = self._map_completion_status_from_finish_reason(finish_reason)

        status = status_norm

        # Prefer AdapterResponse-provided usage when available; otherwise parse provider-native usage.
        usage: LLMAdapter.LLMUsage = {
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        def _apply_adapter_usage(u: Dict[str, Any]) -> None:
            # AdapterResponse.usage may be Responses-style (input/output/total) or
            # chat.completions-style (prompt/completion/total). Normalize into LLMUsage.
            try:
                it = u.get("input_tokens")
                ot = u.get("output_tokens")
                tt = u.get("total_tokens")
                pt = u.get("prompt_tokens")
                ct = u.get("completion_tokens")
                cached = u.get("cached_tokens")
                rtok = u.get("reasoning_tokens")

                if it is None and pt is not None:
                    it = pt
                if ot is None and ct is not None:
                    ot = ct
                if tt is None:
                    try:
                        tt = (int(it or 0) + int(ot or 0))
                    except Exception:
                        tt = 0

                usage["input_tokens"] = int(it or 0)
                usage["output_tokens"] = int(ot or 0)
                usage["total_tokens"] = int(tt or 0)
                usage["cached_tokens"] = int(cached or 0)
                usage["reasoning_tokens"] = int(rtok or 0)

                # completion_tokens should be visible output excluding reasoning where possible.
                try:
                    usage["completion_tokens"] = max(int(usage["output_tokens"]) - int(usage["reasoning_tokens"]), 0)
                except Exception:
                    usage["completion_tokens"] = int(ot or 0)
            except Exception:
                pass

        if adapter_usage is not None:
            _apply_adapter_usage(adapter_usage)
        else:
            usage_block = getattr(resp, "usage", None)
            if usage_block is not None:
                try:
                    def _uget(obj: Any, name: str) -> Any:
                        if obj is None:
                            return None
                        if isinstance(obj, dict):
                            return obj.get(name)
                        return getattr(obj, name, None)

                    completion_tokens = None
                    if provider_norm == "openai":
                        input_tokens = _uget(usage_block, "input_tokens")
                        if input_tokens is None:
                            input_tokens = _uget(usage_block, "prompt_tokens")

                        output_tokens = _uget(usage_block, "output_tokens")
                        if output_tokens is None:
                            output_tokens = _uget(usage_block, "completion_tokens")

                        total_tokens = _uget(usage_block, "total_tokens")

                        details_in = _uget(usage_block, "prompt_tokens_details") or _uget(usage_block, "input_tokens_details")
                        cached_tokens = (
                            getattr(details_in, "cached_tokens", None)
                            if details_in is not None and not isinstance(details_in, dict)
                            else (details_in.get("cached_tokens") if isinstance(details_in, dict) else None)
                        )

                        details_out = _uget(usage_block, "output_tokens_details")
                        reasoning_tokens = None
                        if details_out is not None:
                            if isinstance(details_out, dict):
                                reasoning_tokens = details_out.get("reasoning_tokens")
                            else:
                                reasoning_tokens = getattr(details_out, "reasoning_tokens", None)

                        if reasoning_tokens is None:
                            details_out2 = _uget(usage_block, "completion_tokens_details")
                            if details_out2 is not None:
                                if isinstance(details_out2, dict):
                                    reasoning_tokens = details_out2.get("reasoning_tokens")
                                else:
                                    reasoning_tokens = getattr(details_out2, "reasoning_tokens", None)

                    elif provider_norm == "gemini":
                        prompt_tokens = _uget(usage_block, "prompt_tokens")
                        input_tokens = prompt_tokens if prompt_tokens is not None else _uget(usage_block, "input_tokens")

                        completion_tokens_raw = _uget(usage_block, "completion_tokens")
                        total_tokens = _uget(usage_block, "total_tokens")

                        output_tokens = None
                        cached_tokens = None
                        reasoning_tokens = None

                        if total_tokens is not None and input_tokens is not None:
                            try:
                                output_tokens = int(total_tokens) - int(input_tokens)
                            except Exception:
                                output_tokens = None

                        if output_tokens is not None and completion_tokens_raw is not None:
                            try:
                                reasoning_tokens = int(output_tokens) - int(completion_tokens_raw)
                                if reasoning_tokens < 0:
                                    reasoning_tokens = 0
                            except Exception:
                                reasoning_tokens = None

                        completion_tokens = completion_tokens_raw

                        if reasoning_tokens in (None, 0):
                            try:
                                raw_r = getattr(resp, "model_response", None) or resp
                                um2 = getattr(raw_r, "usage_metadata", None)
                                if um2 is None:
                                    um2 = getattr(raw_r, "usageMetadata", None)
                                if um2 is not None:
                                    rt = getattr(um2, "thoughts_token_count", None)
                                    if rt is None:
                                        rt = getattr(um2, "thoughtsTokenCount", None)
                                    if rt is not None:
                                        reasoning_tokens = int(rt or 0)
                            except Exception:
                                pass

                    else:
                        input_tokens = output_tokens = cached_tokens = reasoning_tokens = total_tokens = None
                        completion_tokens = None

                    input_tokens = input_tokens if input_tokens is not None else 0
                    output_tokens = output_tokens if output_tokens is not None else 0
                    cached_tokens = cached_tokens if cached_tokens is not None else 0
                    reasoning_tokens = reasoning_tokens if reasoning_tokens is not None else 0

                    if total_tokens is None:
                        total_tokens = (input_tokens or 0) + (output_tokens or 0)

                    if completion_tokens is None:
                        try:
                            completion_tokens = int(output_tokens or 0) - int(reasoning_tokens or 0)
                            if completion_tokens < 0:
                                completion_tokens = 0
                        except Exception:
                            completion_tokens = 0

                    usage["input_tokens"] = int(input_tokens or 0)
                    usage["output_tokens"] = int(output_tokens or 0)
                    usage["total_tokens"] = int(total_tokens or 0)
                    usage["cached_tokens"] = int(cached_tokens or 0)
                    usage["reasoning_tokens"] = int(reasoning_tokens or 0)
                    usage["completion_tokens"] = int(completion_tokens or 0)
                except Exception:
                    pass

        def _safe_get(obj: Any, name: str) -> Any:
            try:
                if isinstance(obj, dict):
                    return obj.get(name)
                return getattr(obj, name, None)
            except Exception:
                return None

        text_candidates: list[str] = []
        if adapter_output_text is not None:
            text_candidates.append(adapter_output_text)
        try:
            ot = _safe_get(resp, "output_text")
            if isinstance(ot, str) and ot.strip():
                text_candidates.append(ot.strip())
        except Exception:
            pass

        try:
            output = _safe_get(resp, "output")
            if isinstance(output, list):
                for item in output:
                    it_type = _safe_get(item, "type")
                    if it_type == "text":
                        txt = _safe_get(item, "text")
                        if isinstance(txt, str) and txt.strip():
                            text_candidates.append(txt.strip())
                            continue
                    content = _safe_get(item, "content")
                    if isinstance(content, list):
                        for c in content:
                            txt = _safe_get(c, "text")
                            if isinstance(txt, str) and txt.strip():
                                text_candidates.append(txt.strip())
        except Exception:
            pass

        if not text_candidates:
            try:
                choices = _safe_get(resp, "choices")
                if isinstance(choices, list) and choices:
                    c0 = choices[0]
                    msg = _safe_get(c0, "message")
                    content = _safe_get(msg, "content")
                    if isinstance(content, str) and content.strip():
                        text_candidates.append(content.strip())
            except Exception:
                pass

        best_text = ""
        for c in text_candidates:
            if isinstance(c, str) and len(c) > len(best_text):
                best_text = c

        reasoning_text = None
        try:
            if provider_norm == "gemini" and isinstance(best_text, str):
                start_tag = "<thought>"
                end_tag = "</thought>"
                start = best_text.find(start_tag)
                end = best_text.find(end_tag)
                if start != -1 and end != -1 and end > start:
                    inner = best_text[start + len(start_tag) : end]
                    after = best_text[end + len(end_tag) :].strip()
                    if inner and isinstance(inner, str):
                        reasoning_text = inner.strip()
                    if after:
                        best_text = after
        except Exception:
            pass

        tool_calls: list[LLMAdapter.LLMToolCall] = []
        try:
            output = _safe_get(resp, "output")
            if isinstance(output, list):
                for item in output:
                    it_type = _safe_get(item, "type")
                    if it_type in ("function_call", "tool_use", "tool_call"):
                        name = _safe_get(item, "name")
                        args = _safe_get(item, "arguments")
                        cid = _safe_get(item, "call_id") or _safe_get(item, "id")
                        if isinstance(name, str) and name:
                            tool_calls.append({"name": name, "args": args, "id": cid})
                        continue

                    content = _safe_get(item, "content")
                    if isinstance(content, list):
                        for c in content:
                            c_type = _safe_get(c, "type")
                            if c_type == "tool_call":
                                name = _safe_get(c, "name")
                                args = _safe_get(c, "arguments")
                                cid = _safe_get(c, "id")
                                if isinstance(name, str) and name:
                                    tool_calls.append({"name": name, "args": args, "id": cid})
        except Exception:
            pass

        try:
            if not tool_calls:
                choices = _safe_get(resp, "choices")
                if isinstance(choices, list) and choices:
                    c0 = choices[0]
                    msg = _safe_get(c0, "message")
                    tc_list = _safe_get(msg, "tool_calls")
                    if isinstance(tc_list, list):
                        for t in tc_list:
                            t_type = _safe_get(t, "type")
                            if t_type not in ("function", "tool_call"):
                                continue
                            func = _safe_get(t, "function") or t
                            name = _safe_get(func, "name")
                            args = _safe_get(func, "arguments")
                            cid = _safe_get(t, "id")
                            if isinstance(name, str) and name:
                                tool_calls.append({"name": name, "args": args, "id": cid})
        except Exception:
            pass

        metadata: Optional[Dict[str, Any]] = None
        if adapter_metadata is not None:
            metadata = adapter_metadata
        else:
            try:
                md = getattr(resp, "metadata", None)
                if isinstance(md, dict):
                    metadata = md
            except Exception:
                metadata = None

        result: LLMAdapter.LLMResult = {
            "provider": provider_norm,
            "model": model,
            "id": rid,
            "created_at": created_at,
            "text": best_text or "",
            "reasoning": reasoning_text,
            "role": "assistant",
            "status": status,
            "finish_reason": finish_reason,
            "metadata": metadata,
            "usage": usage,
            "tool_calls": tool_calls,
            "raw": resp,
        }
        return result

    
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

    def _resolve_model_name(self, model: str) -> str:
        """Resolve model alias/registry key to actual model name."""
        try:
            model_info = self.model_registry.get(model)
            if model_info:
                return str(getattr(model_info, "model", model) or model)
        except Exception:
            pass
        return model

    def _lookup_model_info_from_registry(self, model: str) -> Any | None:
        if not model:
            return None
        try:
            info = self.model_registry.get(model)
            if info is not None:
                return info
            for candidate in self.model_registry.values():
                if getattr(candidate, "model", None) == model:
                    return candidate
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

    def _extract_openai_response_usage(self, resp: Any, endpoint: str) -> Optional[Dict[str, int]]:
        if endpoint == "responses":
            try:
                u = getattr(resp, "usage", None)
                if u is None:
                    return None
                it = getattr(u, "input_tokens", None)
                ot = getattr(u, "output_tokens", None)
                tt = getattr(u, "total_tokens", None)
                reasoning = None

                # Extract reasoning_tokens from output_tokens_details if present
                out_details = getattr(u, "output_tokens_details", None)
    
                if out_details is not None:
                    reasoning = getattr(out_details, "reasoning_tokens", None)
                    print(f"[DEBUG _extract_openai_response_usage] reasoning: {reasoning}")
                if it is not None or ot is not None or tt is not None:
                    out: Dict[str, int] = {}
                    if it is not None:
                        out["input_tokens"] = int(it or 0)
                    if ot is not None:
                        out["output_tokens"] = int(ot or 0)
                    if tt is not None:
                        out["total_tokens"] = int(tt or ((it or 0) + (ot or 0)))
                    if reasoning is not None:
                        out["reasoning_tokens"] = int(reasoning or 0)
                    print(f"[DEBUG _extract_openai_response_usage] final out dict: {out}")
                    return out
            except Exception:
                return None
        elif endpoint == "chat_completions":
            try:
                u = getattr(resp, "usage", None)
                if u is None:
                    print(f"[DEBUG _extract_openai_response_usage] chatcompletion - usage object is None")
                    return None
                print(f"[DEBUG _extract_openai_response_usage] chatcompletion - usage object: {u}")
                pt = getattr(u, "prompt_tokens", None)
                ct = getattr(u, "completion_tokens", None)
                tt2 = getattr(u, "total_tokens", None)
                print(f"[DEBUG _extract_openai_response_usage] chatcompletion - pt: {pt}, ct: {ct}, tt2: {tt2}")
                #Extract cached_content_tokens from prompt_tokens_details if present
                cached_details = getattr(u, "prompt_tokens_details", None)
                cc = None
                if cached_details is not None:
                    cc = getattr(cached_details, "cached_tokens", None)

                reasoning = None
                
                if pt is not None or ct is not None or tt2 is not None:
                    out2 = {
                        "prompt_tokens": int(pt or 0),
                        "completion_tokens": int(ct or 0),
                        "total_tokens": int(tt2 or ((pt or 0) + (ct or 0))),
                        "cached_tokens": int(cc or 0),
                    }
                    # Extract reasoning_tokens from completion_tokens_details if present
                    completion_tokens_details = getattr(u, "completion_tokens_details", None)
    
                    if completion_tokens_details is not None:
                        reasoning = getattr(completion_tokens_details, "reasoning_tokens", None)
                        print(f"[DEBUG _extract_openai_response_usage] reasoning: {reasoning}")
                    if reasoning is not None:
                        out2["reasoning_tokens"] = int(reasoning)
                    return out2
            except Exception:
                return None
        return None

    def _extract_gemini_response_usage(self, resp: Any, endpoint: str) -> Optional[Dict[str, int]]:
        """Extract usage information from Gemini responses.
        
        Handles both Gemini chat completions (OpenAI-compatible) and Gemini native SDK responses.
        
        Args:
            resp: The response object from Gemini API
            endpoint: Either "chatcompletion" for OpenAI-compatible or "gemini_sdk" for native SDK
            
        Returns:
            Dictionary with usage metrics or None if not available
        """
        if endpoint == "chat_completions":
            # Gemini OpenAI-compatible chat completions
            try:
                u = getattr(resp, "usage", None)
                if u is None:
                    print(f"[DEBUG _extract_gemini_response_usage] chatcompletion - usage object is None")
                    return None
                print(f"[DEBUG _extract_gemini_response_usage] chatcompletion - usage object: {u}")
                print(f"[DEBUG _extract_gemini_response_usage] chatcompletion - usage type: {type(u)}")
                pt = u.get("prompt_tokens") if isinstance(u, dict) else getattr(u, "prompt_tokens", None)
                ct = u.get("completion_tokens") if isinstance(u, dict) else getattr(u, "completion_tokens", None)
                tt = u.get("total_tokens") if isinstance(u, dict) else getattr(u, "total_tokens", None)
                print(f"[DEBUG _extract_gemini_response_usage] chatcompletion - extracted pt: {pt}, ct: {ct}, tt: {tt}")

                #Extract cached_content_tokens from prompt_tokens_details if present
                cached_details = getattr(u, "prompt_tokens_details", None)
                cc = None
                if cached_details is not None:
                    cc = getattr(cached_details, "cached_tokens", None)
                
                if pt is not None or ct is not None or tt is not None:
                    print(f"[DEBUG _extract_gemini_response_usage] chatcompletion - condition passed, creating usage_dict")
                    usage_dict = {
                        "prompt_tokens": int(pt or 0),
                        "completion_tokens": int(ct or 0),
                        "total_tokens": int(tt or ((pt or 0) + (ct or 0))),
                        "cached_tokens": int(cc or 0),
                    }

                    # Extract reasoning_tokens from completion_tokens_details if present
                    reasoning = None
                    completion_tokens_details = getattr(u, "completion_tokens_details", None)
                    print(f"[DEBUG _extract_gemini_response_usage] chatcompletion - completion_tokens_details: {completion_tokens_details}")

                    if completion_tokens_details is not None:
                        reasoning = getattr(completion_tokens_details, "reasoning_tokens", None)
                        print(f"[DEBUG _extract_gemini_response_usage] reasoning: {reasoning}")

                    if reasoning is not None:
                        usage_dict["reasoning_tokens"] = int(reasoning)

                    try:
                        print(f"[DEBUG _extract_gemini_response_usage] chatcompletion final usage_dict: {usage_dict}")
                    except Exception as e:
                        print(f"[DEBUG _extract_gemini_response_usage] print exception: {e}")
                        print(f"[DEBUG _extract_gemini_response_usage] usage_dict type: {type(usage_dict)}")
                    return usage_dict
            except Exception:
                return None
                
        elif endpoint == "gemini_sdk":
            # Gemini native SDK response (google-genai). In Python this is typically `resp.usage_metadata`
            # with snake_case fields: prompt_token_count, candidates_token_count, thoughts_token_count, total_token_count.
            try:
                print(f"[DEBUG _extract_gemini_response_usage] gemini_sdk - endpoint: {endpoint} - resp type: {type(resp)}")

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
                    print(f"[DEBUG _extract_gemini_response_usage] gemini_sdk - usage metadata object is None")
                    print(f"[DEBUG _extract_gemini_response_usage] gemini_sdk - raw type: {type(raw)}")
                    return None

                print(f"[DEBUG _extract_gemini_response_usage] gemini_sdk - raw type: {type(raw)}")
                print(f"[DEBUG _extract_gemini_response_usage] gemini_sdk - usage metadata type: {type(um)}")

                def _um_get(name_snake: str, name_camel: str) -> Any:
                    if um is None:
                        return None
                    if isinstance(um, dict):
                        return um.get(name_snake) if name_snake in um else um.get(name_camel)
                    v = getattr(um, name_snake, None)
                    if v is None:
                        v = getattr(um, name_camel, None)
                    return v

                # Native SDK (google-genai) snake_case fields
                pt = _um_get("prompt_token_count", "promptTokenCount")
                ct = _um_get("candidates_token_count", "candidatesTokenCount")
                tt = _um_get("total_token_count", "totalTokenCount")
                reasoning = _um_get("thoughts_token_count", "thoughtsTokenCount")
                cc = _um_get("cached_content_token_count", "cachedContentTokenCount")

                print(f"[DEBUG _extract_gemini_response_usage] gemini_sdk - extracted pt: {pt}, ct: {ct}, tt: {tt}")
                print(f"[DEBUG _extract_gemini_response_usage] gemini_sdk - extracted cc: {cc}, reasoning: {reasoning}")

                if pt is not None or ct is not None or tt is not None:
                    usage_dict: Dict[str, int] = {
                        "prompt_tokens": int(pt or 0),
                        # Visible output tokens only (Gemini calls these candidates tokens)
                        "completion_tokens": int(ct or 0),
                        # Total includes prompt + candidates + thoughts
                        "total_tokens": int(tt or ((pt or 0) + (ct or 0) + (reasoning or 0))),
                        "reasoning_tokens": int(reasoning or 0),
                    }

                    # Keep a consistent cached token key across the adapter
                    if cc is not None:
                        usage_dict["cached_tokens"] = int(cc or 0)
                        # Back-compat alias (some callers may have expected this)
                        usage_dict["cached_content_tokens"] = int(cc or 0)
                    else:
                        usage_dict["cached_tokens"] = 0

                    print(f"[DEBUG _extract_gemini_response_usage] gemini_sdk final usage_dict: {usage_dict}")
                    return usage_dict
            except Exception as e:
                print(f"[DEBUG _extract_gemini_response_usage] gemini_sdk - exception: {e}")
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
        def _safe_get(obj: Any, name: str) -> Any:
            try:
                if isinstance(obj, dict):
                    return obj.get(name)
                return getattr(obj, name, None)
            except Exception:
                return None

        # Best-effort text extraction from chat.completions shape
        text = ""
        try:
            choices = _safe_get(resp, "choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                msg = _safe_get(c0, "message")
                content = _safe_get(msg, "content")
                if isinstance(content, str):
                    text = content
        except Exception:
            text = ""

        # Best-effort tool call extraction from chat.completions message.tool_calls
        tool_calls: list[Dict[str, Any]] = []
        try:
            choices = _safe_get(resp, "choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                msg = _safe_get(c0, "message")
                tc_list = _safe_get(msg, "tool_calls")
                if isinstance(tc_list, list):
                    for t in tc_list:
                        t_type = _safe_get(t, "type")
                        if t_type not in ("function", "tool_call"):
                            continue
                        func = _safe_get(t, "function") or t
                        name = _safe_get(func, "name")
                        args = _safe_get(func, "arguments")
                        cid = _safe_get(t, "id")
                        if isinstance(name, str) and name:
                            tool_calls.append({"name": name, "arguments": args, "id": cid, "type": "tool_call"})
        except Exception:
            tool_calls = []

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
        usage_dict = self._extract_gemini_response_usage(resp, "chat_completions")
        print(f"[DEBUG _wrap_gemini_chatcompletion_as_responses] usage dict: {usage_dict}")

        # Wrap as AdapterResponse (Responses-like shim)
        finish_reason = self._extract_finish_reason(resp)
        return AdapterResponse(
            output_text=text or "",
            model=resolved_model,
            usage=usage_dict,
            metadata=self._build_adapter_response_metadata(
                provider="gemini",
                model_key=model_key,
                resolved_model=resolved_model,
            ),
            adapter_response={"output": output_list},
            model_response=resp,
            status=self._map_completion_status_from_finish_reason(finish_reason),
            finish_reason=finish_reason,
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
        resolved_model = self._resolve_model_name(model)
        filtered_kwargs = self._filter_kwargs_by_capabilities(model, kwargs)
        mapped_kwargs = self._apply_openai_reasoning_policy(model, filtered_kwargs)

        # Canonical output token budget (set by create()); map to endpoint-specific param name later.
        mot = None
        mot = mapped_kwargs.pop("max_output_tokens", None)

        endpoint = "responses"
        try:
            model_info = self._lookup_model_info_from_registry(model)
            if model_info is not None:
                endpoint = getattr(model_info, "endpoint", None)
            else:
                endpoint = "chat_completions"
        except Exception:
            endpoint = "chat_completions"

        # Map canonical max_output_tokens to the OpenAI field name for this endpoint.
        if mot is not None:
            try:
                mot_i = int(mot)
            except Exception:
                mot_i = mot
            if endpoint == "responses":
                mapped_kwargs["max_output_tokens"] = mot_i
            elif endpoint == "chat_completions":
                mapped_kwargs["max_completion_tokens"] = mot_i

        if endpoint == "responses":
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
            print(f"[DEBUG _openai_call Responses] usage dict for AdapterResponse: {usage_dict}")
            return AdapterResponse(
                output_text=text,
                model=resolved_model,
                usage=usage_dict,
                metadata=self._build_adapter_response_metadata(
                    provider="openai",
                    model_key=model,
                    resolved_model=resolved_model,
                ),
                adapter_response=resp,
                model_response=resp,
                status=status or ("incomplete" if finish_reason else "completed"),
                finish_reason=finish_reason,
            )

        if endpoint == "chat_completions":
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
                print(f"[DEBUG _openai_call ChatCompletions] usage dict for AdapterResponse: {usage_dict}")
                return AdapterResponse(
                    output_text=self._extract_openai_chatcompletion_text(resp),
                    model=resolved_model,
                    usage=usage_dict,
                    metadata=self._build_adapter_response_metadata(
                        provider="openai",
                        model_key=model,
                        resolved_model=resolved_model,
                    ),
                    adapter_response=resp,
                    model_response=resp,
                    status=self._map_completion_status_from_finish_reason(finish_reason),
                    finish_reason=finish_reason,
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
        resolved_model = self._resolve_model_name(model)
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
        
        return EmbeddingResponse(
            data=vectors,  # Direct list of embedding vectors
            usage=EmbeddingUsage(
                prompt_tokens=(
                    getattr(raw_response.usage, 'input_tokens', 0) if hasattr(raw_response, 'usage') else 0 or
                    getattr(raw_response.usage, 'prompt_tokens', 0) if hasattr(raw_response, 'usage') else 0
                ),
                total_tokens=getattr(raw_response.usage, 'total_tokens', 0) if hasattr(raw_response, 'usage') else 0
            ),
            provider="openai",
            model=resolved_model,
            normalized=self._was_normalization_applied("openai", **kwargs),
            vector_dim=len(vectors[0]) if vectors else None,
            metadata=metadata,
            raw=raw_response
        )

    def _gemini_call(self, *, model: str, input: Any, stream: bool, skip_reasoning: bool = False, **kwargs: Any):
        resolved_model = self._resolve_model_name(model)
        working_kwargs = self._prepare_gemini_adapter_kwargs(model, kwargs)

        # Extract canonical output token budget (create() writes max_output_tokens).
        mot = None
        mot = working_kwargs.pop("max_output_tokens", None)

        endpoint = "chat_completions"
        try:
            model_info = self._lookup_model_info_from_registry(model)
            if model_info is not None:
                endpoint = getattr(model_info, "endpoint", "chat_completions") or "chat_completions"
        except Exception:
            endpoint = "chat_completions"

        if endpoint == "gemini_sdk":
            client = self._get_gemini_native()
        else:
            client = self._get_gemini()

        if endpoint == "gemini_sdk":
            native_client = client

            def _extract_native_text(resp: Any) -> str:
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

            def _extract_native_text_with_collapsed_thoughts(resp: Any) -> str:
                try:
                    candidates = getattr(resp, "candidates", None)
                    if not isinstance(candidates, list) or not candidates:
                        return _extract_native_text(resp)
                    cand = candidates[0]
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) if content is not None else None
                    if not isinstance(parts, list) or not parts:
                        return _extract_native_text(resp)

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
                    return _extract_native_text(resp)
                except Exception:
                    return _extract_native_text(resp)

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

                    def _clean_schema(obj: Any) -> Any:
                        if isinstance(obj, list):
                            return [_clean_schema(x) for x in obj]
                        if not isinstance(obj, dict):
                            return obj
                        forbidden = {"default", "title", "$schema", "additionalProperties", "additional_properties"}
                        out: Dict[str, Any] = {}
                        for k, v in obj.items():
                            if k in forbidden:
                                continue
                            out[k] = _clean_schema(v)
                        return out

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
                        params = _clean_schema(params)
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

                text = _extract_native_text_with_collapsed_thoughts(resp)
                usage_dict = self._extract_gemini_response_usage(resp, "gemini_sdk")
                print(f"[DEBUG _gemini_call Native SDK] usage dict for AdapterResponse: {usage_dict}")

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

                output_list = [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    }
                ]

                wrapper = _GeminiSDKResponsesWrapper(
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

                return AdapterResponse(
                    output_text=text,
                    model=resolved_model,
                    usage=usage_dict,
                    metadata=self._build_adapter_response_metadata(
                        provider="gemini",
                        model_key=model,
                        resolved_model=resolved_model,
                    ),
                    adapter_response=wrapper,
                    model_response=resp,
                    status=self._map_gemini_native_status_from_finish_reason(finish_reason),
                    finish_reason=finish_reason,
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
                            txt = _extract_native_text(chunk)
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
        resolved_model = self._resolve_model_name(model)

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
        resolved_model = self._resolve_model_name(model)

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
