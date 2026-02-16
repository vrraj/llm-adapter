from typing import Any, Dict, Optional
import os

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import get_model_options, is_provider_enabled
from llm_adapter import llm_adapter, LLMError


def _ensure_handler_has_api_key(provider: str) -> None:
    provider = (provider or "").lower().strip()
    if provider == "openai":
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key and not getattr(llm_adapter, "openai_api_key", None):
            llm_adapter.openai_api_key = env_key
            if hasattr(llm_adapter, "_openai"):
                llm_adapter._openai = None
    if provider == "gemini":
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key and not getattr(llm_adapter, "gemini_api_key", None):
            llm_adapter.gemini_api_key = env_key
            if hasattr(llm_adapter, "_gemini"):
                llm_adapter._gemini = None
            if hasattr(llm_adapter, "_gemini_native"):
                llm_adapter._gemini_native = None


class ChatRequest(BaseModel):
    model_key: str
    prompt: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    max_output_tokens: Optional[int] = None
    stream: Optional[bool] = None


class ChatError(BaseModel):
    provider: Optional[str]
    model: Optional[str]
    kind: str
    code: Optional[str] = None
    message: str
    retry_after: Optional[float] = None


class ChatResponse(BaseModel):
    ok: bool
    answer_text: Optional[str] = None
    reasoning_text: Optional[str] = None
    raw_usage: Optional[Dict[str, Any]] = None
    provider_response: Optional[Any] = None
    normalized_result: Optional[Dict[str, Any]] = None
    provider_request: Optional[Dict[str, Any]] = None
    format_request: Optional[Dict[str, Any]] = None
    error: Optional[ChatError] = None


class EmbedRequest(BaseModel):
    model_key: str
    text: str
    normalize_embedding: Optional[bool] = None


class EmbedResponse(BaseModel):
    ok: bool
    embedding: Optional[list[float]] = None
    dimension: Optional[int] = None
    raw_usage: Optional[Dict[str, Any]] = None
    provider_response: Optional[Any] = None
    normalized_result: Optional[Dict[str, Any]] = None
    provider_request: Optional[Dict[str, Any]] = None
    format_request: Optional[Dict[str, Any]] = None
    error: Optional[ChatError] = None


app = FastAPI(title="llm-adapter demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/models")
async def get_models() -> Dict[str, Any]:
    options = get_model_options()
    for k, v in options.items():
        try:
            prov = str(v.get("provider") or "")
            v["enabled"] = is_provider_enabled(prov)
        except Exception:
            v["enabled"] = False
    return {"models": options}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    model_key = req.model_key.strip()
    options = get_model_options()
    mi = options.get(model_key)
    if not mi:
        raise HTTPException(status_code=400, detail=f"Unknown model_key: {model_key}")

    provider = str(mi.get("provider") or "").lower().strip()
    endpoint = str(mi.get("endpoint") or "")

    _ensure_handler_has_api_key(provider)

    if not is_provider_enabled(provider):
        return ChatResponse(
            ok=False,
            error=ChatError(
                provider=provider,
                model=model_key,
                kind="config",
                code="provider_disabled",
                message=f"Provider '{provider}' is not enabled. Check your environment variables.",
                retry_after=None,
            ),
        )

    if endpoint in ("embeddings", "embed_content"):
        return ChatResponse(
            ok=False,
            error=ChatError(
                provider=provider,
                model=model_key,
                kind="request",
                code="wrong_endpoint",
                message="Selected model is an embedding model. Use /api/embed instead.",
                retry_after=None,
            ),
        )

    # Client-level request preview (what the caller passes into llm_adapter).
    provider_request: Dict[str, Any] = {
        "call": "llm_adapter.create",
        "params": {
            "model": model_key,
            "input": [{"role": "user", "content": req.prompt}],
            "stream": bool(req.stream),
            "temperature": req.temperature,
            "top_p": req.top_p,
            "reasoning_effort": req.reasoning_effort,
            "max_output_tokens": req.max_output_tokens,
        },
    }
    try:
        params = provider_request.get("params")
        if isinstance(params, dict):
            provider_request["params"] = {k: v for k, v in params.items() if v is not None}
    except Exception:
        provider_request = provider_request

    try:
        resp = llm_adapter.create(
            model=model_key,
            input=[{"role": "user", "content": req.prompt}],
            stream=bool(req.stream),
            temperature=req.temperature,
            top_p=req.top_p,
            reasoning_effort=req.reasoning_effort,
            max_output_tokens=req.max_output_tokens,
        )

        normalized = llm_adapter.build_llm_result_from_response(resp, provider=provider)
        return ChatResponse(
            ok=True,
            answer_text=normalized.get("text"),
            reasoning_text=normalized.get("reasoning"),
            raw_usage=normalized.get("usage"),
            provider_response=jsonable_encoder(resp),
            normalized_result=jsonable_encoder(llm_adapter.build_llm_result_from_response(resp, provider=provider)),
            provider_request=jsonable_encoder(provider_request),
            format_request=jsonable_encoder(
                {
                    "calls": [
                        {
                            "call": "llm_adapter.build_llm_result_from_response",
                            "params": {
                                "resp": "<adapter_response attribute from llm_adapter.create()>",
                                "provider": provider,
                            },
                        },
                    ]
                }
            ),
        )

    except LLMError as e:
        return ChatResponse(
            ok=False,
            error=ChatError(
                provider=e.provider,
                model=e.model,
                kind=e.kind,
                code=str(e.code) if e.code is not None else None,
                message=str(e),
                retry_after=e.retry_after,
            ),
        )
    except Exception as e:  # pragma: no cover - defensive
        return ChatResponse(
            ok=False,
            error=ChatError(
                provider=None,
                model=model_key,
                kind="exception",
                code=e.__class__.__name__,
                message=str(e),
                retry_after=None,
            ),
        )


@app.post("/api/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    model_key = req.model_key.strip()
    options = get_model_options()
    mi = options.get(model_key)
    if not mi:
        raise HTTPException(status_code=400, detail=f"Unknown model_key: {model_key}")

    provider = str(mi.get("provider") or "").lower().strip()
    endpoint = str(mi.get("endpoint") or "")

    _ensure_handler_has_api_key(provider)

    if not is_provider_enabled(provider):
        return EmbedResponse(
            ok=False,
            error=ChatError(
                provider=provider,
                model=model_key,
                kind="config",
                code="provider_disabled",
                message=f"Provider '{provider}' is not enabled. Check your environment variables.",
                retry_after=None,
            ),
        )

    if endpoint not in ("embeddings", "embed_content"):
        return EmbedResponse(
            ok=False,
            error=ChatError(
                provider=provider,
                model=model_key,
                kind="request",
                code="wrong_endpoint",
                message="Selected model is not an embedding model.",
                retry_after=None,
            ),
        )

    # Client-level request preview (what the caller passes into llm_adapter).
    provider_request: Dict[str, Any] = {
        "call": "llm_adapter.create_embedding",
        "params": {
            "model": model_key,
            "input": req.text,
            "normalize_embedding": bool(req.normalize_embedding) if req.normalize_embedding is not None else None,
        },
    }
    try:
        params = provider_request.get("params")
        if isinstance(params, dict):
            provider_request["params"] = {k: v for k, v in params.items() if v is not None}
    except Exception:
        provider_request = provider_request

    try:
        kwargs: Dict[str, Any] = {}
        if provider == "gemini" and req.normalize_embedding is not None:
            kwargs["normalize_embedding"] = bool(req.normalize_embedding)

        resp = llm_adapter.create_embedding(
            model=model_key,
            input=req.text,
            **kwargs,
        )

        vec = None
        dim = None
        try:
            if hasattr(resp, "data") and resp.data:
                vec = getattr(resp.data[0], "embedding", None)
                if isinstance(vec, list):
                    dim = len(vec)
        except Exception:
            vec = None
            dim = None

        usage_obj = getattr(resp, "usage", None)
        usage = None
        if usage_obj is not None:
            try:
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }
            except Exception:
                usage = None

        return EmbedResponse(
            ok=True,
            embedding=vec,
            dimension=dim,
            raw_usage=usage,
            provider_response=jsonable_encoder(getattr(resp, "model_response", None) or resp),
            normalized_result=None,
            provider_request=jsonable_encoder(provider_request),
            format_request=jsonable_encoder(
                {
                    "calls": [],
                    "note": "Embeddings do not call build_llm_result/build_llm_result_from_response; normalized_result omitted.",
                }
            ),
        )

    except LLMError as e:
        return EmbedResponse(
            ok=False,
            error=ChatError(
                provider=e.provider,
                model=e.model,
                kind=e.kind,
                code=str(e.code) if e.code is not None else None,
                message=str(e),
                retry_after=e.retry_after,
            ),
        )

    except Exception as e:  # pragma: no cover
        return EmbedResponse(
            ok=False,
            error=ChatError(
                provider=None,
                model=model_key,
                kind="exception",
                code=e.__class__.__name__,
                message=str(e),
                retry_after=None,
            ),
        )


# Serve the static test UI from /ui and root
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "llm-adapter demo running. Open /ui/ in a browser for the test UI."}
