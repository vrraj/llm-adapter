# Release Notes

## Version 1.2.0 — NVIDIA NIM Embeddings & Structured Provider Errors

### Overview

Adds NVIDIA NIM embedding support and normalizes provider-side HTTP errors into the
structured `LLMError` type across all providers.

### NVIDIA Embeddings

- `create_embedding()` now supports the `nvidia` provider via the NIM
  OpenAI-compatible `/embeddings` endpoint
- New registry model: `nvidia:nemotron-3-embed-1b` (2048 dimensions)
- NIM-specific parameters `input_type` (`"query"` / `"passage"`) and `truncate`
  are forwarded through `extra_body`
- `normalize_embedding=True` applies client-side L2 normalization

### Structured Provider Error Handling

- `create()` and `create_embedding()` now translate provider SDK/HTTP errors
  into `LLMError` instead of leaking raw SDK exception types
- Mapping: `401/403 → auth`, `404/410 → model_not_found`, `408 → timeout`,
  `429 → rate_limit` (with `retry_after` from response headers),
  `5xx → provider_error`, connection failures → `network`
- The original SDK exception is preserved via exception chaining (`__cause__`)
- Non-SDK errors (programming bugs) continue to propagate unwrapped

### Compatibility Note

Error paths now raise `LLMError` rather than raw `openai`/`google.genai`
exceptions. Callers catching generic `Exception` or `LLMError` are unaffected;
callers relying on SDK-specific exception types (e.g. `except openai.NotFoundError`)
should catch `LLMError` and inspect `kind`/`code` instead. Mid-stream errors
raised during iteration are not wrapped.

---

## Version 1.1.0 — NVIDIA NIM (Nemotron & DeepSeek) Provider

### Overview

Adds **NVIDIA** as a third provider alongside OpenAI and Gemini, served through NIM's OpenAI-compatible endpoint — no new SDK dependency. Fully backward compatible; OpenAI and Gemini behavior is unchanged.

### New Provider: NVIDIA NIM

- New `nvidia` provider via `https://integrate.api.nvidia.com/v1` (OpenAI-compatible)
- `nvidia_api_key` / `nvidia_base_url` / `nvidia_client` constructor params with `NVIDIA_API_KEY` / `NVIDIA_BASE_URL` environment fallbacks
- `NVIDIA_BASE_URL` override supports self-hosted NIM containers
- Registry models (all verified against the hosted API):
  - `nvidia:nemotron-3-super-120b` — hybrid reasoning, budget-based
  - `nvidia:nemotron-3-ultra-550b` — hybrid reasoning, thinking toggle
  - `nvidia:nemotron-3-nano-omni-30b` — lightweight reasoning
  - `nvidia:nemotron-3.5-lightning-30b` — budget-based reasoning
  - `nvidia:deepseek-v4.1-flash` — DeepSeek V4.1 Flash, thinking toggle

### Reasoning Support

- Standard `reasoning_effort` knob (none / minimal / low / medium / high) mapped to NIM's `chat_template_kwargs.enable_thinking` and `reasoning_budget` parameters
- Two registry reasoning modes: `nvidia_budget` (token budgets) and `nvidia_toggle` (for model runners that reject budget parameters)
- NVIDIA hosted reasoning models think by default; `reasoning_effort="none"` disables thinking
- Response `reasoning_content` is surfaced as the normalized `reasoning` field in `LLMResult`

### Demo UI

- NVIDIA models appear in the Interactive Playground dropdown; provider enablement follows `NVIDIA_API_KEY`
- Demo server port moved from 8100 to 7100
- `make start` now runs the server in the background, waits for readiness, and prints the UI URL; foreground mode moved to `make fg`

### Compatibility

- Python 3.10+
- OpenAI, Gemini, and NVIDIA supported
- Custom registry extensions supported (NVIDIA models flow into merged registries automatically)
- Stable 1.x API contract maintained

---

## Version 1.0.0 — Initial Public Release

### Overview

`vrraj-llm-adapter` provides a unified, registry-driven interface for LLM text generation and embeddings across supported providers (currently OpenAI and Gemini), with an extensible architecture for additional providers.

This is the first public release. The complete API surface is documented in [docs/api-reference.md](https://github.com/vrraj/llm-adapter/blob/main/docs/api-reference.md).

---

## Core Capabilities

### Text Generation
- Primary `.create()` entry point (paired with `normalize_adapter_response()` for stable app-facing output)
- Registry-driven provider and endpoint resolution
- OpenAI (`responses`, `chat_completions`) support
- Gemini (OpenAI-compatible and native SDK) support
- Streaming via `AdapterEvent` iterator
- Structured tool-calling support

### Embeddings
- Unified `.create_embedding()` entry point
- Cross-provider response normalization
- Optional vector normalization
- Embedding metadata (dimension, magnitude, usage)

### Response Contract
- `AdapterResponse` as the explicit provider-boundary object (raw + normalized metadata)
- `normalize_adapter_response()` → stable, provider-agnostic `LLMResult` schema
- Explicit separation of reasoning/thought traces from display-safe `text`
- Deterministic normalization of tool calls and usage accounting

### Model Registry
- Centralized `ModelInfo` definitions
- Explicit endpoint semantics and capability metadata
- Per-model pricing metadata
- Strict parameter validation via `param_policy`
- Support for custom user-defined registry extensions

### Parameter Governance
- Registry-controlled `allowed` / `disabled` parameter policies
- Cross-provider parameter isolation
- Silent filtering of incompatible arguments
- Prevents runtime API errors due to invalid parameters

### Pricing Support
- `get_pricing_for_model()` helper
- Per-model token pricing metadata (see [docs/api-reference.md](https://github.com/vrraj/llm-adapter/blob/main/docs/api-reference.md) for field details)

---

## Documentation Structure

- **[README.md](https://github.com/vrraj/llm-adapter/blob/main/README.md)** — Quick start and high-level overview
- **[docs/api-reference.md](https://github.com/vrraj/llm-adapter/blob/main/docs/api-reference.md)** — Complete method signatures and response contracts
- **[docs/model-registry.md](https://github.com/vrraj/llm-adapter/blob/main/docs/model-registry.md)** — Registry architecture and extension guide
- **[examples/README.md](https://github.com/vrraj/llm-adapter/blob/main/examples/README.md)** — Structured learning paths and usage patterns
- **[ReleaseNotes.md](https://github.com/vrraj/llm-adapter/blob/main/ReleaseNotes.md)** — Version history

---

## Public API Surface

Stable entry points:
- `llm_adapter.create(...)`
- `llm_adapter.normalize_adapter_response(...)`
- `llm_adapter.create_embedding(...)`
- `llm_adapter.get_pricing_for_model(...)`

Stable response contracts:
- `AdapterResponse`
- `LLMResult`
- `EmbeddingResponse`
- `AdapterEvent`
- `LLMError`

Fields explicitly marked as debug/opaque in the API reference are not part of the guaranteed stability surface.

---

## Compatibility

- Python 3.10+
- OpenAI and Gemini supported
- Custom registry extensions supported

---

## Notes

This release establishes the stable 1.x API contract for `vrraj-llm-adapter`.
Backward compatibility will be maintained within the 1.x series.