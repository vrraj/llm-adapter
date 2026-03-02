# Release Notes

## Version 1.0.0 — Initial Public Release

### Overview

`vrraj-llm-adapter` provides a unified, registry-driven interface for LLM text generation and embeddings across supported providers (currently OpenAI and Gemini), with an extensible architecture for additional providers.

This is the first public release. The complete API surface is documented in [docs/API_REFERENCE.md](https://github.com/vrraj/llm-adapter/blob/main/docs/API_REFERENCE.md).

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
- Per-model token pricing metadata (see [docs/API_REFERENCE.md](https://github.com/vrraj/llm-adapter/blob/main/docs/API_REFERENCE.md) for field details)

---

## Documentation Structure

- **[README.md](https://github.com/vrraj/llm-adapter/blob/main/README.md)** — Quick start and high-level overview
- **[docs/API_REFERENCE.md](https://github.com/vrraj/llm-adapter/blob/main/docs/API_REFERENCE.md)** — Complete method signatures and response contracts
- **[docs/MODEL_REGISTRY.md](https://github.com/vrraj/llm-adapter/blob/main/docs/MODEL_REGISTRY.md)** — Registry architecture and extension guide
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