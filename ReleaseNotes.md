# Release Notes

## Version 1.0.0 — Initial Public Release

### Overview

`vrraj-llm-adapter` provides a unified, registry-driven interface for LLM text generation and embeddings across multiple providers.

This is the first public release. The API surface defined in `API_REFERENCE.md` is considered stable moving forward.

---

## Core Capabilities

### Unified Generation API
- Single `.create()` method for text generation
- Registry-driven provider resolution
- OpenAI (`responses` and `chat_completions`) support
- Gemini (OpenAI-compatible and native SDK) support
- Streaming support via event iterator
- Tool calling support

### Unified Embeddings API
- Single `.create_embedding()` method
- Cross-provider normalization
- Optional embedding normalization
- Embedding metadata and magnitude support

### Response Normalization
- `AdapterResponse` as the canonical provider return object
- `normalize_adapter_response()` → stable `LLMResult` schema
- Thought/reasoning extraction
- Tool call normalization
- Unified usage accounting

### Model Registry Architecture
- Centralized `ModelInfo` definitions
- Explicit endpoint semantics
- Pricing metadata per model
- Capability metadata
- Strict parameter validation (`param_policy`)
- Support for custom user-defined registries

### Parameter Validation System
- Registry-controlled `allowed` / `disabled` parameter policies
- Cross-provider parameter isolation
- Silent filtering of invalid parameters
- Prevents API failures due to incompatible arguments

### Pricing Metadata
- `get_pricing_for_model()` helper
- Per-million token pricing fields:
  - `input_per_mm`
  - `output_per_mm`
  - `cached_input_per_mm`

---

## Documentation Structure

- **README.md** — Quick start and high-level overview
- **API_REFERENCE.md** — Complete method signatures and response contracts
- **MODEL_REGISTRY.md** — Registry architecture and extension guide
- **examples/README.md** — Structured learning paths and usage patterns
- **ReleaseNotes.md** — Version history

---

## Public API Surface

Stable entry points:
- `llm_adapter.create(...)`
- `llm_adapter.create_embedding(...)`
- `llm_adapter.normalize_adapter_response(...)`
- `llm_adapter.get_pricing_for_model(...)`

Stable response contracts:
- `AdapterResponse`
- `EmbeddingResponse`
- `LLMResult`
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
Future minor versions (1.1.0, 1.2.0, etc.) will introduce additive features without breaking the documented public API.