# vrraj-llm-adapter

> **Development and Demo UI:**  
> This repository ships with a FastAPI-powered **Interactive Playground** for validating text generation, embeddings, and registry configuration end-to-end. See **[Development And Demo UI](#development-and-demo-ui)** below for details and setup instructions.

![CI Status](https://github.com/vrraj/llm-adapter/actions/workflows/ci.yml/badge.svg)

Provider-agnostic LLM adapter for **text generation + embeddings** with a **registry-driven routing layer** (capabilities, param policies, pricing metadata, access control), plus **normalized outputs** (text, tool calls, reasoning, usage).

Currently supports OpenAI and Gemini (extensible architecture for additional providers).

- **PyPI:  ** https://pypi.org/project/vrraj-llm-adapter
- **GitHub:** https://github.com/vrraj/llm-adapter


## Install

```bash
pip install vrraj-llm-adapter
```




## What you get

- **One interface** for generation + embeddings across providers
- **Registry-driven routing (default + extensible)** — ships with built-in model keys and supports **custom** registry extensions
- **Parameter policies** (allowed/disabled filtering per model)
- **Normalized responses** (text, tool calls, reasoning, usage)
- **Model Allowlist** (access control)
- **Pricing metadata** in registry for cost visibility




## Quickstart

The examples below use a registry model key for `model=` (for example: `openai:gpt-4o-mini`, `gemini:openai-3-flash-preview`). For a complete list of default model keys, see [MODEL_REGISTRY.md](https://github.com/vrraj/llm-adapter/blob/main/docs/MODEL_REGISTRY.md) or print keys programmatically (snippet below)

### Option A: Run a ready-to-use example script
Download and run a ready-to-use example script for text generation and embeddings for openai and gemini

**Requires API keys** (OPENAI_API_KEY and/or GEMINI_API_KEY).

```bash
curl -L -O https://raw.githubusercontent.com/vrraj/llm-adapter/main/examples/llm_adapter_basic_usage.py

python llm_adapter_basic_usage.py
```


### Option B: Call the API directly

**Requires API keys** (OPENAI_API_KEY and/or GEMINI_API_KEY).

```python
from llm_adapter import llm_adapter

resp = llm_adapter.create(
    model="openai:gpt-4o-mini", # for gemini, use "gemini:openai-3-flash-preview"
    input="Write a one-sentence bedtime story about a unicorn.",
    max_output_tokens=100,
)

# Normalize to stable app-facing schema
result = llm_adapter.normalize_adapter_response(resp)

print(result["text"])
print(result["usage"])
```

### Discover available model keys

The package ships with a default registry. To list available keys:

```python
from llm_adapter import LLMAdapter

adapter = LLMAdapter()
for key in sorted(adapter.model_registry.keys()):
    print(key)
```




## Interactive Playground (GitHub)

The repo includes a small FastAPI demo + UI to try models, inspect registry metadata, and view normalized responses.

The source includes developer tooling to test **custom model registries** (overrides/extensions) end-to-end in the UI. See **[Development And Demo UI](#development-and-demo-ui)** section below.

![LLM Adapter Interactive Playground](https://github.com/vrraj/llm-adapter/blob/main/images/llm_adapter_interactive_playground.png)


## Public API (overview)

- `llm_adapter.create(...) -> AdapterResponse` — text generation (supports tools + optional streaming)
- `llm_adapter.normalize_adapter_response(...) -> LLMResult` — normalize `AdapterResponse` into a consistent dict schema
- `llm_adapter.create_embedding(...) -> EmbeddingResponse` — create embeddings
- `llm_adapter.get_pricing_for_model(...) -> Pricing | None` — pricing metadata lookup

>📋 For **complete method signatures, parameter details, and full response structures**, see: [API_REFERENCE.md](https://github.com/vrraj/llm-adapter/blob/main/docs/API_REFERENCE.md)

### AdapterResponse (from `create`)

Top-level fields (stable surface; note: `output_text` may include provider thought markup for some Gemini paths):

```python
AdapterResponse(
  output_text: str,
  model: str,
  usage: dict,
  status: str,
  finish_reason: str | None,
  tool_calls: list | None,
  metadata: dict | None,
  adapter_response: Any | None,  # debug/opaque
  model_response: Any | None,    # debug/opaque
)
```

### EmbeddingResponse (from `create_embedding`)

Top-level fields:

```python
EmbeddingResponse(
  data: List[List[float]],
  usage: EmbeddingUsage,
  normalized: bool | None,
  vector_dim: int | None,
  metadata: dict | None,
  raw: Any | None,
)
```

### LLMResult (from `normalize_adapter_response`)

Top-level fields:

```python
{
  "text": str,
  "reasoning": str | None,
  "role": str,
  "status": str,
  "finish_reason": str | None,
  "usage": dict,
  "tool_calls": list,
  "metadata": dict | None,
  "raw": Any,
}
```

### Recommended flow (create → normalize)

The adapter intentionally separates the **provider boundary** from your app-facing schema:

```text
User Input
   │
   ▼
llm_adapter.create(...)  ─────────────►  AdapterResponse
   │                                  (provider-aware: raw responses, metadata)
   │
   ▼
llm_adapter.normalize_adapter_response(resp)  ─►  LLMResult
                                          (stable dict schema for apps)

Notes:
- `create()` performs the network call.
- `normalize_adapter_response()` is a local transform (no additional provider request).
```

Normalize to `LLMResult` for stable, application-facing output.
Use `result["text"]` from `normalize_adapter_response()` for display-safe text; `resp.output_text` may include provider thought markup depending on model configuration.



## Documentation & References

- **Complete API Reference:** [API_REFERENCE.md](https://github.com/vrraj/llm-adapter/blob/main/docs/API_REFERENCE.md)
- **Model Registry docs:** [MODEL_REGISTRY.md](https://github.com/vrraj/llm-adapter/blob/main/docs/MODEL_REGISTRY.md)
- **Ready to use Examples:** [examples](https://github.com/vrraj/llm-adapter/tree/main/examples)
- **Dev notes:** [DEVELOPMENT.md](https://github.com/vrraj/llm-adapter/blob/main/docs/DEVELOPMENT.md)

---


## Usage Examples (PyPI)

Install the adapter from PyPI, then download and run the standalone example scripts to explore common usage patterns such as chat, embeddings, streaming, and custom registry overrides.

### Text Generation - Application Wrapper Pattern

Some applications prefer a one-step helper that standardizes on `LLMResult` internally:

```python
from llm_adapter import llm_adapter


def create_result(**kwargs):
    resp = llm_adapter.create(**kwargs)
    return llm_adapter.normalize_adapter_response(resp)

result = create_result(
    model="openai:gpt-4o-mini",
    input="Hello"
)

print(result["text"])
```

This pattern keeps the library surface minimal while allowing your application to standardize on the normalized contract.


**Core Examples:**
-  llm_adapter_basic_usage.py - Basic usage and normalization
-  create_and_normalize_example.py - Recommended create → normalize flow (Gemini-safe)
-  llm_adapter_model_spec_example.py - ModelSpec configuration

**Provider-Specific Examples:**
-  openai_embedding_example.py - OpenAI embeddings
-  openai_adapter_example.py - OpenAI chat
-  streaming_call_example.py - Streaming responses

**Advanced Examples:**
-  set_adapter_allowed_models.py - Allowlist demo
   *(See "Model Allowlist (Access Control)" section for environment variable details)*
-  custom_registry.py - Custom registry

### Text Generation

For application-facing output, use the create → normalize flow (see **Text Generation - Application Wrapper Pattern** above).
If you need the raw provider boundary object for debugging, `llm_adapter.create(...)` returns an `AdapterResponse`.

### Accessing Reasoning Content

Some models (like Gemini) return reasoning content separately.

```python
from llm_adapter import llm_adapter, LLMError

try:
    response = llm_adapter.create(
        model="gemini:native-sdk-reasoning-2.5-flash",
        input="Explain why the sky is blue",
        reasoning_effort="high",   # adapter-level reasoning knob
        max_output_tokens=1000
    )

    normalized_response = llm_adapter.normalize_adapter_response(response)

    if normalized_response.get('reasoning'):
        print(f"Reasoning: {normalized_response['reasoning']}")

    print(normalized_response['text'])
except LLMError as e:
    print(f"Error: {e.code} - {e}")
```

### Streaming

```python
from llm_adapter import llm_adapter

for event in llm_adapter.create(model="openai:gpt-4o-mini", input="Hello", stream=True):
    if event.type == "output_text.delta":
        print(event.delta, end="")
```

### Model Registry & Extensibility

The LLM adapter uses a registry of model definitions (ModelInfo) that control:
- Provider routing
- Endpoint selection
- Parameter policies (allowed/disabled)
- Pricing and limits
- Capabilities (reasoning, tools, dimensions, etc.)

You can override or extend the registry by passing your own mapping to `LLMAdapter(...)`.

```python
from llm_adapter.model_registry import ModelInfo, validate_registry
from llm_adapter import ModelSpec
```

### Example Custom Registry

```python
from llm_adapter import LLMAdapter
from llm_adapter.model_registry import ModelInfo, Pricing

custom_registry = {
    "my-openai-model": ModelInfo(
        provider="openai",
        model="gpt-4o-mini",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.05, output_per_mm=0.15),
        param_policy={"allowed": {"temperature", "max_tokens"}},
        limits={"max_output_tokens": 1000}
    )
}

adapter = LLMAdapter(model_registry=custom_registry)
```

### Model Allowlist

```bash
export LLM_ADAPTER_ALLOWED_MODELS="openai:gpt-4o-mini,openai:embed_small"
```

**For comprehensive registry documentation, see:**
- https://github.com/vrraj/llm-adapter/blob/main/docs/MODEL_REGISTRY.md
- https://github.com/vrraj/llm-adapter/blob/main/examples/custom_registry.py
- https://github.com/vrraj/llm-adapter/blob/main/src/llm_adapter/model_registry.py

### Validate Custom Registry

```python
from llm_adapter.model_registry import validate_registry
validate_registry(custom_registry, strict=False)
```

### Embeddings

```python
from llm_adapter import llm_adapter, LLMError

try:
    response = llm_adapter.create_embedding(
        model="openai:embed_small",
        input="The quick brown fox jumps over the lazy dog"
    )
    print(f"Generated {len(response.data)} embeddings")
    print(f"First embedding dimension: {len(response.data[0])}")
except LLMError as e:
    print(f"Error: {e.code} - {e}")
```

## Development And Demo UI

Running the **demo UI** (runs on port 8100) and/or **customize** the code.
The package includes a **Makefile** with quick start, stop and other helpful commands.

1. Clone the repository and run the setup script.

```bash
git clone https://github.com/vrraj/llm-adapter.git
cd llm-adapter
bash scripts/llm_adapter_setup.sh
```

>This quick setup script (scripts/llm_adapter_setup.sh) checks prerequisites (`python3`, `make`), creates `.env` if missing, sets up a local `.venv`, installs the package (`pip install -e .`), and shows **next steps**. The demo UI and FastAPI server run in this `.venv` virtual environment. Safe to run multiple times.

2. Set required API keys (see **Environment variables** section below).

3. Start the application.

```bash
make start
```

>**Note:** Run `make start` to run in foreground or `make start-bg` to run in background. Use `make stop` to stop the server.

4. Open the demo UI:

- http://localhost:8100/ui/


### Manual start (optional)

If you prefer not to use the Makefile helpers, you can start the FastAPI server directly:

```bash
uvicorn llm_adapter_demo.api:app --reload --port 8100
```


### For Developers: Running Tests

#### Install Dev dependencies

```bash
pip install -e ".[dev]"
```

#### Run Tests

```bash
pytest
pytest -m integration
pytest -m "integration or unit"
```

## Project structure

For internal design and architecture notes, see [DEVELOPMENT.md](https://github.com/vrraj/llm-adapter/blob/main/docs/DEVELOPMENT.md).

## ModelSpec: Structured Configuration

`ModelSpec` provides a type-safe, reusable way to configure model parameters as an alternative to passing individual parameters.

>**Note**: See `examples/llm_adapter_model_spec_example.py` for a comprehensive example demonstrating ModelSpec usage with different providers and parameter configurations.

### Using ModelSpec

```python
from llm_adapter import llm_adapter
from llm_adapter import ModelSpec

chat_spec = ModelSpec(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,
    max_output_tokens=1000,
    extra={"custom_param": "value"}
)

resp1 = llm_adapter.create(spec=chat_spec, input=[{"role": "user", "content": "Hello"}])
resp2 = llm_adapter.create(spec=chat_spec, input=[{"role": "user", "content": "How are you?"}])

embed_spec = ModelSpec(
    provider="openai",
    model="embed_small"
)
resp = llm_adapter.create_embedding(spec=embed_spec, input="Text to embed")
```

### ModelSpec vs Individual Parameters

| Approach | Provider | Model Name | Auto-detection | Type Safety |
|----------|----------|------------|----------------|-------------|
| **Individual params** | Optional (auto-detected from registry) | Registry key (`openai:gpt-4o-mini`) | ✅ Yes | ❌ Runtime |
| **ModelSpec** | Required (explicit) | Provider-native (`gpt-4o-mini`) | ❌ No | ✅ Static type-checkers |


## Unified Token Accounting

LLMAdapter returns a consistent usage schema across all providers:

### Usage Schema

```json
{
  "prompt_tokens": 0,
  "cached_tokens": 0,
  "output_tokens": 0,
  "reasoning_tokens": 0,
  "answer_tokens": 0,
  "total_tokens": 0
}
```

**Key relationships:**
- `output_tokens = answer_tokens + reasoning_tokens`
- `total_tokens = prompt_tokens + cached_tokens + output_tokens`

## Environment variables

Copy `.env.example` to `.env` and to set up your API keys (or use your existing environment variables):

```bash
cp .env.example .env
```

Supported env vars:

**Minimal working sets:**
- **OpenAI-only**: `OPENAI_API_KEY`
- **Gemini native SDK**: `GEMINI_API_KEY`
- **Gemini OpenAI-compatible**: `GEMINI_API_KEY` + `GEMINI_OPENAI_BASE_URL`

**All supported variables:**
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `GEMINI_OPENAI_BASE_URL`
- `LLM_ADAPTER_ALLOWED_MODELS` (comma-separated list) - Restrict which models can be used in each environment.

## Model Allowlist

The `LLM_ADAPTER_ALLOWED_MODELS` environment variable allows you to restrict which models can be used. *By default, all models are allowed*.

```bash
export LLM_ADAPTER_ALLOWED_MODELS="openai:gpt-4o-mini,gemini:native-sdk-reasoning-2.5-flash"
```


## Supported Providers

Supports:
- **OpenAI** (Responses API, Chat Completions API, Embeddings API)
- **Gemini** (native `google-genai` SDK and OpenAI-compatible endpoint)

Models and capabilities are defined in `src/llm_adapter/model_registry.py`.

## Adding New Models

To add support for new models or override existing configurations, use **custom registries** rather than modifying the core registry:

1. **Create a custom registry** - See `examples/custom_registry.py` for a complete example
2. **Define ModelInfo entries** - Configure endpoints, capabilities, pricing, and parameter policies
3. **Load your registry** - Pass it to `LLMAdapter(model_registry=your_registry)` or merge with defaults
4. **Test via Demo UI** - The Interactive Playground supports custom registry testing

📖 **For complete custom registry documentation**, see:
- [MODEL_REGISTRY.md - Custom Registry](https://github.com/vrraj/llm-adapter/blob/main/docs/MODEL_REGISTRY.md#custom-registry)
- [examples/custom_registry.py](https://github.com/vrraj/llm-adapter/blob/main/examples/custom_registry.py) - Working example

## Development

This is a standalone package. Development happens directly in this repo.

```bash
pip install -e .
make start
```

## License

This project is licensed under the MIT License.

