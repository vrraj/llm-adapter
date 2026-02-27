# vrraj-llm-adapter

![CI Status](https://github.com/vrraj/llm-adapter/actions/workflows/ci.yml/badge.svg)

Provider-agnostic LLM adapter for **text generation + embeddings** with a **registry-driven routing layer** (capabilities, param policies, pricing metadata, access control), plus **normalized outputs** (text, tool calls, reasoning, usage).

- **PyPI:** https://pypi.org/project/vrraj-llm-adapter
- **GitHub:** https://github.com/vrraj/llm-adapter

## Install

```bash
pip install vrraj-llm-adapter
```


## Quickstart

```python
from llm_adapter import llm_adapter

resp = llm_adapter.create(
    model="openai:gpt-4o-mini",
    input="Write a one-sentence bedtime story about a unicorn.",
    max_output_tokens=100,
)

# Normalize to stable app-facing schema
result = llm_adapter.normalize_adapter_response(resp)

print(result["text"])
print(result["usage"])
```

## Recommended flow (create → normalize)

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
Use `normalized["text"]` for display-safe text; `resp.output_text` may include provider thought markup depending on model configuration.


## Public API (overview)

- `llm_adapter.create(...) -> AdapterResponse` — text generation (supports tools + optional streaming)
- `llm_adapter.normalize_adapter_response(...) -> LLMResult` — normalize `AdapterResponse` into a consistent dict schema
- `llm_adapter.create_embedding(...) -> EmbeddingResponse` — embeddings across providers
- `llm_adapter.get_pricing_for_model(...) -> Pricing | None` — pricing metadata lookup from the registry

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

### LLMResult (from `normalize_adapter_response`)

Top-level fields:

```python
{
  "provider": str,
  "model": str,
  "text": str,
  "reasoning": str | None,
  "usage": dict,
  "tool_calls": list,
  "status": str,
  "finish_reason": str | None,
  "raw": Any,
}
```

📋 For **complete method signatures, parameter details, and full response structures**, see: https://github.com/vrraj/llm-adapter/blob/main/API_REFERENCE.md

## What you get

- **One interface** for generation + embeddings across providers
- **Extensible registry-driven routing** (provider, endpoint, capabilities, limits)
- **Parameter policies** (allowed/disabled filtering per model)
- **Normalized responses** (text, tool calls, reasoning, usage)
- **Access control** via allowlist env var (`LLM_ADAPTER_ALLOWED_MODELS`)
- **Pricing metadata** in registry for cost visibility

## Interactive Playground (GitHub)

The repo includes a small FastAPI demo + UI to try models, inspect registry metadata, and view normalized responses.

![LLM Adapter Interactive Playground](https://github.com/vrraj/llm-adapter/blob/main/images/llm_adapter_interactive_playground.png)

➡️ Run it from GitHub: see **Development & Demo UI** in the expanded section below.

## Quick links (recommended next)

- **📋 Complete API Reference:** https://github.com/vrraj/llm-adapter/blob/main/API_REFERENCE.md
- **Model Registry docs:** https://github.com/vrraj/llm-adapter/blob/main/MODEL_REGISTRY.md
- **Examples folder:** https://github.com/vrraj/llm-adapter/tree/main/examples
- **Dev notes:** https://github.com/vrraj/llm-adapter/blob/main/DEVELOPMENT.md

---

<details>
<summary><strong>Full GitHub README (expand)</strong></summary>

## LLM Adapter Interactive Playground

Try models, inspect registry metadata, adjust parameters, and view normalized responses using the built-in test UI.

> See section **[Development & Demo UI](#development-and-demo-ui)** for instructions on how to run the interactive playground.

![LLM Adapter Interactive Playground](https://github.com/vrraj/llm-adapter/blob/main/images/llm_adapter_interactive_playground.png)

## Examples & Getting Started

Install the adapter from PyPI, then download and run the standalone example scripts to explore common usage patterns such as chat, embeddings, streaming, and custom registry overrides.

### Application Wrapper Pattern

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

### 📥 Run Standalone Examples

Install the package in a virtual environment:

```bash
pip install vrraj-llm-adapter
```

### Download and execute an example script

```bash
curl -O https://raw.githubusercontent.com/vrraj/llm-adapter/main/examples/llm_adapter_basic_usage.py

# Set your API keys
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."

# Run the example
python llm_adapter_basic_usage.py
```

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

## API Reference

📋 **For complete method signatures, parameter details, and response structures, see the [API Reference](https://github.com/vrraj/llm-adapter/blob/main/API_REFERENCE.md).**

Below is a quick overview of the main methods.

#### Generate Response (`llm_adapter.create(...)`)
Standard contract listed below. Additional model specific parameters can be passed as kwargs (limited by the param_policy defined in the model registry).

```python
response: AdapterResponse = llm_adapter.create(
    model: str = None,
    input: str | list[dict] = None,
    spec: Optional[ModelSpec] = None,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
    include_thoughts: Optional[bool] = False,
    stream: bool = False,
    **kwargs
)
```

- **model**: registry model key (e.g., "openai:gpt-4o-mini")
- **input**: prompt text or structured chat messages
- **spec**: alternative structured configuration (ModelSpec)
- **reasoning_effort**: adapter-level reasoning hint
- **Common policy-controlled parameters**: max_output_tokens, temperature, top_p
- **tools, tool_choice**: for structured tool invocation
- **include_thoughts**: include reasoning traces when supported
- **stream**: stream tokens as events

#### Normalize Response (`llm_adapter.normalize_adapter_response(...)`)
Converts provider-specific AdapterResponse to standardized LLMResult format.

```python
normalized: LLMResult = llm_adapter.normalize_adapter_response(
    resp: AdapterResponse,
    provider: Optional[str] = None,
    model_key: Optional[str] = None
)
```

**LLMResult Structure:**
```python
class LLMResult(TypedDict, total=False):
    provider: str
    model: str
    id: Optional[str]
    created_at: Optional[float]
    text: str
    reasoning: Optional[str]              # Reasoning content (Gemini)
    role: str
    status: str
    finish_reason: Optional[str]
    usage: "LLMUsage"                    # Standardized usage metrics
    tool_calls: List["LLMToolCall"]
    metadata: Optional[Dict[str, Any]]
    raw: Any                              # Original provider response
```

#### Generate Embeddings (`llm_adapter.create_embedding(...)`)
Provider-agnostic embedding generation.

```python
response: EmbeddingResponse = llm_adapter.create_embedding(
    model: str = None,
    input: str | list[str] = None,
    spec: Optional[ModelSpec] = None,
    dimensions: Optional[int] = None,
    **kwargs
)
```

- **model**: registry model key (e.g., "openai:embed_small")
- **input**: text or list of texts to embed
- **spec**: alternative structured configuration (ModelSpec)
- **dimensions**: output embedding dimensions (when supported)

#### Model Pricing (`llm_adapter.get_pricing_for_model(...)`)
Access pricing metadata for any model in the registry.

```python
pricing: Optional[Pricing] = llm_adapter.get_pricing_for_model(model_key: str)
```

**Pricing Structure:**
```python
class Pricing:
    input_per_mm: Optional[float]         # Cost per 1M input tokens
    output_per_mm: Optional[float]        # Cost per 1M output tokens
    cached_input_per_mm: Optional[float]  # Cost per 1M cached input tokens
```

### Text Generation

Generate text using any supported LLM provider.

```python
from llm_adapter import llm_adapter, LLMError

try:
    response = llm_adapter.create(
        model="openai:gpt-4o-mini",
        input="Explain quantum entanglement in simple terms."
    )
    print(response.output_text)
except LLMError as e:
    print(f"Error: {e.code} - {e}")
```

#### Available Models

**Discover all available models:**
```python
from llm_adapter import LLMAdapter

adapter = LLMAdapter()
print("Available models:")
for key in adapter.model_registry.keys():
    print(f"  - {key}")
```

**Key model categories:**
- **OpenAI**: `openai:gpt-4o-mini`, `openai:reasoning_o3-mini`, `openai:embed_small`
- **Gemini**: `gemini:openai-3-flash-preview`, `gemini:native-embed`, `gemini:openai-reasoning-2.5-flash`

📖 **For complete model registry documentation**, see https://github.com/vrraj/llm-adapter/blob/main/MODEL_REGISTRY.md#discovering-available-models

### Response normalization

For consistent response format across providers:

```python
from llm_adapter import llm_adapter, LLMError

try:
    response = llm_adapter.create(
        model="openai:gpt-4o-mini",
        input="Write a one-sentence bedtime story about a unicorn."
    )

    # Normalize to standard format
    normalized_response = llm_adapter.normalize_adapter_response(response)
    print(normalized_response['text'])
    print(normalized_response['usage'])
    print(normalized_response['finish_reason'])

except LLMError as e:
    print(f"Error: {e.code} - {e}")
```

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

### Models Allowlist (Access control)

```bash
export LLM_ADAPTER_ALLOWED_MODELS="openai:gpt-4o-mini,openai:embed_small"
```

**For comprehensive registry documentation, see:**
- https://github.com/vrraj/llm-adapter/blob/main/MODEL_REGISTRY.md
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

Do this to run the **demo UI** (runs on port 8100) or **customize** the code.

1. Clone the repository and run the setup script.

```bash
git clone https://github.com/vrraj/llm-adapter.git
cd llm-adapter
bash scripts/llm_adapter_setup.sh
```

>This script (scripts/llm_adapter_setup.sh) checks prerequisites (`python3`, `make`), creates `.env` if missing, sets up a local `.venv`, installs the package (`pip install -e .`), and shows **next steps**. The demo UI and FastAPI server run in this `.venv` virtual environment. Safe to run multiple times.

2. Set required API keys (see **Environment variables** section below).

3. Start the application.

```bash
make start
```

>**Note:** Run `make start` to run in foreground or `make start-bg` to run in background. Use `make stop` to stop the server.

4. Open the demo UI:

- http://localhost:8100/ui/

(See **Run the demo FastAPI server + UI** section below for full details.)

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

For internal design and architecture notes, see https://github.com/vrraj/llm-adapter/blob/main/DEVELOPMENT.md.

📖 **For parameter validation and filtering details**, see https://github.com/vrraj/llm-adapter/blob/main/MODEL_REGISTRY.md#parameter-validation-system.

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

## Response Structure

For complete response contracts (AdapterResponse, LLMResult, EmbeddingResponse), see https://github.com/vrraj/llm-adapter/blob/main/API_REFERENCE.md.

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
- `LLM_ADAPTER_ALLOWED_MODELS` (comma-separated list) - Restrict which models can be used.

## Model Allowlist (Access Control)

The `LLM_ADAPTER_ALLOWED_MODELS` environment variable allows you to restrict which models can be used. *By default, all models are allowed*.

```bash
export LLM_ADAPTER_ALLOWED_MODELS="openai:gpt-4o-mini,gemini:native-sdk-reasoning-2.5-flash"
```

## Run the demo FastAPI server + UI

### Option A: direct uvicorn

```bash
uvicorn llm_adapter_demo.api:app --reload --port 8100
```

* API root: http://127.0.0.1:8100/
* **LLM Adapter Interactive Playground**: http://127.0.0.1:8100/ui/

### Option B: Makefile helpers

```bash
make install
make start
make start-bg
make stop
make kill
make logs
```

For streaming examples, see `examples/streaming_call_example.py`.

## Examples

The `examples/` folder contains practical scripts demonstrating llm-adapter usage.

### Quick Start Examples

```bash
python examples/openai_adapter_example.py "Say hello from llm-adapter"
python examples/openai_embedding_example.py "Embed this text via llm-adapter"

export OPENAI_API_KEY="..."
python examples/streaming_call_example.py --model-key openai:gpt-4o-mini --prompt "seattle attractions"
```

## Supported Providers

Supports:
- **OpenAI** (Responses API, Chat Completions API, Embeddings API)
- **Gemini** (native `google-genai` SDK and OpenAI-compatible endpoint)

Models and capabilities are defined in `src/llm_adapter/model_registry.py`.

## Adding New Models

To add support for a new model:

1. Open `src/llm_adapter/model_registry.py`
2. Add a new entry to the `MODEL_INFO` dictionary
3. Define its endpoint (e.g., `chat_completions` or `gemini_sdk`) and its capabilities
4. Test it via the Demo UI

## Development

This is a standalone package. Development happens directly in this repo.

```bash
pip install -e .
make start
```

## License

This project is licensed under the MIT License.

</details>
