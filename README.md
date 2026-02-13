
# vrraj-llm-adapter

![CI Status](https://github.com/vrraj/llm-adapter/actions/workflows/ci.yml/badge.svg)


Registry-driven LLM routing and response normalization for generation and embeddings (explicit endpoint semantics, model capability filtering, and parameter mapping). 
Install from **PyPI** for the core library, or clone from **GitHub** to run the demo UI and test scripts.

GitHub: https://github.com/vrraj/llm-adapter • PyPI: https://pypi.org/project/vrraj-llm-adapter/

This package provides:

- LLM **provider-agnostic** entrypoints for generation and embeddings
- **Standardized response helper** with normalized access to response text, tool calls, and usage
- **Registry-based pricing** metadata helpers
- Explicit **endpoint routing** (Responses api, chat completions, embeddings, Gemini SDK, Genini OpenAI API)
- `ModelRegistry` for registry-driven model resolution, capability filtering, and parameter mapping
- `ModelSpec` for reusable, typed configuration (structured alternative to passing kwargs)
- **Streaming** supported at library level

>**Note:** Demo UI and helper scripts are available when running from source.

## Prerequisites

- **Python 3.10+** - Required for union type syntax (`|`) used in the code
  - Tested primarily on Python 3.10–3.12 (3.13 may work but depends on upstream SDK compatibility).
- **pip** - Package installer (use `python3 -m pip` if `pip` not found)
- **LLM API Keys** Currently Supported: OpenAI and Gemini models



## Getting Started


### Option 1: Install from PyPI

1. **Setup virtual environment:**

```bash
# Option A: create a new environment for testing
python3 -m venv .venv
source .venv/bin/activate

# Option B: use your existing application environment
# source your-app-env/bin/activate
```

2. **Install vrraj-llm-adapter:**

```bash
pip install vrraj-llm-adapter
```

The PyPI package includes the core library only. The demo UI and helper scripts are available when running from source (see Option 2 below).

3. **Test it out:**

Set required API keys (see **Environment variables** section below).

Save the following as `test_llm_adapter.py`, then run:

```bash
python test_llm_adapter.py
```

Notes:
- The script below uses **OpenAI registry model keys** (`openai:...`).
- To test Gemini, swap the model keys (from model_registry - `src/llm_adapter/model_registry.py`) to `gemini:native-sdk-3-flash-preview` (generation) and `gemini:native-embed` (embeddings).

```python
from llm_adapter import llm_adapter

# Chat
resp = llm_adapter.create(
    model="openai:gpt-4o-mini",  # provider derived from model registry
    input=[{"role": "user", "content": "Hello"}],
    max_output_tokens=200,
)
print("Chat Response:")
print(resp.output_text)
print(f"Usage: {getattr(resp, 'usage', 'Usage info not available')}")

# Embeddings
emb_resp = llm_adapter.create_embedding(
    model="openai:embed_small",  # provider derived from model registry
    input="Hello world"
)
print("Embedding Response:")
print(f"Embedding (Truncated): {str(emb_resp.data)[:100]}...")
print(f"Usage: {getattr(emb_resp, 'usage', 'Usage info not available')}")
```

### Option 2: Run from source (demo UI + editable install)

Do this if you want to run the **demo UI** (runs on port 8100) or make **changes to the code**.

1. Clone the repository and run the setup script.

```bash
git clone https://github.com/vrraj/llm-adapter.git
cd llm-adapter
bash scripts/llm_adapter_setup.sh
```
>This script (scripts/llm_adapter_setup.sh) checks prerequisites (`python3`, `make`), creates `.env` if missing (from `.env.example` when available), sets up a local `.venv`, installs the package in editable mode (`pip install -e .`), and prints next steps. Safe to run multiple times.

2. Set required API keys (see **Environment variables** section below).

3. Start the application. 

```bash
make start
```
>**Note:** Run `make start` to run in foreground or `make start-bg` to run in background. Use `make stop` to stop the server.

4. Open the demo UI:

- http://localhost:8100/ui/

(See **Run the demo FastAPI server + UI** section below for full details.)


## Project structure

### Core components

- `src/llm_adapter/`
  - `llm_adapter.py` — standalone adapter implementation (adapted from `chat-with-rag/llm/llm_handler.py`)
  - `ModelSpec.py` — standalone version of `ModelSpec`
  - `model_registry.py` — registry of supported model keys/capabilities/endpoints
  - `__init__.py` — exports `LLMAdapter`, `llm_adapter`, `LLMError`, `ModelSpec`, `model_registry`
- `src/llm_adapter_demo/`
  - `api.py` — FastAPI app exposing `/api/models` and `/api/chat`, plus mounting the UI under `/ui`
  - `config.py` — environment checks + model options (derived from `llm_adapter.model_registry`)
- `ui/`
  - `index.html` — minimal test UI for trying registry model keys
  - `app.js` — frontend wiring to `/api/models` and `/api/chat`
  - `styles.css` — simple styling
- `examples/`
  - `test_openai_chat.py` — CLI example calling `llm_adapter.create` for OpenAI chat
  - `test_openai_embeddings.py` — CLI example calling `llm_adapter.create_embedding` for OpenAI embeddings
  - `test_streaming.py` — CLI example calling `llm_adapter.create(stream=True)` and printing deltas as they arrive
  - `test_model_spec.py` — Comprehensive test script demonstrating ModelSpec usage with different providers and parameter configurations


## Architecture and design notes

### Core components

1. **Model Registry** (`src/llm_adapter/model_registry.py`)
   - Central database of model metadata (`ModelInfo`)
   - Endpoint routing hint (e.g. `responses`, `chat_completions`, `embeddings`, `gemini_sdk`, `embed_content`)
   - Capability flags (e.g. `temperature`, `reasoning_effort`, `tools`, `stream`)
   - Parameter mappings (e.g. `max_output_tokens` vs `max_completion_tokens`)

2. **LLM Adapter** (`src/llm_adapter/llm_adapter.py`)
   - Routes generation calls based on registry metadata
   - Routes embedding calls based on registry metadata
   - Applies parameter mapping and capability-based filtering
   - Handles provider routing (OpenAI adapter vs Gemini native SDK)

### Endpoint semantics (important)

This standalone package uses these routing semantics:

- `endpoint="responses"`
  - Uses OpenAI **Responses API** (`client.responses.create(...)`).
- `endpoint="chat_completions"`
  - Uses OpenAI **Chat Completions API** (`client.chat.completions.create(...)`).
- `endpoint="embeddings"`
  - Uses OpenAI **embeddings create** (`client.embeddings.create(...)`).
- `endpoint="gemini_sdk"`
  - Uses Gemini native **SDK** (`google-genai` `models.generate_content(...)` / `generate_content_stream(...)`).
- `endpoint="embed_content"`
  - Uses Gemini **embed_content** (`google-genai` `models.embed_content(...)`).


## Call Signature

`LLMAdapter.create(...)` accepts a small set of explicit parameters (`input`, `provider`, `model`, `spec`, `stream`) plus **arbitrary keyword arguments** (`**kwargs`). The adapter then uses the model registry to map/filter some of those kwargs before calling the underlying provider SDK.

### How kwargs are processed (high level)

1. **Spec merge (optional)**
   - If `spec=ModelSpec(...)` is provided, merge `spec.to_kwargs()` with explicit `kwargs` (explicit wins).

2. **Drop `None` values**
   - Remove `kwargs` entries where the value is `None`.

3. **Token limit mapping (registry-driven)**
   - Remap token-limit params to the model’s configured name via `ModelInfo.max_tokens_parameter`
     (e.g. `max_output_tokens` → `max_completion_tokens`).

4. **Capability filtering (registry-driven)**
   - Drop known parameters that are explicitly unsupported for that model (e.g. `temperature`, `top_p`, `reasoning_effort`, `stream`, `tools`).
   - Unknown kwargs are generally passed through and may be accepted/rejected by the downstream SDK.

5. **Reasoning effort mapping/defaults (registry-driven)**
   - Map `reasoning_effort` into provider-specific parameters using `ModelInfo.reasoning_parameter` (Gemini examples: `thinking_level`, `thinking_budget`).
   - Apply defaults when the model is reasoning-capable and no effort is provided.

6. **Gemini thinking tax / config injection (Gemini-only)**
   - For Gemini models with `thinking_tax`, adjust token budgets and/or inject thinking config via `extra_body`.

### Common parameters you can pass today

These are the parameters the demo UI and handler paths are designed around:

- `temperature`: `float`
- `top_p`: `float`
- `max_output_tokens`: `int`
- `reasoning_effort`: `str` (e.g. `none`, `minimal`, `low`, `medium`, `high`)
- `tools`: list of OpenAI-style tool/function declarations
- `stream`: `bool`

In addition, the handler passes through many provider-specific kwargs. Two common escape hatches are:

- `extra_body`: `dict`
  - For OpenAI-compatible calls this can be used to pass provider-specific JSON fields.
- `timeout`: number/seconds (only effective on some SDK paths).

### Passing additional parameters safely

- Prefer **registry-known parameters** (those already used by existing call sites) for portability.
- If you pass a provider SDK parameter that the downstream method does not accept, you may get a `TypeError` (unexpected keyword argument) from the SDK.
- If you want the handler to reliably drop/accept a parameter for a given model key, add it to that model’s `capabilities` in `src/llm_adapter/model_registry.py`.

## Embeddings API (registry-routed)

`llm_adapter.create_embedding(...)` is provider-agnostic:

```python
resp = llm_adapter.create_embedding(
    model="...",  # registry key (provider auto-detected)
    input="...",  # str or list[str]
    **kwargs,
)
```

## ModelSpec: Structured Configuration

`ModelSpec` provides a type-safe, reusable way to configure model parameters as an alternative to passing individual parameters.

>**Note**: See `examples/test_model_spec.py` for a test script demonstrating ModelSpec usage with different providers and parameter configurations.

### Using ModelSpec

```python
from llm_adapter import llm_adapter
from llm_adapter import ModelSpec

# Create a reusable configuration
# Note: ModelSpec requires explicit provider and uses provider-native model names
chat_spec = ModelSpec(
    provider="openai",                    # Required: explicit provider
    model="gpt-4o-mini",                # Provider-native model name
    temperature=0.7,
    max_output_tokens=1000,
    extra={"custom_param": "value"}       # General provider-specific parameters
)

# Alternative with extra_body for OpenAI-compatible providers
chat_spec_with_body = ModelSpec(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,
    extra={"extra_body": {"custom_field": "value"}}  # For provider-specific JSON fields
)

# Use the spec in multiple calls
resp1 = llm_adapter.create(spec=chat_spec, input=[{"role": "user", "content": "Hello"}])
resp2 = llm_adapter.create(spec=chat_spec, input=[{"role": "user", "content": "How are you?"}])

# Works with embeddings too
embed_spec = ModelSpec(
    provider="openai",                    # Required: explicit provider
    model="text-embedding-3-small"       # Provider-native model name
)
resp = llm_adapter.create_embedding(spec=embed_spec, input="Text to embed")
```

### ModelSpec vs Individual Parameters

| Approach | Provider | Model Name | Auto-detection | Type Safety |
|----------|----------|------------|----------------|-------------|
| **Individual params** | Optional (auto-detected from registry) | Registry key (`openai:gpt-4o-mini`) | ✅ Yes | ❌ Runtime |
| **ModelSpec** | Required (explicit) | Provider-native (`gpt-4o-mini`) | ❌ No | ✅ Static type-checkers |

### Benefits

- **Type Safety**: Provider type hints support static checking (e.g. mypy/pyright)
- **Configuration Reuse**: Define once, use multiple times
- **Stable call sites**: Keep call sites consistent even when provider parameters differ (registry maps where needed)
- **Clean API**: Organized parameter grouping
- **Flexibility**: Mix spec with additional kwargs

## API usage and normalization

### Non-streaming usage (common case)

```python
from llm_adapter import llm_adapter

resp = llm_adapter.create(
    provider="openai",
    model="openai:gpt-4o-mini",
    input=[{"role": "user", "content": "Hello"}],
    stream=False,
    max_output_tokens=200,
)

print(f"Response: {resp}")
print(f"Output text: {resp.output_text}")
print(f"Usage: {resp.usage}")
```

>`llm_adapter.create(...)` returns the provider-native response object. Use `llm_adapter.build_llm_result_from_response(...)` for a provider-agnostic normalized view (see API Reference below).


### Streaming

When `stream=True`, `llm_adapter.create(...)` returns an **iterator**, not a normal JSON-serializable response.

- If you call `LLMAdapter.create(stream=True)` in Python code, you must iterate the returned events.
- The included demo FastAPI `/api/chat` endpoint is designed for **non-streaming JSON** responses.


>Use the CLI script `examples/test_streaming.py` to test streaming at the library level.

**Usage Example:** `python examples/test_streaming.py --provider openai --model openai:gpt-4o-mini --prompt "explain quantum physics in less than 50 words"`

## API Reference

Below are the primary public APIs exposed by `llm_adapter`.

### `llm_adapter.create(...)`

Unified generation entrypoint across providers.

```python
resp = llm_adapter.create(
    input: str | list[dict],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    spec: Optional[ModelSpec] = None,
    stream: bool = False,
    **kwargs
)
```

- Resolves provider automatically from registry key if not passed.
- Routes to the appropriate endpoint (`responses`, `chat_completions`, `gemini_sdk`, etc.).
- Applies registry-driven parameter mapping and capability filtering.
- Returns a provider-native response (or an iterator when `stream=True`).

---

### `llm_adapter.create_embedding(...)`

Unified embeddings entrypoint across providers.

```python
emb = llm_adapter.create_embedding(
    provider: Optional[str] = None,
    model: str,
    input: str | list[str],
    **kwargs
)
```

- Infers provider from registry key when possible.
- Routes to OpenAI embeddings or Gemini `embed_content`.
- Returns provider-native embedding response.

---

### `llm_adapter.build_llm_result_from_response(...)`

Provider-agnostic normalization helper.

```python
result = llm_adapter.build_llm_result_from_response(
    resp,
    provider: Optional[str] = None,
    model_key: Optional[str] = None
)
```

- Normalizes text, usage, reasoning tokens, and tool calls.
- If `provider` is not provided, attempts inference from:
  1. `model_key`
  2. `resp.model`
  3. defaults to `"openai"`
- Returns a consistent dictionary structure across providers.

#### Example 1: Let the adapter infer provider from registry key

```python
resp = llm_adapter.create(
    model="gemini:openai-reasoning-2.5-flash",
    input=[{"role": "user", "content": "Explain gravity briefly"}],
)

result = llm_adapter.build_llm_result_from_response(
    resp,
    model_key="gemini:openai-reasoning-2.5-flash"
)

print(result["text"])
print(result["usage"])
```

---

#### Example 2: Explicit provider override

```python
resp = llm_adapter.create(
    provider="openai",
    model="openai:gpt-4o-mini",
    input="Summarize AI in one sentence",
)

result = llm_adapter.build_llm_result_from_response(
    resp,
    provider="openai"
)

print(result["text"])
print(result["tool_calls"])
```

---

#### Example 3: Fully automatic inference (from `resp.model` when available)

```python
resp = llm_adapter.create(
    model="openai:gpt-4o-mini",
    input="Hello",
)

result = llm_adapter.build_llm_result_from_response(resp)

print(result["provider"])
print(result["text"])
```

---

### `llm_adapter.get_pricing_for_model_key(...)`

Registry-based pricing metadata lookup.

```python
pricing = llm_adapter.get_pricing_for_model_key("openai:gpt-4o-mini")
```

- Returns pricing metadata stored in the model registry (if defined).
- Does not compute costs — exposes registry metadata only.

---

### `llm_adapter.get_pricing_for_model(...)`

Lookup pricing metadata using provider-native model name.

```python
pricing = llm_adapter.get_pricing_for_model("gpt-4o-mini-2024-07-18")
```

- Resolves registry entry from provider model name.
- Returns associated pricing metadata if present.

## Environment variables

Copy `.env.example` to `.env` and to set up your API keys (or use your existing environment variables):

```bash
cp .env.example .env
```

Supported env vars:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `GEMINI_OPENAI_BASE_URL`


## Run the demo FastAPI server + UI

You can start the demo directly with `uvicorn` or via the `Makefile`.

### Option A: direct uvicorn

```bash
uvicorn llm_adapter_demo.api:app --reload --port 8100
```

* API root: http://127.0.0.1:8100/
* Demo UI: http://127.0.0.1:8100/ui/

### Option B: Makefile helpers

From the repo root:

```bash
# Create venv and install package (if not already done)
make install

# Run in foreground (logs to console)
make start

# Run in background (logs to logs/llm_adapter_demo.log)
make start-bg

# Stop / kill background server and free the port
make stop   # SIGTERM
make kill   # SIGKILL

# Tail background logs
make logs
```

The UI will:

- Call `/api/models` to list available registry model keys (and whether each provider is enabled based on env vars).
- Call `/api/chat` to send prompts to the selected model key via `llm_adapter.create(...)`.

The UI/API supports additional inference parameters:

- `temperature`
- `top_p`
- `reasoning_effort`
- `max_output_tokens`

When `reasoning_effort` is set for a reasoning-capable Gemini model, the handler requests thoughts via `include_thoughts` and the UI displays both **Reasoning** and **Answer** separately.


## CLI examples

The `examples/` folder contains simple scripts that exercise the handler directly.

**Prerequisite**: Install the package (via PyPI or `pip install -e .` if running from source).

### OpenAI calls via llm_adapter
From the repository root directory:

```bash
python3 examples/test_openai_chat.py "Say hello from standalone vrraj-llm-adapter"
```

The script will:

- Import `vrraj-llm-adapter` from the installed package.
- Call `llm_adapter.create(provider="openai", model="openai:fast", input=..., stream=False)`.
  - Print the response text and best-effort token usage if available.

### OpenAI embeddings via llm_adapter

From the repository root directory:
```bash
python3 examples/test_openai_embeddings.py "Embed this text via llm_adapter"

```

You can override the embedding model via `TEST_EMBEDDING_MODEL` (defaults to `text-embedding-3-small` - direct model name).
The script will:

- Call `llm_adapter.create_embedding(provider="openai", model=..., input=...)`.
- Print the embedding vector and usage information.

### Streaming via llm_adapter (OpenAI or Gemini)

```bash
export OPENAI_API_KEY="..."
python3 examples/test_streaming.py --model-key openai:fast --prompt "seattle attractions" --max-output-tokens 200
```

```bash
export GEMINI_API_KEY="..."
export GEMINI_OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
python3 examples/test_streaming.py --model-key gemini:native-sdk-3-flash-preview --prompt "seattle attractions" --max-output-tokens 200
```

### ModelSpec configuration testing

From the repository root directory:
```bash
python3 examples/test_model_spec.py
```

The script will:
- Test ModelSpec usage with different providers (OpenAI, Gemini)
- Demonstrate parameter configurations (basic, extra, extra_body)
- Show ModelSpec reusability and validation
- Test both chat completions and embeddings with ModelSpec

Note: API key errors are expected without valid credentials, but the script structure validates ModelSpec functionality.

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

This is a standalone package. Development happens directly in this repo. To install and test changes locally:

```bash
pip install -e .
make start
```


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
