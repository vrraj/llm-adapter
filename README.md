# llm-adapter

Standalone, pip-installable Python module for LLM generation and embeddings, usable either via PyPI or directly from source.

This package provides:

- Unified API for LLM generation and embeddings
- Standardized LLM response object with normalized access to text, tool calls, and usage
- Explicit endpoint routing (responses, chat_completions, embeddings, Gemini SDK)
- Model Registry-driven capability filtering and parameter mapping
- Streaming supported at library level

>**Note:** Includes a tiny FastAPI + Standalone HTML Demo + streaming scripts to sanity-check connectivity and validate responses from supported LLM providers (currently OpenAI and Gemini).

## Prerequisites

- **Python 3.10+** - Required for modern union type syntax (`|`) used in the code
- **pip** - Package installer (use `python3 -m pip` if `pip` not found)
- **OpenAI API Key** - For OpenAI provider testing
- **Gemini API Key** - For Gemini provider testing (optional)




## Getting Started


### Option 1: Install from PyPI (recommended)

1. **Setup virtual environment:**

```bash
# Option A: Create new environment for testing
python3 -m venv .venv
source .venv/bin/activate

# Option B: Use your existing application environment
source your-app-env/bin/activate  # Activate your existing venv first

```

2. **Install llm-adapter:**

```bash
pip install llm-adapter

```

3. **Test it out:**

  Set **API keys** via environment variables or a `.env` file.

  Run the following. 

>>**Note:** The script below uses OpenAI as the models. To test with Gemini, use gemini:native-sdk-3-flash-preview and gemini:native-embed.
> The model registry hosts these model specs. It is defined in src/llm_adapter/model_registry.py

```python
from llm_adapter import llm_adapter

# Chat
resp = llm_adapter.create(
    model="openai:gpt-4o-mini",  # provider derived from model registry
    input=[{"role": "user", "content": "Hello"}],
    max_output_tokens=200,
)
print(resp.output_text)
print(resp.usage)

# Embeddings
emb_resp = llm_adapter.create_embedding(
    model="openai:embed_small",  # provider derived from model registry
    input="Hello world"
)
print(emb_resp.data)
print(emb_resp.usage)

```

### Option 2: Run from source (demo UI + editable install)

Do this if you want to run the **demo UI** or make **changes to the code**.

1. Clone the repository and run the setup script.

```bash
git clone https://github.com/vrraj/llm-adapter.git
cd llm-adapter
bash scripts/llm_adapter_setup.sh

```

2. Set **API keys** via environment variables or a `.env` file.

3. Start the application. 

```bash
make start

```
**Note:** Run `make start` to run in foreground or `make start-bg` to run in background. Use `make stop` to stop the server.

4. Open the demo UI:

- Demo UI: http://localhost:8100/ui/


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
  - `test_openai_embeddings.py` — CLI example calling `llm_adapter.embeddings.create` for OpenAI embeddings
  - `test_streaming.py` — CLI example calling `llm_adapter.create(stream=True)` and printing deltas as they arrive


## Architecture and design notes

### Core components

1. **Model Registry** (`src/llm_adapter/model_registry.py`)
   - Central database of model metadata (`ModelInfo`)
   - Endpoint routing hint (e.g. `responses`, `chat_completions`, `embeddings`, `gemini_sdk`, `embed_content`)
   - Capability flags (e.g. `temperature`, `reasoning_effort`, `tools`, `stream`)
   - Parameter mappings (e.g. `max_output_tokens` vs `max_completion_tokens`)

2. **LLM Adapter** (`src/llm_adapter/llm_adapter.py`)
   - Unified `create(...)` entry point for chat/generation across providers
   - Unified `create_embedding(...)` entry point for embeddings across providers
   - Automatic parameter name conversion and capability-based filtering
   - Registry-driven provider routing for Gemini adapter vs native SDK

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


## Parameter passing and consumption

`LLMAdapter.create(...)` accepts a small set of explicit parameters (`input`, `provider`, `model`, `spec`, `stream`) plus **arbitrary keyword arguments** (`**kwargs`). The adapter then uses the model registry to map/filter some of those kwargs before calling the underlying provider SDK.

### How kwargs are processed (high level)

1. **Spec merge (optional)**
   - If you pass `spec=ModelSpec(...)`, the handler merges `spec.to_kwargs()` with your explicit kwargs.
   - Explicit kwargs win over spec values.

2. **Drop `None` values**
   - The handler removes any `kwargs` entries where the value is `None`.
   - This avoids provider SDK errors from receiving `max_output_tokens=None`, etc.

3. **Token limit parameter mapping (registry-driven)**
   - The registry controls which token-limit parameter name is used for a model via `ModelInfo.max_tokens_parameter`.
   - If you pass `max_output_tokens=...` (or `max_tokens=...`), the handler may remap it to the model’s configured token param.
     - Example: chat-completions models may map to `max_completion_tokens`.

4. **Capability-based filtering (registry-driven)**
   - The registry provides a best-effort `capabilities` dict for each model key.
   - For known capability-gated parameters (e.g. `temperature`, `top_p`, `reasoning_effort`, `stream`, `tools`), the handler will drop parameters that are explicitly marked unsupported.
   >[!TIP]
   >This repo’s filtering is intentionally permissive; unknown kwargs are generally passed through and may be accepted or rejected by the downstream SDK.

5. **Reasoning effort mapping/defaults (registry-driven)**
   - Some models map `reasoning_effort` into a provider-specific parameter using `ModelInfo.reasoning_parameter`.
   >[!TIP]
   >Example: Gemini adapter models may map `reasoning_effort` to `thinking_level`.
   - Some models also apply a default reasoning value if the model is reasoning-capable and no effort is provided.

6. **Gemini thinking tax / config injection (Gemini-only, registry-driven)**
   - For Gemini models with `thinking_tax` configured, the handler may adjust token budgets and/or inject provider-specific thinking config into `extra_body`.

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

## Embeddings API: adapter vs native SDK

`LLMAdapter.create_embedding(...)` is provider-agnostic:

```python
resp = llm_adapter.create_embedding(
    provider="openai" | "gemini" | "gemini_native",
    model="...",  # registry key or provider-native id
    input="...",  # str or list[str]
    **kwargs,
)
```

## API usage and normalization

### Non-streaming usage (common case)

```python
from llm_adapter import llm_adapter

resp = llm_adapter.create(
    provider="openai",
    model="openai:fast",
    input=[{"role": "user", "content": "Hello"}],
    stream=False,
    max_output_tokens=200,
)

print(resp.output_text)
print(resp.usage)
```

`llm_adapter.create(...)` returns the provider-native response object. Normalization is opt-in via a helper that derives a consistent, provider-agnostic view of text, usage, and tool calls.

### Normalized LLMResult helper

When you need a provider-agnostic view of text/usage/tool calls:

```python
result = llm_adapter.build_llm_result_from_response(resp, provider="openai")
print(result["text"])
print(result["usage"])

```

### Streaming

When `stream=True`, `llm_adapter.create(...)` returns an **iterator**, not a normal JSON-serializable response.

- If you call `LLMAdapter.create(stream=True)` in Python code, you must iterate the returned events.
- The included demo FastAPI `/api/chat` endpoint is designed for **non-streaming JSON** responses.
- If you set stream=true through the demo UI/API, the server may error because it cannot JSON-encode a live stream.

Use the CLI script `examples/test_streaming.py` to test streaming at the library level.

## Environment variables

Copy `.env.example` to `.env` and fill in any keys you want to test:

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

You can use this to quickly sanity-check that your keys, base URLs, and models are working.

## CLI examples

The `examples/` folder contains simple scripts that exercise the handler directly.

**Prerequisite**: Install the package (via PyPI or `pip install -e .` if running from source).

### OpenAI calls via llm_adapter
From the repository root directory:

```bash
python3 examples/test_openai_chat.py "Say hello from standalone llm-adapter"
```

If you omit the argument, the script will prompt you for input. It will:

- Import `llm_adapter` from the installed package.
- Call `llm_adapter.create(provider="openai", model="openai:fast", input=..., stream=False)`.
  - Print the response text and best-effort token usage if available.

### OpenAI embeddings via llm_adapter

From the repository root directory:
```bash
python3 examples/test_openai_embeddings.py "Embed this text via llm_adapter"
```

You can override the embedding model via `TEST_EMBEDDING_MODEL` (defaults to `text-embedding-3-small`).
The script will:

- Call `llm_adapter.create(provider="openai", model=..., input=...)`.
- Print the embedding vector length and a small prefix of the values.
- Print best-effort token usage if present.

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

---
