# Development & Architecture Documentation

This document contains internal design notes, architecture details, and project structure information for developers working on the llm-adapter codebase.

## Project Structure

### Core Components

- `src/llm_adapter/`
  - `llm_adapter.py` — standalone adapter implementation (adapted from `chat-with-rag/llm/llm_handler.py`)
  - `ModelSpec.py` — standalone version of `ModelSpec`
  - `model_registry.py` — registry of supported model keys/capabilities/endpoints
  - `__init__.py` — exports `LLMAdapter`, `llm_adapter`, `LLMError`, `ModelSpec`, `model_registry`
- `src/llm_adapter_demo/`
  - `api.py` — FastAPI app exposing `/api/models` and `/api/chat`, plus mounting the UI under `/ui/`
  - `config.py` — environment checks + model options (derived from `llm_adapter.model_registry`)
- `ui/`
  - `index.html` — minimal test UI for trying registry model keys
  - `app.js` — frontend wiring to `/api/models` and `/api/chat`
  - `styles.css` — simple styling
- `examples/`
  - `openai_adapter_example.py` — CLI example calling `llm_adapter.create` for OpenAI chat
  - `openai_embedding_example.py` — CLI example calling `llm_adapter.create_embedding` for OpenAI embeddings
  - `streaming_call_example.py` — CLI example calling `llm_adapter.create(stream=True)` and printing deltas as they arrive
  - `setting_openai_base_url.py` — Example showing how to configure custom OpenAI base URL
  - `get_model_pricing_example.py` — Example for retrieving model pricing information
  - `set_adapter_allowed_models.py` — Example for configuring model allowlists
  - `custom_registry.py` — Template for creating custom model registries
  - `import_custom_registry.py` — Example demonstrating custom registry usage
  - `llm_adapter_model_spec_example.py` — Comprehensive example demonstrating ModelSpec usage with different providers and parameter configurations
  - `test_magnitude_metadata.py` — Example showing magnitude metadata for embeddings
  - `test_provider_agnostic_embeddings.py` — Example demonstrating provider auto-detection
- `tests/`
  - `unit/` — Unit tests that don't require API keys
    - `test_imports.py` — Basic package import validation
  - `integration/` — Integration tests that require API keys
    - `test_llm_adapter.py` — Basic chat and embeddings integration tests
    - `test_adapter_embedding_calls.py` — Advanced embedding functionality tests
    - `test_gemini_tool_calls_flow.py` — Gemini tool calling integration tests
    - `test_openai_tool_calls_flow.py` — OpenAI tool calling integration tests

## Architecture and Design Notes

### Core Components

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

## Custom Model Registry (Override / Extend Defaults)

`LLMAdapter` uses the default registry in `src/llm_adapter/model_registry.py`.

You can provide your own registry mapping to `LLMAdapter(model_registry=...)`.
The adapter will merge registries as:

- Default registry (package) + user registry overrides
- User entries replace default entries for the same key

### User-side Call Signature

```python
from llm_adapter import LLMAdapter
from my_app.my_registry import REGISTRY as USER_REGISTRY

llm = LLMAdapter(model_registry=USER_REGISTRY)
```

### Restricting Which Models Can Be Used (Allowlist)

You can restrict which registry keys may be used via an environment variable. When an allowlist is enabled, models must be referenced by **registry key** (for example: `openai:gpt-4o-mini`).

#### Environment Variable Configuration (`LLM_ADAPTER_ALLOWED_MODELS`)

```bash
export LLM_ADAPTER_ALLOWED_MODELS="openai:gpt-4o-mini,openai:embed_small"
```

### Validating Registries

If you provide your own registry, you can validate it before instantiating the adapter.

```python
from llm_adapter.model_registry import validate_registry
from my_app.my_registry import REGISTRY as USER_REGISTRY

validate_registry(USER_REGISTRY, strict=False)
```

You can also validate the merged registry (defaults + your overrides) after creating the adapter:

```python
from llm_adapter import LLMAdapter
from llm_adapter.model_registry import validate_registry
from my_app.my_registry import REGISTRY as USER_REGISTRY

llm = LLMAdapter(model_registry=USER_REGISTRY)
validate_registry(llm.model_registry, strict=False)
```

## Merging Custom Registries (Interactive Demo)

The LLM Adapter includes an **interactive demo UI** that allows you to test custom registries without writing any code. This is perfect for experimenting with new model configurations before integrating them into your application.

### Quick Start with Demo UI

1. **Start the demo server:**
   ```bash
   cd llm-adapter
   source .venv/bin/activate  # or activate your venv
   uvicorn llm_adapter_demo.api:app --reload --host 0.0.0.0 --port 8100
   ```

2. **Open the UI:** Navigate to `http://localhost:8100/ui`

3. **Enable custom registry:** Check the "Merge custom registry (examples/custom_registry.py)" checkbox

4. **Select your custom models:** The dropdown will now show both default and custom models

### Testing Your Custom Registry

**Script:** `examples/import_custom_registry.py`

Run this test script to verify your custom registry is set up correctly:

```bash
cd llm-adapter
source .venv/bin/activate
python examples/import_custom_registry.py
```

This script validates:
- Custom registry import and merging
- Model availability (custom + default)
- Pricing lookup for custom models
- Registry validation
- Model resolution functionality

### Live Updates

The custom registry is **dynamically imported** on each request, so you can iterate quickly on model configurations.

### Integration Examples

See these files for complete working examples:

- **`examples/custom_registry.py`** - Sample custom registry with reasoning models
- **`examples/import_custom_registry.py`** - Test script demonstrating programmatic usage

### Production Usage

For production use, you can:

1. **Create your registry** in your application package
2. **Use the same pattern** as the demo: `LLMAdapter(model_registry=YOUR_REGISTRY)`
3. **Validate before deployment** using `validate_registry()`

> The demo UI provides a sandbox for testing before integrating into your production code.

## Custom Registry Override

For quick registry overrides without the demo UI:

1. **Copy the template from examples/custom_registry.py**
   ```bash
   # Start with the provided template
   cp examples/custom_registry.py my_app/my_registry.py
   ```

2. **Define your complete model configurations**
   ```python
   # Option A: Override existing model (define complete configuration)
   "openai:custom_reasoning_o3-mini": ModelInfo(
       provider="openai",
       model="o3-mini", 
       endpoint="chat_completions",
       pricing=Pricing(input_per_mm=0.8, output_per_mm=3.2),
       limits={"max_output_tokens": 3000},
       # ... all required fields must be defined
   )
   
   # Option B: Add entirely new model (define complete configuration)
   "openai:custom-gpt4-turbo": ModelInfo(
       key="openai:custom-gpt4-turbo",
       provider="openai", 
       model="gpt-4-turbo",
       endpoint="chat_completions",
       pricing=Pricing(input_per_mm=0.3, output_per_mm=0.9),
       limits={"max_output_tokens": 4096},
       capabilities={"assistant_role": "assistant"},
   )
   ```

3. **Instantiate LLMAdapter with your registry**
   ```python
   from llm_adapter import LLMAdapter
   from my_app.my_registry import REGISTRY as CUSTOM_REGISTRY
   
   adapter = LLMAdapter(model_registry=CUSTOM_REGISTRY)
   ```

## Development Workflow

### Setting Up Development Environment

1. **Clone and setup:**
   ```bash
   git clone https://github.com/vrraj/llm-adapter.git
   cd llm-adapter
   bash scripts/llm_adapter_setup.sh
   ```

2. **Set API keys** in `.env`:
   ```bash
   OPENAI_API_KEY=sk-...
   GEMINI_API_KEY=AIza...
   ```

3. **Run tests:**
   ```bash
   # Unit tests only
   pytest
   
   # Integration tests (requires API keys)
   pytest -m integration
   
   # All tests
   pytest -m "integration or unit"
   ```

4. **Start demo UI:**
   ```bash
   make start
   ```

### Code Organization Principles

- **Registry-driven design** - All model behavior controlled through model_registry.py
- **Provider abstraction** - Single interface for multiple LLM providers
- **Parameter validation** - Automatic filtering and mapping of parameters
- **Extensibility** - Easy to add new providers and models
- **Testing** - Comprehensive unit and integration test coverage

### Adding New Providers

1. **Add provider constants** to `llm_adapter.py`
2. **Implement provider-specific methods** for generation and embeddings
3. **Add client initialization** and authentication
4. **Create model registry entries** with proper endpoints and capabilities
5. **Add integration tests** for the new provider
6. **Update documentation** with provider-specific examples

### Adding New Models

1. **Create ModelInfo entries** in `model_registry.py` or custom registry
2. **Define capabilities** and parameter policies
3. **Set pricing and limits** if applicable
4. **Test with demo UI** before production deployment
5. **Add examples** if model has special capabilities
