# Examples
These scripts demonstrate real-world usage patterns of the public API documented in [docs/API_REFERENCE.md](https://github.com/vrraj/llm-adapter/blob/main/docs/API_REFERENCE.md).

This directory contains practical examples demonstrating how to use the LLM Adapter for various use cases. Each example is self-contained and can be run independently.

## Quick Start

### Prerequisites

All examples assume:
- `pip install vrraj-llm-adapter`
- At least one provider API key set (e.g. `OPENAI_API_KEY` or `GEMINI_API_KEY`)

### Running Examples

```bash
# Download and run an example
curl -O https://raw.githubusercontent.com/vrraj/llm-adapter/main/examples/llm_adapter_basic_usage.py
python llm_adapter_basic_usage.py

# Or run from the cloned repository
cd llm-adapter
python examples/llm_adapter_basic_usage.py
```

---

## Core Examples

### `llm_adapter_basic_usage.py`
Basic Usage and Response Normalization

**Purpose:** Demonstrates basic OpenAI text generation.

**Key Features:**
- Text Generation with Error Handling
- Embedding Generation
- Response Normalization (`normalize_adapter_response`)
- Structured Error Handling with `LLMError`

**Usage:**
```bash
python examples/llm_adapter_basic_usage.py
```

**Requirements:**
- Required: OPENAI_API_KEY and/or GEMINI_API_KEY

---

### `llm_adapter_model_spec_example.py`
ModelSpec for Structured Configuration

**Purpose:** Demonstrates `ModelSpec` for type-safe, reusable model configuration.

**Key Features:**
- Reusable ModelSpec Configurations
- ModelSpec vs Individual Parameter Comparison
- Multi-Provider Testing (OpenAI, Gemini)
- Type Safety and Validation

**Usage:**
```bash
python examples/llm_adapter_model_spec_example.py
```

**Requirements:**
- Required: OPENAI_API_KEY and/or GEMINI_API_KEY

---

## Provider-Specific Examples

### `openai_adapter_example.py`
OpenAI Text Generation

**Purpose:** Demonstrates basic OpenAI text generation.

**Key Features:**
- OpenAI Text Generation
- Usage Statistics Extraction
- Basic Error Handling
- CLI Prompt Support

**Usage:**
```bash
# Interactive
python examples/openai_adapter_example.py

# Or pass prompt directly
python examples/openai_adapter_example.py "Explain quantum computing"
```

**Requirements:**
- Required: OPENAI_API_KEY and/or GEMINI_API_KEY

---

### `openai_embedding_example.py`
OpenAI Embeddings

**Purpose:** Demonstrates OpenAI embedding generation.

**Key Features:**
- Single and Batch Embedding
- Dimension Inspection
- Usage Tracking
- CLI Text Input Support

**Usage:**
```bash
# Interactive
python examples/openai_embedding_example.py

# Or pass text directly
python examples/openai_embedding_example.py "Text to embed"
```

**Requirements:**
- Required: OPENAI_API_KEY and/or GEMINI_API_KEY

---

### `streaming_call_example.py`
Streaming Responses

**Purpose:** Demonstrates streaming text generation.

**Key Features:**
- Streaming Text Generation
- Event and Delta Handling
- CLI Parameter Support
- Multi-Provider Compatibility

**Usage:**
```bash
python examples/streaming_call_example.py \
  --model-key openai:gpt-4o-mini \
  --prompt "Tell me a story" \
  --max-output-tokens 500 \
  --temperature 0.7
```

**Requirements:**
- Required: OPENAI_API_KEY and/or GEMINI_API_KEY

**Parameters:**
- `--model-key`: Registry model key
- `--prompt`: Text prompt (or read from stdin)
- `--max-output-tokens`: Output limit
- `--temperature`: Sampling temperature
- `--reasoning-effort`: Reasoning level (none/minimal/low/medium/high)

---

## Advanced Examples

### `custom_registry.py`
Custom Model Registry

**Purpose:** Demonstrates custom model registries and parameter policies.

**Key Features:**
- Custom ModelInfo Definitions
- Parameter Validation and Filtering
- Reasoning Policy Configuration
- Registry Validation and Pricing Metadata

**Usage:**
```bash
python examples/custom_registry.py
```

**Requirements:**
- Required: OPENAI_API_KEY and/or GEMINI_API_KEY

**Demonstrates:**
- OpenAI reasoning models with specific parameter policies
- Gemini reasoning models with full parameter support
- Embedding models with provider-specific dimension handling
- Parameter allowlist/disabled lists for security

---

### `get_model_pricing_example.py`
Model Pricing Lookup

**Purpose:** Demonstrates model pricing metadata lookup.

**Key Features:**
- Pricing Metadata Lookup
- Model Discovery
- Multiple Pricing Structure Support
- Cost Transparency

**Usage:**
```bash
# List all available models
python examples/get_model_pricing_example.py

# Get pricing for specific model
python examples/get_model_pricing_example.py openai:gpt-4o-mini
```

**Requirements:**
- None (uses registry data only)

---

### `set_adapter_allowed_models.py`
Access Control with Allowlists

**Purpose:** Demonstrates model access control via allowlists.

**Key Features:**
- Model Allowlist Configuration
- Environment Variable Setup
- Access Control Enforcement
- Blocked Model Error Handling

**Usage:**
```bash
# Set allowlist via environment
export LLM_ADAPTER_ALLOWED_MODELS="openai:gpt-4o-mini,gemini:native-embed"

python examples/set_adapter_allowed_models.py
```

**Requirements:**
- Required: OPENAI_API_KEY and/or GEMINI_API_KEY

**Environment Variables:**
- `LLM_ADAPTER_ALLOWED_MODELS`: Comma-separated list of allowed model keys

---

### `embeddings_magnitude_metadata.py`
Embedding Magnitude and Metadata

**Purpose:** Demonstrates embedding magnitude calculation and metadata extraction.

**Key Features:**
- Embedding Magnitude Calculation
- Normalization Options
- Metadata Extraction
- Provider Comparison

**Usage:**
```bash
python examples/embeddings_magnitude_metadata.py
```

**Requirements:**
- Required: OPENAI_API_KEY and/or GEMINI_API_KEY

---

### `provider_agnostic_embeddings.py`
Provider-Agnostic Embedding Usage

**Purpose:** Demonstrates provider-agnostic embedding usage.

**Key Features:**
- Registry Key Auto-Detection
- Provider-Agnostic Code Patterns
- Registry-Driven Routing
- Cross-Provider Compatibility

**Usage:**
```bash
python examples/provider_agnostic_embeddings.py
```

**Requirements:**
- Required: OPENAI_API_KEY and/or GEMINI_API_KEY

---

### `import_custom_registry.py`
Custom Registry Import

**Purpose:** Demonstrates importing and using external custom registries.

**Key Features:**
- External Registry Loading
- Registry Validation
- Custom Model Definitions
- Integration Patterns

**Usage:**
```bash
python examples/import_custom_registry.py
```

**Requirements:**
- Required: OPENAI_API_KEY and/or GEMINI_API_KEY

---

### `setting_openai_base_url.py`
Custom Base URL Configuration

**Purpose:** Demonstrates custom OpenAI base URL configuration.

**Key Features:**
- Custom Base URL Configuration
- Environment Variable Setup
- Endpoint Testing
- Connection Validation

**Usage:**
```bash
# Set custom base URL
export OPENAI_BASE_URL="https://your-proxy.example.com/v1"

python examples/setting_openai_base_url.py
```

**Requirements:**
- Required: OPENAI_API_KEY
- Optional: OPENAI_BASE_URL

---

## How to Navigate These Examples

### Learning Path

1. **Start with:** `llm_adapter_basic_usage.py` - Basic usage
2. **Then try:** `openai_adapter_example.py` - Provider-specific
3. **Explore:** `llm_adapter_model_spec_example.py` - Structured config
4. **Advanced:** `custom_registry.py` - Custom registry

### By Use Case

**Text Generation:**
- `llm_adapter_basic_usage.py` - Basic generation
- `openai_adapter_example.py` - OpenAI-specific
- `streaming_call_example.py` - Real-time streaming

**Embeddings:**
- `openai_embedding_example.py` - OpenAI embeddings
- `embeddings_magnitude_metadata.py` - Advanced metadata
- `provider_agnostic_embeddings.py` - Cross-provider

**Configuration & Control:**
- `llm_adapter_model_spec_example.py` - Structured config
- `custom_registry.py` - Custom registry
- `set_adapter_allowed_models.py` - Access control
- `setting_openai_base_url.py` - Custom endpoints

**Information & Debugging:**
- `get_model_pricing_example.py` - Pricing lookup
- `import_custom_registry.py` - Registry management

### By Complexity

**Beginner:**
- `llm_adapter_basic_usage.py`
- `openai_adapter_example.py`
- `openai_embedding_example.py`

**Intermediate:**
- `streaming_call_example.py`
- `llm_adapter_model_spec_example.py`
- `get_model_pricing_example.py`
- `set_adapter_allowed_models.py`

**Advanced:**
- `custom_registry.py`
- `embeddings_magnitude_metadata.py`
- `provider_agnostic_embeddings.py`
- `import_custom_registry.py`
- `setting_openai_base_url.py`

---

## Common Patterns

### Error Handling
Most examples demonstrate proper error handling with `LLMError`:

```python
from llm_adapter import llm_adapter, LLMError

try:
    response = llm_adapter.create(model="openai:gpt-4o-mini", input="Hello")
except LLMError as e:
    print(f"Error: {e.code} - {e}")
```

### Environment Setup
Examples use environment variables for API keys:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export LLM_ADAPTER_ALLOWED_MODELS="openai:gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### Response Normalization
Convert responses to standardized format:

```python
response = llm_adapter.create(model="openai:gpt-4o-mini", input="Hello")
normalized = llm_adapter.normalize_adapter_response(response)
print(f"Text: {normalized['text']}")
print(f"Usage: {normalized['usage']}")
```

---

## Development Tips

### Running Examples Locally

```bash
# Clone repository
git clone https://github.com/vrraj/llm-adapter.git
cd llm-adapter

# Install in development mode
pip install -e .

# Set API keys
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."

# Run any example
python examples/llm_adapter_basic_usage.py
```

### Testing Multiple Providers

Many examples work with both OpenAI and Gemini. Set both API keys to test cross-provider functionality:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
python examples/streaming_call_example.py --model-key openai:gpt-4o-mini --prompt "Test"
python examples/streaming_call_example.py --model-key gemini:native-sdk-3-flash-preview --prompt "Test"
```

### Customization

Examples are intentionally:
- Self-Contained
- Easy to Adapt
- Focused on Public API Usage
- Suitable for Extension into Production Code

---

## Troubleshooting

### Common Issues

- Import Errors: Ensure the package is installed (`pip install vrraj-llm-adapter`)
- Missing API Keys: Verify required environment variables are set
- Model Not Found: Confirm the registry key exists in [docs/MODEL_REGISTRY.md](https://github.com/vrraj/llm-adapter/blob/main/docs/MODEL_REGISTRY.md)
- Access Denied: Check `LLM_ADAPTER_ALLOWED_MODELS` configuration

### Debug Mode

Enable debug output via environment variable:

```bash
export DEBUG=1
python examples/llm_adapter_basic_usage.py
```

### Getting Help

- Check the [API Reference](https://github.com/vrraj/llm-adapter/blob/main/docs/API_REFERENCE.md)
- Review the [Model Registry](https://github.com/vrraj/llm-adapter/blob/main/docs/MODEL_REGISTRY.md)
- Open an issue on GitHub for specific problems
