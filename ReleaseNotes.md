# Release Notes

## Version 0.1.0 - Initial Release

### Overview
The `llm-adapter` package provides a unified interface for LLM generation and embeddings across multiple providers, with automatic parameter mapping, capability filtering, and response normalization.

### Key Features
- **Unified API**: Single interface for OpenAI and Gemini providers
- **Model Registry**: Centralized model configuration and parameter mapping
- **ModelSpec**: Structured, reusable configuration alternative to passing individual kwargs
- **Automatic Provider Resolution**: Infer provider from model registry keys
- **Parameter Mapping**: Convert generic parameters to provider-specific formats
- **Response Normalization**: Consistent response format across providers
- **Streaming Support**: Built-in streaming capabilities
- **Embedding Metadata**: Track magnitude and normalization information
- **Interactive Demo**: **LLM Adapter Interactive Playground** for testing and comparison (available when running from source)

### Supported Providers
- **OpenAI**: GPT models and text embeddings
- **Gemini**: Native SDK and OpenAI-compatible endpoints

### Installation
```bash
pip install vrraj-llm-adapter
```

### Quick Start
```python
from llm_adapter import llm_adapter

# Chat completion
resp = llm_adapter.create(
    model="openai:gpt-4o-mini",
    input="Hello, world!"
)

# Embeddings
resp = llm_adapter.create_embedding(
    model="openai:embed_small",
    input="Text to embed"
)
```

### Documentation
- Full documentation in README.md
- API reference with examples
- **LLM Adapter Interactive Playground** at http://localhost:8100/ui/
- Test scripts in examples/ folder
- **Unit and integration tests** in tests/ folder

### Technical Details
- **Python 3.10+** required for union type syntax
- **FastAPI** demo server included
- **Model Registry** for provider configuration
- **Pricing metadata** helpers available
- **Environment variables** for API keys

### Examples and Testing
- **User Examples**: Practical scripts demonstrating real-world usage
  - `openai_adapter_example.py` - Chat completions
  - `openai_embedding_example.py` - Text embeddings
  - `streaming_call_example.py` - Real-time streaming
  - `setting_openai_base_url.py` - Custom endpoints
  - `get_model_pricing_example.py` - Pricing lookup
  - `set_adapter_allowed_models.py` - Model allowlists
  - `custom_registry.py` - Custom model definitions
- **Unit Tests**: Fast tests without API key requirements
- **Integration Tests**: Full API tests with real providers
- **CI/CD**: Automated testing across Python 3.10-3.13

### Known Limitations
- Currently supports OpenAI and Gemini providers
- Integration tests require API keys for full functionality
- Some advanced provider features may need direct SDK usage

### Future Roadmap
- Additional provider support (Anthropic, Cohere, etc.)
- Enhanced streaming UI demo
- Advanced tool calling examples
- Performance optimization and caching
