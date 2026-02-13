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
- **Interactive Demo**: Web UI for testing and comparison (available when running from source)

### Supported Providers
- **OpenAI**: GPT models and text embeddings
- **Gemini**: Native SDK and OpenAI-compatible endpoints

### Installation
```bash
pip install llm-adapter
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
- Demo UI at http://localhost:8100/ui/
- Test scripts in examples/

### Technical Details
- **Python 3.10+** required for union type syntax
- **FastAPI** demo server included
- **Model Registry** for provider configuration
- **Pricing metadata** helpers available
- **Environment variables** for API keys

### Known Limitations
- Currently supports OpenAI and Gemini providers
- Streaming examples require separate test scripts
- Some advanced provider features may need direct SDK usage

### Future Roadmap
- Additional provider support (Anthropic, Cohere, etc.)
- Enhanced streaming UI demo
- Advanced tool calling examples
- Performance optimization and caching
