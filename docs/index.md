# vrraj-llm-adapter

Provider-agnostic Python adapter for LLM text generation and embeddings, supporting **OpenAI** and **Google Gemini** with a unified interface and normalized response schema.
 
Includes a FastAPI-powered **interactive playground** for testing models, custom configurations, and adapter behavior.

## Install

```bash
pip install vrraj-llm-adapter
```
## Quick Example

```python
from llm_adapter import llm_adapter

resp = llm_adapter.create(
    model="openai:gpt-4o-mini", # for gemini, use "gemini:openai-3-flash-preview"
    input="Explain quantum computing in simple terms.",
    max_output_tokens=300,
)

# Normalize to stable app-facing schema
result = llm_adapter.normalize_adapter_response(resp)

print(result["text"])
print(result["usage"])
```

## Links

- [Github Repository](https://github.com/vrraj/llm-adapter)

- [PyPI Package](https://pypi.org/project/vrraj-llm-adapter/)


## Detailed Documentation


- [Full Documentation (README)](https://github.com/vrraj/llm-adapter#readme)

- [API Reference](API_REFERENCE.md) - Complete API documentation and usage examples

- [Model Registry Guide](MODEL_REGISTRY.md) - Model configuration, reasoning policies, and extensible custom registry

- [Development Guide](DEVELOPMENT.md) - Contributing, development setup, and demo UI

## Interactive Demo UI

The repository includes a FastAPI-powered **interactive playground** for testing.

This allows developers to experiment with models, registry configuration, and adapter behavior without writing code.

→ See setup instructions in the README: [Development and Demo UI](https://github.com/vrraj/llm-adapter#development-and-demo-ui)