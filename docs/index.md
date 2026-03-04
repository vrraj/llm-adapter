# vrraj-llm-adapter

Provider-agnostic Python adapter for LLM text generation and embeddings.

Supports OpenAI and Google Gemini with a unified interface and normalized request/response schema.

## Installation

```bash
pip install vrraj-llm-adapter
```
### Quick Example

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

## Project Links

- [GitHub Repository](https://github.com/vrraj/llm-adapter)

- [PyPI Package](https://pypi.org/project/vrraj-llm-adapter/)


## Detailed Documentation


- [Full Documentation (README)](https://github.com/vrraj/llm-adapter#readme)

- [API Reference](API_REFERENCE.md) - Complete API documentation and usage examples

- [Model Registry Guide](MODEL_REGISTRY.md) - Model configuration, reasoning policies, and registry setup

- [Development Guide](DEVELOPMENT.md) - Contributing, development setup, and demo UI
