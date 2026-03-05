---
layout: default
title: "vrraj-llm-adapter: Provider-agnostic LLM Abstraction Layer"
description: "A standardized Python wrapper for OpenAI and Gemini that normalizes responses into a single predictable API."
---

# vrraj-llm-adapter


<p align="left">
  <a href="https://pypi.org/project/vrraj-llm-adapter/">
    <img src="https://img.shields.io/pypi/v/vrraj-llm-adapter?color=blue&logo=pypi&logoColor=white" alt="PyPI - Version">
  </a>
  <a href="https://github.com/vrraj/llm-adapter/releases">
    <img src="https://img.shields.io/github/v/release/vrraj/llm-adapter?label=github%20release&color=orange&logo=github" alt="GitHub Release">
  </a>
  <a href="https://github.com/vrraj/llm-adapter/actions">
    <img src="https://github.com/vrraj/llm-adapter/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
</p>

Provider-agnostic Python adapter for LLM text **generation** and **embeddings**. Seamlessly support **OpenAI** and **Google Gemini** with a unified interface and normalized response schema.

## Key Features

- **Unified API**: Switch between openai and gemini by changing a single string - the model identifier.

- **Stable Schemas**: Stop parsing different JSON structures; get consistent LLMResult objects every time.

- **Interactive Playground**: Includes a built-in FastAPI dashboard to test model configurations, custom registry testing and compare responses in real-time.

- **Registry-Driven**: Manage model metadata, pricing, and routing through a centralized registry.

## Install

```bash
pip install vrraj-llm-adapter
```
## Quick Example

Requires **LLM provider API keys**. See README for setup.

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

- [Story on Medium](https://medium.com/@vr.rajkumar99/beyond-the-api-a-practical-registry-driven-adapter-for-openai-and-gemini-1298b437f41a) - Beyond the API: A Practical Registry-Driven Adapter for OpenAI and Gemini

## Interactive Demo UI

The repository includes a FastAPI-powered **interactive playground** for testing.

This allows developers to experiment with models, registry configuration, and adapter behavior without writing code.

→ See setup instructions in the README: [Development and Demo UI](https://github.com/vrraj/llm-adapter#development-and-demo-ui)