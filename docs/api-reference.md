# API Reference

This document provides the complete API reference for the LLM Adapter, including method signatures, parameter details, response structures, and common usage patterns.

> **New here?** Start with the project overview on the home page: **[vrraj-llm-adapter docs home](https://vrraj.github.io/llm-adapter/)**.
>
> **Source + releases:** GitHub repo and PyPI package are linked from the home page.

> **Note:** The package exposes a convenience singleton `llm_adapter` which is an instance of `LLMAdapter`. You can either use this pre-configured instance or create your own `LLMAdapter` instance for custom configuration.

## Table of Contents

- [Core Classes](#core-classes)
- [Main API Methods](#main-api-methods)
- [Response Structures](#response-structures)
- [Error Handling](#error-handling)
- [Common Usage Patterns](#common-usage-patterns)

---

## Core Classes

### `LLMAdapter`

The main adapter class that provides unified access to multiple LLM providers.

```python
class LLMAdapter:
    def __init__(
        *,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        gemini_base_url: Optional[str] = None,
        model_registry: Optional[Dict[str, Any]] = None,
        openai_client: Any = None,
        gemini_client: Any = None,
    )
```

**Parameters:**
- `openai_api_key`: OpenAI API key (defaults to `OPENAI_API_KEY` env var)
- `gemini_api_key`: Gemini API key (defaults to `GEMINI_API_KEY` env var)
- `openai_base_url`: OpenAI base URL (defaults to `OPENAI_BASE_URL` env var)
- `gemini_base_url`: Gemini base URL (defaults to `GEMINI_OPENAI_BASE_URL` env var)
- `model_registry`: Custom model registry to override/extend defaults
- `openai_client`: Pre-configured OpenAI client (for dependency injection)
- `gemini_client`: Pre-configured Gemini client (for dependency injection)

### `ModelSpec`

Structured configuration for model parameters.

```python
@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
```

---

## Main API Methods

### `create()`

Generate text completions using any supported LLM provider.

```python
def create(
    self,
    *,
    input: Any,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    spec: Optional[ModelSpec] = None,
    stream: bool = False,
    **kwargs: Any,
) -> Union[AdapterResponse, Iterator[AdapterEvent]]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `str | list[dict]` | ✅ | Prompt text or structured chat messages |
| `provider` | `str` | ❌ | Provider override (`openai`, `gemini`). Inferred from model if not specified |
| `model` | `str` | ❌* | Registry model key (e.g., `"openai:gpt-4o-mini"`). Required if `spec` not provided |
| `spec` | `ModelSpec` | ❌* | Alternative structured configuration. Required if `model` not provided |
| `stream` | `bool` | ❌ | Enable streaming responses (default: `False`) |
| `**kwargs` | `Any` | ❌ | Provider-specific parameters (filtered by model's param_policy) |

**Common `**kwargs` (model-dependent):**
- `reasoning_effort`: `"none" | "minimal" | "low" | "medium" | "high"` - Adapter-level reasoning hint
- `max_output_tokens`: `int` - Maximum output tokens
- `temperature`: `float` - Sampling temperature (0.0-2.0)
- `top_p`: `float` - Nucleus sampling (0.0-1.0)
- `tools`: `list[dict]` - Tool/function definitions
- `tool_choice`: `str | dict` - Tool choice strategy
- `include_thoughts`: `bool` - Include reasoning traces when supported (legacy, use `reasoning_effort` instead)

**Response Fields:**
- `text`: `str` - Main response text
- `reasoning`: `str` - Reasoning/thinking content from `generate.reasoning` field
- `usage`: `dict` - Token usage information

**Returns:**
- Non-streaming: `AdapterResponse`
- Streaming: `Iterator[AdapterEvent]`

**Example:**
```python
# Basic usage
response = llm_adapter.create(
    model="openai:gpt-4o-mini",
    input="Explain quantum entanglement in simple terms."
)

# With structured messages
response = llm_adapter.create(
    model="gemini:native-sdk-3-flash-preview",
    input=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_output_tokens=500
)

# Streaming
for event in llm_adapter.create(
    model="openai:gpt-4o-mini",
    input="Tell me a story",
    stream=True
):
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

### `create_embedding()`

Generate embeddings using any supported provider.

```python
def create_embedding(
    self,
    *,
    input: Any,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    spec: Optional[ModelSpec] = None,
    **kwargs: Any,
) -> EmbeddingResponse
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `str | list[str]` | ✅ | Text or list of texts to embed |
| `provider` | `str` | ❌ | Provider override. Inferred from model if not specified |
| `model` | `str` | ❌* | Registry model key. Required if `spec` not provided |
| `spec` | `ModelSpec` | ❌* | Alternative structured configuration. Required if `model` not provided |
| `**kwargs` | `Any` | ❌ | Provider-specific parameters |

**Common `**kwargs` (model-dependent):**
- `dimensions`: `int` - Output embedding dimensions (when supported)
- `normalize_embedding`: `bool` - Whether to normalize vectors (Gemini only)
- `task_type`: `str` - Task type for Gemini native embeddings
- `output_dimensionality`: `int` - Output dimensions for Gemini native embeddings

**Returns:** `EmbeddingResponse`

**Example:**
```python
# Basic embedding
response = llm_adapter.create_embedding(
    model="openai:embed_small",
    input="The quick brown fox jumps over the lazy dog"
)

# Batch embeddings
response = llm_adapter.create_embedding(
    model="gemini:native-embed",
    input=["Text 1", "Text 2", "Text 3"],
    dimensions=768
)

# Normalized embeddings (Gemini)
response = llm_adapter.create_embedding(
    model="gemini:openai-embed",
    input="Text to normalize",
    normalize_embedding=True
)
```

### `normalize_adapter_response()`

Convert an `AdapterResponse` to a standardized `LLMResult` format.

```python
def normalize_adapter_response(
    self,
    resp: AdapterResponse,
    *,
    provider: Optional[str] = None,
    model_key: Optional[str] = None,
) -> LLMResult
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resp` | `AdapterResponse` | ✅ | Response from `llm_adapter.create()` |
| `provider` | `str` | ❌ | Provider override (inferred from response if not specified) |
| `model_key` | `str` | ❌ | Model key override (inferred from response if not specified) |

**Returns:** `LLMResult`

**Example:**
```python
response = llm_adapter.create(model="openai:gpt-4o-mini", input="Hello")
normalized = llm_adapter.normalize_adapter_response(response)

print(f"Text: {normalized['text']}")
print(f"Reasoning: {normalized.get('reasoning')}")
print(f"Usage: {normalized['usage']}")
```

### `get_pricing_for_model()`

Access pricing metadata for any model in the registry.

> **Note:** This method is also available as `get_model_pricing()` for backward compatibility. `get_pricing_for_model()` is the canonical name.

```python
def get_pricing_for_model(self, model: str) -> Optional[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | ✅ | Registry model key or provider-native model name |

**Returns:** `Optional[Dict[str, Any]]` - Pricing metadata or `None` if not found

**Example:**
```python
pricing = llm_adapter.get_pricing_for_model("openai:gpt-4o-mini")
if pricing:
    print(f"Input cost: ${pricing['input_per_mm']}/1M tokens")
    print(f"Output cost: ${pricing['output_per_mm']}/1M tokens")
```

---

## Response Structures

### `AdapterResponse`

Primary response from `create()` method for non-streaming calls.

```python
class AdapterResponse:
    def __init__(
        self,
        *,
        output_text: str,
        model: str,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        adapter_response: Any | None = None,
        model_response: Any | None = None,
        status: Optional[str] = None,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    )
```

**Fields:**

| Field | Type | Stability | Description |
|-------|------|------------|-------------|
| `output_text` | `str` | ✅ Guaranteed | Generated text content |
| `model` | `str` | ✅ Guaranteed | Model identifier used |
| `usage` | `Dict[str, int]` | ✅ Guaranteed | Token usage information |
| `status` | `str` | ✅ Guaranteed | Completion status (`"completed"`, `"incomplete"`) |
| `finish_reason` | `str` | ✅ Guaranteed | Why generation stopped |
| `tool_calls` | `List[Dict]` | ✅ Guaranteed | Tool/function calls if any |
| `metadata` | `Dict[str, Any]` | ✅ Guaranteed | Provider and routing metadata |
| `adapter_response` | `Any` | 🔧 Debug/Opaque | Adapter-processed response (may vary) |
| `model_response` | `Any` | 🔧 Debug/Opaque | Original provider response (may vary) |

**Legend:**
- ✅ **Guaranteed**: Stable interface across providers and versions
- 🔧 **Debug/Opaque**: For debugging only, may change between providers/versions

### `EmbeddingResponse`

Response from `create_embedding()` method.

```python
class EmbeddingResponse:
    def __init__(
        self,
        data: List[List[float]],
        usage: Any,
        normalized: Optional[bool] = None,
        vector_dim: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw: Optional[Any] = None,
    )
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `data` | `List[List[float]]` | Direct list of embedding vectors |
| `usage` | `EmbeddingUsage` | Token usage information |
| `normalized` | `bool` | Whether vectors were normalized |
| `vector_dim` | `int` | Dimension of each vector |
| `metadata` | `Dict[str, Any]` | Additional metadata (includes provider, model, etc.) |
| `raw` | `Any` | Original response for debugging |

### `EmbeddingUsage`

Usage information for embedding responses.

```python
class EmbeddingUsage:
    def __init__(self, prompt_tokens: int = 0, total_tokens: int = 0)
```

**Fields:**
- `prompt_tokens`: `int` - Number of input tokens
- `total_tokens`: `int` - Total tokens processed

### `LLMResult`

Standardized response format from `normalize_adapter_response()`.

```python
class LLMResult(TypedDict, total=False):
    text: str
    reasoning: Optional[str]
    role: str
    status: str
    finish_reason: Optional[str]
    usage: "LLMUsage"
    tool_calls: List["LLMToolCall"]
    metadata: Optional[Dict[str, Any]]
    raw: Any
```

**Key Fields:**
- `text`: `str` - Main response text
- `reasoning`: `Optional[str]` - Separate reasoning content (Gemini)
- `usage`: `LLMUsage` - Standardized usage metrics
- `tool_calls`: `List[LLMToolCall]` - Normalized tool calls

### `LLMUsage`

Standardized usage metrics across all providers.

```python
class LLMUsage(TypedDict, total=False):
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    answer_tokens: int
    total_tokens: int
```

**Key Relationships:**
- `output_tokens = answer_tokens + reasoning_tokens`
- `total_tokens = prompt_tokens + cached_tokens + output_tokens`

### `AdapterEvent`

Streaming event from `create(stream=True)`.

```python
class AdapterEvent:
    def __init__(self, event_type: str, delta: Optional[str] = None)
```

**Fields:**
- `type`: `str` - Event type (`"response.output_text.delta"`, `"response.output_text.done"`)
- `delta`: `Optional[str]` - Text delta for delta events

### `LLMError`

Structured error for provider or configuration failures.

```python
class LLMError(Exception):
    def __init__(
        self,
        *,
        provider: str,
        model: Optional[str] = None,
        kind: str = "llm_error",
        code: Optional[Any] = None,
        message: str = "",
        retry_after: Optional[float] = None,
    )
```

**Common Error Kinds:**
- `"config"` - Configuration issues (missing API keys, invalid models)
- `"rate_limit"` - Rate limiting errors
- `"auth"` - Authentication failures
- `"model_not_found"` - Model not available
- `"request"` - Invalid request parameters
- `"provider_error"` - Provider-side errors

---

## Error Handling

All methods can raise `LLMError` for structured error handling.

```python
from llm_adapter import llm_adapter, LLMError

try:
    response = llm_adapter.create(
        model="openai:gpt-4o-mini",
        input="Hello world"
    )
except LLMError as e:
    print(f"Provider: {e.provider}")
    print(f"Model: {e.model}")
    print(f"Error kind: {e.kind}")
    print(f"Error code: {e.code}")
    print(f"Message: {e}")
    if e.retry_after:
        print(f"Retry after: {e.retry_after} seconds")
```

---

## Common Usage Patterns

### 1. Basic Text Generation

```python
from llm_adapter import llm_adapter

response = llm_adapter.create(
    model="openai:gpt-4o-mini",
    input="Write a haiku about programming"
)

print(response.output_text)
print(f"Usage: {response.usage}")
```

### 2. Streaming Responses

```python
def stream_response(model: str, prompt: str):
    collected_text = []
    
    for event in llm_adapter.create(
        model=model,
        input=prompt,
        stream=True
    ):
        if event.type == "response.output_text.delta":
            delta = event.delta or ""
            print(delta, end="", flush=True)
            collected_text.append(delta)
        elif event.type == "response.output_text.done":
            print("\n[Streaming complete]")
            break
    
    return "".join(collected_text)

full_text = stream_response("openai:gpt-4o-mini", "Tell me a story")
```

### 3. Tool/Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = llm_adapter.create(
    model="openai:gpt-4o-mini",
    input="What's the weather in Seattle?",
    tools=tools,
    tool_choice="auto"
)

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Function: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
```

### 4. Reasoning with Gemini

```python
response = llm_adapter.create(
    model="gemini:native-sdk-reasoning-2.5-flash",
    input="Solve this step by step: 15 * 23 - 7",
    reasoning_effort="high",
    max_output_tokens=1000
)

normalized = llm_adapter.normalize_adapter_response(response)

if normalized.get('reasoning'):
    print(f"Reasoning: {normalized['reasoning']}")
print(f"Answer: {normalized['text']}")
```

### 5. Batch Embeddings

```python
texts = [
    "The cat sat on the mat",
    "Artificial intelligence is transforming society",
    "Machine learning models require training data"
]

response = llm_adapter.create_embedding(
    model="openai:embed_small",
    input=texts
)

print(f"Generated {len(response.data)} embeddings")
print(f"Dimensions: {response.vector_dim}")
print(f"Usage: {response.usage}")

# Access individual embeddings
for i, embedding in enumerate(response.data):
    print(f"Text {i}: {len(embedding)} dimensions")
```

### 6. ModelSpec for Reusable Configuration

```python
from llm_adapter import ModelSpec

# Create reusable configuration
chat_spec = ModelSpec(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,
    max_output_tokens=1000
)

# Use with multiple requests
for prompt in ["Hello", "How are you?", "Goodbye"]:
    response = llm_adapter.create(spec=chat_spec, input=prompt)
    print(response.output_text)
```

### 7. Error Handling with Fallbacks

```python
def safe_generate(prompt: str, primary_model: str, fallback_model: str):
    try:
        return llm_adapter.create(model=primary_model, input=prompt)
    except LLMError as e:
        if e.kind in ["rate_limit", "model_not_found"]:
            print(f"Primary model failed: {e}. Trying fallback...")
            return llm_adapter.create(model=fallback_model, input=prompt)
        else:
            raise

response = safe_generate(
    "Explain photosynthesis",
    "openai:gpt-4o-mini",
    "gemini:native-sdk-3-flash-preview"
)
```

### 8. Access Control with Allowlist

```python
import os

# Set allowlist (or use LLM_ADAPTER_ALLOWED_MODELS env var)
os.environ["LLM_ADAPTER_ALLOWED_MODELS"] = "openai:gpt-4o-mini,gemini:native-sdk-3-flash-preview"

try:
    # This will work
    response = llm_adapter.create(model="openai:gpt-4o-mini", input="Hello")
    
    # This will raise LLMError with code="model_not_allowed"
    response = llm_adapter.create(model="openai:gpt-4o", input="Hello")
except LLMError as e:
    if e.code == "model_not_allowed":
        print(f"Model not in allowlist: {e.model}")
```

---

## Parameter Stability

### Stable Parameters (guaranteed across providers)

| Parameter | Stability | Notes |
|-----------|-----------|-------|
| `input` | ✅ Stable | Core parameter for all methods |
| `model` | ✅ Stable | Registry model key |
| `provider` | ✅ Stable | Provider override |
| `spec` | ✅ Stable | ModelSpec configuration |
| `stream` | ✅ Stable | Streaming flag |
| `max_output_tokens` | ✅ Stable | Canonical output limit |
| `temperature` | ✅ Stable | When supported by model |
| `top_p` | ✅ Stable | When supported by model |

### Provider-Specific Parameters (passed via `**kwargs`)

| Parameter | Provider | Stability | Notes |
|-----------|----------|-----------|-------|
| `reasoning_effort` | OpenAI, Gemini | 🔄 Adapter-level | Normalized by adapter |
| `tools`, `tool_choice` | OpenAI, Gemini | 🔄 Adapter-level | Normalized by adapter |
| `include_thoughts` | Gemini | ⚠️ Legacy | Use `reasoning_effort` instead; reasoning available via `generate.reasoning` field |
| `normalize_embedding` | Gemini | 🔄 Provider-level | Gemini embeddings only |
| `dimensions` | OpenAI, Gemini | 🔄 Provider-level | When supported |
| `task_type`, `output_dimensionality` | Gemini native | 🔄 Provider-level | Native SDK only |

**Legend:**
- ✅ **Stable**: Guaranteed interface, won't change
- 🔄 **Adapter-level**: Normalized by adapter for consistent behavior
- 🔄 **Provider-level**: Passed through to provider SDK

For complete parameter policies per model, see the model registry configuration.
