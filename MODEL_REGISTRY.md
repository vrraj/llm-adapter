# Model Registry Guide

The model registry maps stable **registry keys** (e.g. `openai:gpt-4o-mini`) to provider-native configurations, enabling provider-agnostic LLM calls.

## Discovering Available Models

### For Installed Packages (pip install users)

After installing via pip, you can discover all available registry keys programmatically:

```python
from llm_adapter import LLMAdapter
from llm_adapter.model_registry import REGISTRY

# Method 1: List all available models
adapter = LLMAdapter()
print("Available models:")
for key in adapter.model_registry.keys():
    print(f"  - {key}")

# Method 2: Get model details
for key, model_info in adapter.model_registry.items():
    print(f"{key}:")
    print(f"  Provider: {model_info.provider}")
    print(f"  Model: {model_info.model}")
    print(f"  Endpoint: {model_info.endpoint}")
    if model_info.pricing:
        print(f"  Pricing: ${model_info.pricing.input_per_mm}/1M input, ${model_info.pricing.output_per_mm}/1M output")
    print()

# Method 3: Filter by provider
openai_models = [k for k, v in adapter.model_registry.items() if v.provider == "openai"]
gemini_models = [k for k, v in adapter.model_registry.items() if v.provider == "gemini"]

print(f"OpenAI models: {openai_models}")
print(f"Gemini models: {gemini_models}")

# Method 4: Check if a specific model exists
if "openai:gpt-4o-mini" in adapter.model_registry:
    print("openai:gpt-4o-mini is available!")
```

### Quick Model Reference

Here are the default registry keys included with the package:

#### OpenAI Models
- `openai:gpt-4o-mini` - Text generation (Responses API)
- `openai:gpt-4o` - Text generation (Responses API)  
- `openai:chat_gpt-4o-mini` - Text generation (Chat Completions)
- `openai:chat_gpt-4o` - Text generation (Chat Completions)
- `openai:reasoning_o3-mini` - Reasoning model
- `openai:reasoning_gpt-5-mini` - Reasoning model
- `openai:embed_small` - Embeddings (1536 dimensions)
- `openai:embed_large` - Embeddings (3072 dimensions)

#### Gemini Models
- `gemini:openai-2.5-flash-lite` - Text generation (OpenAI-compatible)
- `gemini:openai-3-flash-preview` - Text generation with reasoning (OpenAI-compatible)
- `gemini:native-sdk-3-flash-preview` - Text generation (Native SDK)
- `gemini:openai-reasoning-2.5-flash` - Budget-based reasoning (OpenAI-compatible)
- `gemini:native-sdk-reasoning-2.5-flash` - Budget-based reasoning (Native SDK)
- `gemini:native-embed` - Embeddings (Native SDK)

## ModelInfo Structure

```python
ModelInfo(
    provider="openai",                    # LLM provider
    model="gpt-4o-mini",                  # Provider-native model name
    endpoint="responses",                 # API endpoint type
    pricing=Pricing(...),                  # Cost information
    param_policy={...},                    # Parameter restrictions
    limits={...},                         # Model limitations
    reasoning_policy={...},                # Reasoning configuration
    capabilities={...}                    # Model features
)
```

## Registry Examples

### OpenAI Models

#### Text Generation (Responses API)
```python
"openai:gpt-4o-mini": ModelInfo(
    provider="openai",
    model="gpt-4o-mini",
    endpoint="responses",
    pricing=Pricing(input_per_mm=0.15, output_per_mm=0.60, cached_input_per_mm=0.075),
    limits={"max_output_tokens": 2000},
    param_policy={
        "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
        "disabled": {"reasoning_effort", "stream", "include_thoughts"}
    },
    capabilities={"assistant_role": "assistant"}
)
```

#### Text Generation (Chat Completions API)
```python
"openai:chat_gpt-4o-mini": ModelInfo(
    provider="openai",
    model="gpt-4o-mini",
    endpoint="chat_completions",
    pricing=Pricing(input_per_mm=0.15, output_per_mm=0.60, cached_input_per_mm=0.075),
    limits={"max_output_tokens": 2000},
    param_policy={
        "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
        "disabled": {"reasoning_effort", "stream", "include_thoughts"}
    },
    capabilities={"assistant_role": "assistant"}
)
```

#### Reasoning Models
```python
"openai:reasoning_o3-mini": ModelInfo(
    provider="openai",
    model="o3-mini",
    endpoint="responses",
    pricing=Pricing(input_per_mm=1.10, output_per_mm=4.40),
    limits={"max_output_tokens": 2000},
    param_policy={
        "allowed": {"max_output_tokens", "reasoning_effort", "reasoning", "tools", "tool_choice"},
        "disabled": {"stream", "temperature", "top_p", "include_reasoning"}
    },
    reasoning_policy={
        "mode": "openai_effort",
        "default": "low"
    },
    reasoning_parameter=("reasoning_effort", "low"),
    capabilities={"assistant_role": "assistant"}
)
```

#### Embeddings
```python
"openai:embed_small": ModelInfo(
    provider="openai",
    model="text-embedding-3-small",
    endpoint="embeddings",
    pricing=Pricing(input_per_mm=0.02, output_per_mm=0.0),
    param_policy={
        "allowed": {"normalize_embedding", "dimensions"},
        "disabled": {"include_thoughts", "output_dimensionality"}
    },
    capabilities={"dimensions": 1536, "assistant_role": "assistant"}
)
```

### Gemini Models

#### Text Generation (OpenAI-Compatible)
```python
"gemini:openai-3-flash-preview": ModelInfo(
    provider="gemini",
    model="models/gemini-3-flash-preview",
    endpoint="chat_completions",
    pricing=Pricing(input_per_mm=0.50, output_per_mm=3.00),
    limits={"max_output_tokens": 2000},
    param_policy={
        "allowed": {
            "max_output_tokens", "reasoning_effort", "include_reasoning", 
            "thinking_level", "thinking_budget", "temperature", "top_p", 
            "tools", "tool_choice"
        },
        "disabled": set()
    },
    reasoning_policy={
        "mode": "gemini_level",
        "param": "thinking_level",
        "default": "minimal",
        "map": {
            "none": "minimal",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high"
        },
        "reserve_ratio": {
            "none": 0.0,
            "minimal": 0.25,
            "low": 0.30,
            "medium": 0.50,
            "high": 0.80
        },
        "counts_against_output": True
    },
    reasoning_parameter=("thinking_level", "minimal"),
    capabilities={"assistant_role": "model"}
)
```

#### Text Generation (Native SDK)
```python
"gemini:native-sdk-3-flash-preview": ModelInfo(
    provider="gemini",
    model="models/gemini-3-flash-preview",
    endpoint="gemini_sdk",
    pricing=Pricing(input_per_mm=0.50, output_per_mm=3.00),
    limits={"max_output_tokens": 2000},
    param_policy={
        "allowed": {
            "max_output_tokens", "reasoning_effort", "include_reasoning", 
            "thinking_level", "thinking_budget", "temperature", "top_p", 
            "tools", "tool_choice"
        },
        "disabled": set()
    },
    reasoning_policy={
        "mode": "gemini_level",
        "param": "thinking_level",
        "default": "low",
        "map": {
            "none": "minimal",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high"
        },
        "reserve_ratio": {
            "none": 0.0,
            "minimal": 0.25,
            "low": 0.30,
            "medium": 0.50,
            "high": 0.80
        },
        "counts_against_output": True
    },
    reasoning_parameter=("thinking_level", "low"),
    capabilities={"assistant_role": "model"}
)
```

#### Embeddings (OpenAI-Compatible)
```python
"gemini:embed_openai_compat": ModelInfo(
    provider="gemini",
    model="models/embedding-001",
    endpoint="embeddings",
    pricing=Pricing(input_per_mm=0.10, output_per_mm=0.0),
    param_policy={
        "allowed": {"normalize_embedding", "dimensions"},
        "disabled": {"include_thoughts", "output_dimensionality"}
    },
    capabilities={"dimensions": 768, "assistant_role": "model"}
)
```

#### Embeddings (Native SDK)
```python
"gemini:native-embed": ModelInfo(
    provider="gemini",
    model="gemini-embedding-001",
    endpoint="embed_content",
    pricing=Pricing(input_per_mm=0.10, output_per_mm=0.0),
    param_policy={
        "allowed": {"normalize_embedding", "dimensions", "task_type", "output_dimensionality"},
        "disabled": set()
    },
    capabilities={
        "dimensions": 1536,
        "assistant_role": "model",
        "task_type": "RETRIEVAL_DOCUMENT",
        "output_dimensionality": 1536
    }
)
```

## Field Explanations

### Core Fields

| Field | Purpose | Values |
|-------|---------|--------|
| `provider` | LLM provider | `"openai"`, `"gemini"` |
| `model` | Provider-native model name | `"gpt-4o-mini"`, `"models/gemini-3-flash-preview"` |
| `endpoint` | API endpoint type | `"responses"`, `"chat_completions"`, `"embeddings"`, `"gemini_sdk"`, `"embed_content"` |

### Endpoint Types

| Endpoint | OpenAI | Gemini | Description |
|----------|--------|--------|-------------|
| `responses` | ✅ OpenAI Responses API | ❌ | New OpenAI API with built-in reasoning |
| `chat_completions` | ✅ Chat Completions API | ✅ OpenAI-compatible endpoint | Standard chat API |
| `embeddings` | ✅ Embeddings API | ✅ OpenAI-compatible embeddings | Standard embeddings |
| `gemini_sdk` | ❌ | ✅ Native SDK | Gemini native generation |
| `embed_content` | ❌ | ✅ Native SDK | Gemini native embeddings |

### Configuration Fields

#### `pricing` - Cost Information
```python
Pricing(
    input_per_mm=0.15,           # $ per 1M input tokens
    output_per_mm=0.60,          # $ per 1M output tokens
    cached_input_per_mm=0.075    # $ per 1M cached input tokens
)
```

#### `param_policy` - Parameter Control
```python
param_policy={
    "allowed": {"temperature", "top_p", "max_output_tokens"},  # Parameters to allow
    "disabled": {"reasoning_effort", "stream"},               # Parameters to block
    "mapped": {"top_p": "top_p"}                              # Parameter mappings
}
```

#### `limits` - Model Limitations
```python
limits={
    "max_output_tokens": 2000,     # Maximum output tokens
    "max_input_length": 1000000    # Maximum input length
}
```

#### `reasoning_policy` - Reasoning Configuration
```python
# OpenAI-style reasoning
reasoning_policy={
    "mode": "openai_effort",
    "default": "low"
}

# Gemini level-based reasoning
reasoning_policy={
    "mode": "gemini_level",
    "param": "thinking_level",
    "default": "minimal",
    "map": {"low": "low", "medium": "medium"},
    "reserve_ratio": {"low": 0.30, "medium": 0.50},
    "counts_against_output": True
}

# Gemini budget-based reasoning
reasoning_policy={
    "mode": "gemini_budget",
    "param": "thinking_budget",
    "default": "low",
    "budget_map": {"low": 1000, "medium": 2000},
    "counts_against_output": True
}
```

#### `capabilities` - Model Features
```python
capabilities={
    "assistant_role": "assistant",    # Response role in conversations
    "dimensions": 1536,               # Embedding dimensions
    "reasoning_effort": True,         # Supports reasoning
    "task_type": "RETRIEVAL_DOCUMENT" # Gemini embedding task type
}
```

## Usage Examples

### Basic Usage
```python
from llm_adapter import LLMAdapter

adapter = LLMAdapter()
response = adapter.create(
    model="openai:gpt-4o-mini",
    input="Hello, world!",
    temperature=0.7
)
```

### Gemini Thinking Configuration

#### Using `include_thoughts` and Thinking Parameters
```python
from llm_adapter import LLMAdapter

adapter = LLMAdapter()

# Method 1: Automatic thinking inclusion (recommended)
response = adapter.create(
    model="gemini:openai-3-flash-preview",
    input="Solve this step by step: 2 + 2 * 3",
    reasoning_effort="medium"  # Automatically sets include_thoughts=True
)

# Method 2: Explicit thinking control
response = adapter.create(
    model="gemini:openai-3-flash-preview",
    input="Explain your reasoning process",
    include_reasoning=True,     # Forces include_thoughts=True
    thinking_level="high"       # Explicit thinking intensity
)

# Method 3: Budget-based thinking
response = adapter.create(
    model="gemini:openai-reasoning-2.5-flash",
    input="Analyze this problem deeply",
    reasoning_effort="medium",  # Maps to thinking_budget=1000
    thinking_budget=2000        # Override with specific budget
)
```

#### Processing Thinking Responses
```python
# Raw response with thinking blocks
print(response.output_text)
# Output: <thought>
# Let me think step by step...
# 2 + 2 * 3 = 2 + 6 = 8
# </thought>
# 
# The answer is 8.

# Normalized response separates reasoning from answer
normalized = adapter.normalize_adapter_response(response)
print(f"Reasoning: {normalized['reasoning']}")
print(f"Answer: {normalized['text']}")

# Output:
# Reasoning: Let me think step by step...
# 2 + 2 * 3 = 2 + 6 = 8
# Answer: The answer is 8.
```

#### Thinking Configuration Examples
```python
# Level-based thinking (gemini_level)
"gemini:thinking-model": ModelInfo(
    provider="gemini",
    model="models/gemini-3-flash-preview",
    endpoint="chat_completions",
    reasoning_policy={
        "mode": "gemini_level",
        "param": "thinking_level",
        "default": "minimal",
        "map": {
            "none": "minimal",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium", 
            "high": "high"
        }
    },
    param_policy={
        "allowed": {
            "thinking_level",      # Direct thinking control
            "reasoning_effort",    # Public interface (maps to thinking_level)
            "include_reasoning",   # Forces include_thoughts=True
            "include_thoughts"     # Direct control over thought blocks
        },
        "disabled": set()
    }
)

# Budget-based thinking (gemini_budget)
"gemini:budget-thinking-model": ModelInfo(
    provider="gemini",
    model="models/gemini-2.5-flash",
    endpoint="chat_completions",
    reasoning_policy={
        "mode": "gemini_budget",
        "param": "thinking_budget",
        "default": "low",
        "budget_map": {
            "none": 0,
            "minimal": 500,
            "low": 1000,
            "medium": 2000,
            "high": 5000
        }
    },
    param_policy={
        "allowed": {
            "thinking_budget",     # Exact token budget
            "reasoning_effort",    # Maps to budget values
            "include_reasoning",   # Forces include_thoughts=True
            "include_thoughts"     # Direct control
        },
        "disabled": set()
    }
)
```

### Custom Registry

#### Complete Custom Registry Example (`custom_registry.py`)
```python
# custom_registry.py
from llm_adapter.model_registry import ModelInfo, Pricing, validate_registry

# Custom registry with new models and overrides
REGISTRY = {
    # New model not in default registry
    "company:internal-gpt4": ModelInfo(
        provider="openai",
        model="gpt-4",  # Internal deployment
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.0, output_per_mm=0.0),  # Free internal
        param_policy={
            "allowed": {"temperature", "max_output_tokens", "tools"},
            "disabled": {"reasoning_effort", "stream"}
        },
        capabilities={"assistant_role": "assistant"}
    ),
    
    # Override default model configuration
    "openai:gpt-4o-mini": ModelInfo(
        provider="openai",
        model="gpt-4o-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=0.10, output_per_mm=0.40),  # Custom pricing
        param_policy={
            "allowed": {"temperature", "max_output_tokens", "tools"},
            "disabled": {"reasoning_effort"}  # Allow streaming
        },
        capabilities={"assistant_role": "assistant"}
    ),
    
    # Gemini reasoning model with custom configuration
    "gemini:custom-reasoning": ModelInfo(
        provider="gemini",
        model="models/gemini-2.0-flash-exp",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.20, output_per_mm=1.00),
        reasoning_policy={
            "mode": "gemini_level",
            "param": "thinking_level",
            "default": "medium",
            "map": {
                "none": "minimal",
                "low": "low", 
                "medium": "medium",
                "high": "high"
            },
            "reserve_ratio": {
                "none": 0.0,
                "low": 0.25,
                "medium": 0.50,
                "high": 0.80
            },
            "counts_against_output": True
        },
        reasoning_parameter=("thinking_level", "medium"),
        param_policy={
            "allowed": {
                "max_output_tokens", "reasoning_effort", "include_reasoning",
                "thinking_level", "temperature", "top_p", "tools", "tool_choice"
            },
            "disabled": set()
        },
        capabilities={"assistant_role": "model", "reasoning_effort": True}
    )
}

# Validate the custom registry
validate_registry(REGISTRY)
```

#### Using Custom Registry
```python
from llm_adapter.model_registry import REGISTRY as DEFAULT_REGISTRY
from custom_registry import REGISTRY as CUSTOM_REGISTRY

# Method 1: Use only custom registry
adapter = LLMAdapter(model_registry=custom_registry)
response = adapter.create(model="my-model", input="Hello!")

# Method 2: Merge with default registry (custom overrides defaults)
MERGED_REGISTRY = {**DEFAULT_REGISTRY, **CUSTOM_REGISTRY}
adapter = LLMAdapter(model_registry=MERGED_REGISTRY)

# Both default and custom models available
response1 = adapter.create(model="openai:gpt-4o-mini")  # Uses custom pricing
response2 = adapter.create(model="company:internal-gpt4")  # New custom model
response3 = adapter.create(model="gemini:custom-reasoning")  # Custom reasoning
```

**Registry Override Behavior:**
- **Whole registry keys are overridden** - If a key exists in both registries, the custom version completely replaces the default
- **Partial merging not supported** - You can't merge individual fields, only entire ModelInfo objects
- **New keys are added** - Custom registry keys not in defaults are added to the final registry
- **Example**: `"openai:gpt-4o-mini"` in custom registry completely replaces the default configuration with all custom settings
