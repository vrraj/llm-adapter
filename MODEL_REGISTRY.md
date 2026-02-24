# Model Registry Guide

The model registry is the configuration backbone of **llm_adapter**. It maps stable **registry keys** (e.g. `openai:gpt-4o-mini`) to provider-native model names, endpoints, pricing, limits, and reasoning/parameter policies — so your application code stays provider-agnostic.

## Quick Reference: Custom Registry Setup

### 🚀 3-Step Custom Registry Setup

#### **Step 1: Create Your Custom Registry**
Create `my_custom_registry.py`:
```python
from llm_adapter.model_registry import ModelInfo, Pricing, validate_registry

REGISTRY = {
    # Add your custom models here
    "openai:my-custom-model": ModelInfo(
        provider="openai",
        model="gpt-4-turbo",  # Provider-native model name
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.30, output_per_mm=1.20),
        capabilities={"assistant_role": "assistant"},
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": {"reasoning_effort", "stream", "include_thoughts"}
        },
    ),
    
    "gemini:my-reasoning-model": ModelInfo(
        provider="gemini",
        model="models/gemini-2.0-flash-exp",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.25, output_per_mm=1.50),
        capabilities={"reasoning_effort": True},
        param_policy={
            "allowed": {"max_output_tokens", "reasoning_effort", "include_thoughts", "thinking_level", "thinking_budget", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": set()
        },
        reasoning_policy={
            "mode": "gemini_level",
            "param": "thinking_level",
            "default": "medium",
        },
    ),
}

# Validate your registry
validate_registry(REGISTRY)
```

#### **Step 2: Use Custom Registry in Code**
```python
from llm_adapter import LLMAdapter
from llm_adapter.model_registry import REGISTRY as DEFAULT_REGISTRY
from my_custom_registry import REGISTRY as CUSTOM_REGISTRY

# Merge registries (custom registry keys override defaults)
MERGED_REGISTRY = {**DEFAULT_REGISTRY, **CUSTOM_REGISTRY}

# Create adapter with merged registry
adapter = LLMAdapter(model_registry=MERGED_REGISTRY)

# Now you can use both default and custom models
response = adapter.create(
    model="openai:my-custom-model",  # Your custom model
    input="Hello, world!"
)
```

#### **Step 3: Restrict Access with Environment Variable**
```bash
# Set allowed registry keys (comma-separated)
export LLM_ADAPTER_ALLOWED_MODELS="openai:my-custom-model,gemini:my-reasoning-model,openai:embed_small"
```

```python
# Adapter will only allow registry keys in the allowlist
adapter = LLMAdapter(
    model_registry=MERGED_REGISTRY,
    allowed_model_keys=["openai:my-custom-model", "gemini:my-reasoning-model"]
)
```

### 📋 Common Custom Registry Patterns

#### **Add Private/Internal Models**
```python
"company:internal-gpt4": ModelInfo(
    provider="openai",
    model="gpt-4",  # Internal deployment
    endpoint="chat_completions",
    pricing=Pricing(input_per_mm=0.0, output_per_mm=0.0),  # Free internal
    capabilities={"assistant_role": "assistant"},
),
```

#### **Override Default Model Configuration**
```python
# Override default pricing for openai:gpt-4o-mini
"openai:gpt-4o-mini": ModelInfo(
    provider="openai",
    model="gpt-4o-mini",
    endpoint="chat_completions",
    pricing=Pricing(input_per_mm=0.10, output_per_mm=0.40),  # Custom pricing
    param_policy={"disabled": {"reasoning_effort"}},  # Allow streaming
    capabilities={"assistant_role": "assistant"},
),
```

#### **Add Experimental Models**
```python
"openai:experimental-o1": ModelInfo(
    provider="openai",
    model="o1-preview",
    endpoint="responses",  # New endpoint type
    pricing=Pricing(input_per_mm=2.50, output_per_mm=10.00),
    param_policy={"disabled": {"stream", "temperature"}},
    capabilities={"reasoning_effort": True},
),
```

### 🎯 Registry Key Examples

| Registry Key | Provider | Model Name | Use Case |
|--------------|-----------|------------|----------|
| `openai:gpt-4o-mini` | OpenAI | `gpt-4o-mini` | Default chat model |
| `openai:embed_small` | OpenAI | `text-embedding-3-small` | Embeddings |
| `gemini:native-embed` | Gemini | `gemini-embedding-001` | Gemini embeddings |
| `company:custom-model` | OpenAI | `internal-gpt-4` | Private model |
| `research:experimental` | Gemini | `models/gemini-exp` | Experimental model |

### ⚙️ Configuration Field Reference

| Field | Purpose | Example |
|-------|---------|---------|
| `provider` | LLM provider | `"openai"`, `"gemini"` |
| `model` | Provider-native name | `"gpt-4o-mini"`, `"models/gemini-3-flash"` |
| `endpoint` | API endpoint | `"responses"`, `"chat_completions"`, `"gemini_sdk"`, `"embeddings"` |
| | | `chat_completions` = OpenAI Chat Completions **OR** Gemini OpenAI-compatible endpoint |
| | | `gemini_sdk` = Gemini native SDK (generation or embeddings) |
| `pricing` | Cost information | `Pricing(input_per_mm=0.15, output_per_mm=0.60)` |
| `param_policy` | Parameter restrictions | `{"allowed": {...}, "disabled": {...}}` |
| `capabilities` | Model features | `{"reasoning_effort": True}` |
| `reasoning_policy` | Reasoning config | `{"mode": "gemini_level", "default": "low"}` |

## 🛡️ Parameter Validation System

The model registry includes a **comprehensive parameter validation system** that ensures only valid parameters reach the provider APIs and prevents unauthorized or unsupported parameters from being used.

### Parameter Policy Structure

Each model can define a `param_policy` with two components:

```python
param_policy={
    "allowed": {"max_output_tokens", "temperature", "top_p"},  # Permitted provider params
    "disabled": {"reasoning_effort", "include_thoughts"}     # Filtered out params
}
```

### Validation Rules

1. **`allowed` lists** - Only these provider-specific parameters can reach the API
2. **`disabled` lists** - These parameters are always filtered out
3. **Framework parameters** (`model`, `input`, `messages`, `stream`) are never in `allowed` lists
4. **`reasoning_effort`** - Never disabled for models with `reasoning_policy`

### Example: Provider Isolation

```python
# OpenAI model - Gemini parameters blocked
"openai:gpt-4o-mini": ModelInfo(
    param_policy={
        "allowed": {"max_output_tokens", "temperature", "top_p"},
        "disabled": {"reasoning_effort", "include_thoughts", "thinking_level"}
    }
)

# Gemini model - Full Gemini support
"gemini:openai-3-flash-preview": ModelInfo(
    param_policy={
        "allowed": {"max_output_tokens", "reasoning_effort", "include_thoughts", "thinking_level"},
        "disabled": set()
    }
)
```

### Benefits

- **🔒 Provider Isolation**: Gemini parameters can't reach OpenAI APIs
- **🛡️ API Safety**: Invalid parameters are filtered out automatically
- **🎯 Clear Documentation**: `allowed` lists show supported parameters explicitly
- **🔄 Silent Protection**: Users don't see errors from parameter mismatches

### 🔧 Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `LLM_ADAPTER_ALLOWED_MODELS` | Restrict available models | `"openai:gpt-4o-mini,gemini:native-embed"` |
| `OPENAI_API_KEY` | OpenAI authentication | `sk-...` |
| `GEMINI_API_KEY` | Gemini authentication | `AIza...` |
| `OPENAI_BASE_URL` | Custom OpenAI endpoint | `https://api.openai.com/v1` |

### 🚨 Common Pitfalls

#### **❌ Don't: Use Provider-Native Names Directly**
```python
# This may fail if allowlist is enabled
adapter.create(model="gpt-4o-mini")  # ❌ Wrong
adapter.create(model="openai:gpt-4o-mini")  # ✅ Correct
```

#### **❌ Don't: Forget Registry Validation**
```python
# Always validate your custom registry
validate_registry(REGISTRY)  # ✅ Do this
```

#### **❌ Don't: Mix Up Provider and Model**
```python
"openai:gpt-4o-mini": ModelInfo(
    provider="openai",      # ✅ Correct
    model="gpt-4o-mini",    # ✅ Correct
    # NOT:
    # provider="gpt-4o-mini",  # ❌ Wrong
    # model="openai",          # ❌ Wrong
)
```

#### **❌ Don't: Disable reasoning_effort for Reasoning Models**
```python
# Wrong - reasoning_effort won't work for reasoning models
"openai:reasoning_o3-mini": ModelInfo(
    param_policy={"disabled": {"reasoning_effort"}}  # ❌ Wrong
)

# Correct - reasoning_effort is consumed by reasoning_policy
"openai:reasoning_o3-mini": ModelInfo(
    param_policy={"disabled": {"temperature", "top_p"}}  # ✅ Correct
)
```

#### **❌ Don't: Include Framework Parameters in allowed**
```python
# Wrong - framework params don't go through filtering
param_policy={"allowed": {"model", "input", "temperature"}}  # ❌ Wrong

# Correct - only provider-specific parameters
param_policy={"allowed": {"temperature", "top_p"}}  # ✅ Correct
```

### ✅ Best Practices Checklist

- [ ] **Use registry key format**: `{provider}:{model}`
- [ ] **Validate custom registry**: `validate_registry(REGISTRY)`
- [ ] **Set allowlist**: Use `LLM_ADAPTER_ALLOWED_MODELS` for production
- [ ] **Document custom models**: Add comments explaining special configurations
- [ ] **Test model access**: Verify custom models work before deployment
- [ ] **Check pricing**: Ensure cost information is accurate
- [ ] **Configure param_policy**: Set `allowed` and `disabled` lists for each model
- [ ] **Don't disable reasoning_effort**: For models with `reasoning_policy`
- [ ] **Exclude framework params**: Don't put `model`, `input`, `messages`, `stream` in `allowed`

### 📚 Next Steps

1. **Read the full documentation**: See detailed sections below
2. **Check examples**: Review `examples/custom_registry.py`
3. **Test integration**: Use `examples/import_custom_registry.py`
4. **Deploy with allowlist**: Set `LLM_ADAPTER_ALLOWED_MODELS` in production

---

## Summary

The **Model Registry** is the configuration system that powers **llm_adapter**. It enables:

- **Stable registry keys** for provider-agnostic calls
- **🛡️ Parameter validation/gating**: Registry-controlled filtering to prevent unauthorized/unsupported parameters
- **Explicit endpoint routing** (Responses, Chat Completions, Embeddings, Gemini SDK)
- **Policy-driven parameter mapping** (including reasoning/thinking controls)
- **Pricing + limits metadata** for usage/cost-aware apps
- **Custom registries** to override or extend defaults

### Parameter Control Architecture

```
User Call                    Model Registry                Provider API
─────────────────           ──────────────────           ──────────────────
llm_adapter.create(         REGISTRY["openai:gpt-4o"]     OpenAI API
  model="openai:gpt-4o",  → param_policy: {              → Only valid
  include_thoughts=True)    →   allowed: {...}              → parameters
  temperature=0.7          →   disabled: {...}             → reach API
                           → Filter parameters
                           → Remove invalid params
```

## Architecture Overview

```
User Call                    Model Registry                Provider API
─────────────────           ──────────────────           ──────────────────
llm_adapter.create(         REGISTRY["openai:gpt-4o"]     OpenAI API
    model="openai:gpt-4o" ) → ModelInfo(                  → Chat Completions
    temperature=0.7 )         provider="openai",           (gpt-4o)
                              model="gpt-4o",
                              endpoint="chat_completions",
                              param_policy={...},
                              capabilities={...}
                          )
```

## Core Components

### 1. ModelInfo Dataclass

The `ModelInfo` dataclass represents a single model configuration:

```python
@dataclass(frozen=True)
class ModelInfo:
    provider: Provider                    # "openai", "gemini"
    model: str                           # Provider-native model name
    endpoint: Endpoint                   # API endpoint type
    pricing: Optional[Pricing]           # Cost information
    param_policy: Dict[str, Any]         # Parameter restrictions/mappings
    limits: Dict[str, Any]              # Model limitations
    reasoning_policy: Dict[str, Any]    # Reasoning configuration
    thinking_tax: Dict[str, Any]         # Thinking token costs
    reasoning_parameter: Optional[Tuple[str, Any]]  # Reasoning param mapping
    capabilities: Dict[str, Any]         # Model features
```

### 2. Registry Structure

The default registry (`REGISTRY`) is a dictionary mapping registry keys to `ModelInfo` objects:

```python
REGISTRY: Dict[str, ModelInfo] = {
    "openai:gpt-4o-mini": ModelInfo(
        provider="openai",
        model="gpt-4o-mini", 
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.15, output_per_mm=0.60),
        param_policy={"disabled": {"reasoning_effort", "stream"}},
        capabilities={"assistant_role": "assistant"},
    ),
    # ... more models
}
```

### 3. Integration with LLMAdapter

The `LLMAdapter` class uses the registry for:

- **Model Resolution**: `_lookup_model_info_from_registry()` finds `ModelInfo` by registry key
- **Parameter Mapping**: Converts generic parameters to provider-specific formats
- **Endpoint Routing**: Selects correct API endpoint based on `ModelInfo.endpoint`
- **Capability Validation**: Enforces model-specific parameter restrictions

## Registry Key Format

Registry keys follow the pattern: `{provider}:{model_identifier}`

### Examples:
- `"openai:gpt-4o-mini"` - OpenAI GPT-4o Mini model
- `"gemini:native-embed"` - Gemini native embedding model  
- `"openai:reasoning_o3-mini"` - OpenAI O3 Mini reasoning model

### Key Benefits:
- **Provider Agnostic**: Users don't need to know provider-specific model names
- **Explicit Intent**: Clear which provider and model being used
- **Namespace Separation**: Avoids naming conflicts between providers

## Model Configuration Fields

### Core Identification
- **`provider`**: LLM provider ("openai", "gemini")
- **`model`**: Provider-native model name ("gpt-4o-mini", "gemini-embedding-001")
- **`endpoint`**: API endpoint type ("responses", "chat_completions", "embeddings", "gemini_sdk")

### Behavioral Configuration
- **`param_policy`**: Parameter restrictions and mappings
  ```python
  param_policy={
      "disabled": {"reasoning_effort", "stream"},  # Disallowed parameters
      "mapped": {"temperature": "temperature"},     # Parameter mappings
  }
  ```

- **`limits`**: Model limitations
  ```python
  limits={
      "max_output_tokens": 2000,  # Token limits
      "max_input_length": 1000000,  # Input length limits
  }
  ```

- **`reasoning_policy`**: Reasoning model configuration
  ```python
  reasoning_policy={
      "mode": "openai_effort",      # Reasoning mode
      "default": "low",             # Default reasoning level
  }
  ```

### Metadata
- **`pricing`**: Cost information
  ```python
  Pricing(
      input_per_mm=0.15,           # $0.15 per 1M input tokens
      output_per_mm=0.60,          # $0.60 per 1M output tokens  
      cached_input_per_mm=0.075    # $0.075 per 1M cached tokens
  )
  ```

- **`capabilities`**: Model features
  ```python
  capabilities={
      "assistant_role": "assistant",     # Response role
      "reasoning_effort": True,           # Supports reasoning
      "dimensions": 1536,                 # Embedding dimensions
  }
  ```

## Provider-Specific Configurations

### OpenAI Models
```python
"openai:gpt-4o-mini": ModelInfo(
    provider="openai",
    model="gpt-4o-mini",
    endpoint="chat_completions",  # Uses OpenAI Chat Completions API
    param_policy={"disabled": {"reasoning_effort"}},
    capabilities={"assistant_role": "assistant"},
)
```

### Gemini Models  
```python
"gemini:openai-3-flash-preview": ModelInfo(
    provider="gemini", 
    model="models/gemini-3-flash-preview",
    endpoint="chat_completions",  # Uses Gemini OpenAI-compatible endpoint
    reasoning_policy={
        "mode": "gemini_level",
        "param": "thinking_level",
        "default": "minimal",
    },
    capabilities={"reasoning_effort": True},
)
```

### Native SDK Models
```python
"gemini:native-sdk-3-flash-preview": ModelInfo(
    provider="gemini",
    model="models/gemini-3-flash-preview", 
    endpoint="gemini_sdk",  # Uses Gemini native SDK
    reasoning_policy={
        "mode": "gemini_level",
        "param": "thinking_level", 
        "default": "low",
    },
)
```

### Endpoint Routing Logic

The `chat_completions` endpoint is **shared** between providers:

```python
# LLMAdapter determines actual client based on provider + endpoint
if provider == "openai" and endpoint == "chat_completions":
    client = OpenAI()  # OpenAI Chat Completions API
elif provider == "gemini" and endpoint == "chat_completions":
    client = OpenAI(base_url=gemini_openai_base_url)  # Gemini OpenAI-compatible
elif provider == "gemini" and endpoint == "gemini_sdk":
    client = genai.GenerativeModel()  # Gemini native SDK
```

**Key Points:**
- **`chat_completions`** = OpenAI-style API (used by both OpenAI and Gemini)
- **`gemini_sdk`** = Gemini native SDK (generation and embeddings)
- **Provider field** determines which client to instantiate
- **Endpoint field** determines which API/SDK to use

## Extending the Registry

### Creating Custom Registries

Users can create custom registries to:
- Add new models not in the default registry
- Override configurations of existing models
- Add private/internal models
- Configure custom endpoints or parameters

### Example: Custom Registry (`examples/custom_registry.py`)

```python
from llm_adapter.model_registry import ModelInfo, Pricing, validate_registry

# Custom registry with new models
REGISTRY = {
    "openai:custom_reasoning_o3-mini": ModelInfo(
        provider="openai",
        model="o3-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=1.10, output_per_mm=4.40),
        limits={"max_output_tokens": 2000},
        capabilities={
            "assistant_role": "assistant",
            "reasoning_effort": True,
        },
        param_policy={"disabled": {"stream", "temperature", "top_p"}},
        reasoning_policy={
            "mode": "openai_effort",
            "default": "low",
        },
        reasoning_parameter=("reasoning_effort", "low"),
    ),
    
    "openai:custom_gpt-4-turbo": ModelInfo(
        provider="openai", 
        model="gpt-4-turbo",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.30, output_per_mm=1.20),
        capabilities={"assistant_role": "assistant"},
    ),
}

# Validate the custom registry
validate_registry(REGISTRY)
```

### Using Custom Registries

#### Method 1: Direct Usage
```python
from llm_adapter import LLMAdapter
from my_custom_registry import REGISTRY as CUSTOM_REGISTRY

adapter = LLMAdapter(model_registry=CUSTOM_REGISTRY)
response = adapter.create(
    model="openai:custom_reasoning_o3-mini",
    input="Hello, world!"
)
```

#### Method 2: Merging with Default Registry
```python
from llm_adapter import LLMAdapter
from llm_adapter.model_registry import REGISTRY as DEFAULT_REGISTRY
from my_custom_registry import REGISTRY as CUSTOM_REGISTRY

# Merge registries (custom takes precedence)
MERGED_REGISTRY = {**DEFAULT_REGISTRY, **CUSTOM_REGISTRY}
adapter = LLMAdapter(model_registry=MERGED_REGISTRY)

# Both default and custom models available
response1 = adapter.create(model="openai:gpt-4o-mini")  # Default
response2 = adapter.create(model="openai:custom_gpt-4-turbo")  # Custom
```

#### Method 3: Demo UI Integration
The demo UI supports custom registry merging via the UI checkbox:

```python
# In demo API (src/llm_adapter_demo/api.py)
if merge_custom_registry:
    from examples.custom_registry import REGISTRY as CUSTOM_REGISTRY
    registry = {**model_registry.REGISTRY, **CUSTOM_REGISTRY}
else:
    registry = model_registry.REGISTRY
```

## Registry Validation

The `validate_registry()` function ensures registry integrity:

```python
def validate_registry(registry: Dict[str, ModelInfo], *, strict: bool = True) -> None:
    """Validate a model registry mapping.
    
    Checks:
    - registry is a non-empty dict
    - provider and endpoint are valid
    - pricing values are non-negative numbers (when pricing is present)
    - limits/capabilities/param_policy/reasoning_policy/thinking_tax are dicts
    """
```

### Validation Rules:
- **Non-empty registry**: Must contain at least one model
- **Valid providers**: Only supported provider names
- **Valid endpoints**: Only supported endpoint types
- **Pricing validation**: Non-negative cost values
- **Structure validation**: Dict fields must be dictionaries
- **Type checking**: Fields match expected types

## Advanced Features

### Reasoning Models
Reasoning models have special configuration:

```python
"openai:reasoning_o3-mini": ModelInfo(
    # ... basic config ...
    reasoning_policy={
        "mode": "openai_effort",
        "default": "low",
    },
    reasoning_parameter=("reasoning_effort", "low"),  # Maps parameter
    thinking_tax={
        "effort_map": {
            "none": {"reserve_ratio": 0.0},
            "low": {"reserve_ratio": 0.25}, 
            "medium": {"reserve_ratio": 0.50},
            "high": {"reserve_ratio": 0.80},
        },
        "kind": "budget",
    },
)
```

### Parameter Policies
Control which parameters are allowed or mapped:

```python
param_policy={
    "disabled": {"stream", "temperature"},  # Disallow these
    "mapped": {"top_p": "top_p"},           # Map parameter names
}
```

### Model Limits
Define model-specific constraints:

```python
limits={
    "max_output_tokens": 2000,
    "max_input_length": 1000000,
    "supports_functions": True,
}
```

## Best Practices

### 1. Registry Key Naming
- Use consistent `{provider}:{model}` format
- Include model capabilities in key name (e.g., `reasoning`, `embed`)
- Keep keys descriptive but concise

### 2. Configuration Organization
- Group related fields logically
- Use consistent ordering (core → behavioral → metadata)
- Document special configurations with comments

### 3. Custom Registry Design
- **Override vs Extend**: Decide whether to replace or supplement defaults
- **Validation**: Always validate custom registries
- **Documentation**: Document custom models and their special behaviors

### 4. Performance Considerations
- Registry lookup is O(1) dictionary access
- ModelInfo objects are immutable (frozen dataclass)
- Consider memory usage for very large registries

## Integration Examples

### Basic Usage
```python
from llm_adapter import LLMAdapter

adapter = LLMAdapter()
response = adapter.create(
    model="openai:gpt-4o-mini",  # Registry key lookup
    input="Hello, world!",
    temperature=0.7
)
```

### With Custom Registry
```python
from llm_adapter import LLMAdapter
from my_custom_registry import REGISTRY

adapter = LLMAdapter(model_registry=REGISTRY)
response = adapter.create(
    model="my-company:custom-model",
    input="Hello, world!"
)
```

### Model Information Access
```python
# Get model info
model_info = adapter.model_registry["openai:gpt-4o-mini"]
print(f"Provider: {model_info.provider}")
print(f"Pricing: ${model_info.pricing.input_per_mm}/1M input tokens")

# Get pricing
pricing = adapter.get_pricing_for_model("openai:gpt-4o-mini")
```

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   LLMError: Model 'unknown:model' not found in registry
   ```
   **Solution**: Check registry key spelling and ensure model is in registry

2. **Invalid Provider**
   ```
   LLMError: Provider 'invalid_provider' not supported
   ```
   **Solution**: Use supported provider names ("openai", "gemini")

3. **Parameter Not Allowed**
   ```
   LLMError: Parameter 'reasoning_effort' not allowed for this model
   ```
   **Solution**: Check model's `param_policy` for allowed parameters

### Debug Information

Enable debug mode to see registry lookups:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

adapter = LLMAdapter()
# Debug output will show registry lookups and parameter mapping
```

## Conclusion

The Model Registry is the foundation that enables the LLM Adapter's provider-agnostic interface. By centralizing model configurations, it provides:

- **Unified API**: Single interface for multiple providers
- **Flexibility**: Easy to extend with custom models
- **Safety**: Built-in validation and parameter checking  
- **Transparency**: Clear model capabilities and limitations
- **Maintainability**: Centralized configuration management

For most users, the default registry provides comprehensive coverage of popular models. For specialized use cases, custom registries enable seamless integration of private or experimental models while maintaining the same unified interface.
