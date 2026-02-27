# Release Notes

## Version 0.3.0 - Documentation Restructuring & API Clarity

### Overview
**Major Documentation Enhancement**: Complete documentation restructuring with improved API clarity, consolidated examples, and enhanced developer experience.

### Key Features
- **📋 Comprehensive API Reference**: New `API_REFERENCE.md` with complete method signatures, parameter details, and response structures
- **🎯 Stability Indicators**: Clear distinction between guaranteed API surfaces and debug/internal fields
- **📚 Consolidated Examples**: Restructured examples with `examples/README.md` providing comprehensive navigation
- **🔄 Improved Naming**: Renamed key examples for clarity (`llm_adapter_import_example.py` → `llm_adapter_basic_usage.py`)
- **✅ Model Key Consistency**: Ensured all documentation uses correct registry keys across all files
- **🧹 Documentation Cleanup**: Removed redundant files and consolidated information

### Documentation Structure Changes
- **New**: `API_REFERENCE.md` - Complete API documentation with stability indicators
- **New**: `examples/README.md` - Comprehensive examples guide with learning paths
- **Removed**: `AdapterResponse.md` - Content consolidated into API reference
- **Removed**: `EMBEDDING_MAGNITUDE.md` - Content consolidated into API reference
- **Updated**: `README.md` - Enhanced with quick API overview and better navigation

### API Clarity Improvements
- **Stability Indicators**: ✅ Guaranteed fields vs 🔧 Debug/Opaque fields clearly marked
- **Singleton Clarification**: Added note about `llm_adapter` being a convenience singleton instance
- **Method Naming Consistency**: Standardized on `get_pricing_for_model()` as canonical name
- **Parameter Stability**: Clear documentation of stable vs provider-specific parameters

### Example Improvements
- **Renamed**: `llm_adapter_import_example.py` → `llm_adapter_basic_usage.py` for clarity
- **Enhanced**: `examples/README.md` with learning paths, use case categorization, and complexity levels
- **Updated**: All examples to use correct registry model keys
- **Test Naming**: Improved test function names for clarity (`test_llm_adapter_import` → `test_llm_adapter_singleton_import`)

### Model Key Consistency
- **Fixed**: All documentation now uses registry keys (e.g., `"openai:embed_small"`) instead of native model names
- **Validated**: All model keys across README, API reference, examples, and tests match registry
- **Updated**: Fixed invalid keys in test files to use correct registry models

### Developer Experience
- **Quick Start**: Enhanced README with immediate API overview
- **Navigation**: Clear links between documentation files
- **Learning Paths**: Structured progression from basic to advanced usage
- **Troubleshooting**: Improved error handling and debugging guidance

### Breaking Changes
- **File Structure**: Removed redundant documentation files (content moved to consolidated locations)
- **Example Names**: `llm_adapter_import_example.py` renamed to `llm_adapter_basic_usage.py`
- **Test Names**: Internal test function names updated for clarity

### Migration Guide
- **Documentation Users**: Update bookmarks to use new `API_REFERENCE.md` for complete API docs
- **Example Users**: Update references from `llm_adapter_import_example.py` to `llm_adapter_basic_usage.py`
- **No Code Changes**: All existing code continues to work unchanged

### Documentation Updates
- **README.md**: Added "Public API (overview)" section with quick reference
- **API_REFERENCE.md**: Complete method signatures, response structures, and usage patterns
- **examples/README.md**: Comprehensive guide with learning paths and categorization
- **Model Registry**: All examples now use correct registry keys

---

## Version 0.2.0 - Parameter Validation System

### Overview
**Major Enhancement**: Added comprehensive parameter validation system that prevents cross-provider parameter contamination and API failures through registry-based parameter gating.

### Key Features
- **🛡️ Parameter Validation System**: Registry-controlled parameter filtering with explicit `allowed`/`disabled` lists
- **🔒 Provider Isolation**: Gemini parameters cannot reach OpenAI APIs and vice versa
- **🎯 Explicit Parameter Control**: Each model defines exactly which parameters are permitted
- **🔄 Silent Protection**: Invalid parameters filtered out automatically without user errors
- **⚙️ Enhanced Registry**: All models now include comprehensive `param_policy` configurations
- **📚 Updated Documentation**: Complete parameter validation documentation and examples

### Breaking Changes
- **Method Rename**: `_filter_kwargs_by_capabilities` → `_apply_registry_param_policy`
- **Stricter Validation**: Some parameters that previously passed through may now be filtered (improves API safety)

### Parameter Validation Examples
```python
# Before: Could cause API failures
llm_adapter.create(
    model="openai:gpt-4o-mini",
    include_thoughts=True,  # ❌ Would reach OpenAI API and fail
    temperature=0.7
)

# After: Automatically filtered
llm_adapter.create(
    model="openai:gpt-4o-mini", 
    include_thoughts=True,  # ✅ Filtered out silently
    temperature=0.7        # ✅ Allowed through
)
# Result: Only valid parameters reach provider APIs
```

### Registry Parameter Policies
All models now include explicit parameter policies:
```python
"openai:gpt-4o-mini": ModelInfo(
    param_policy={
        "allowed": {"max_output_tokens", "temperature", "top_p"},
        "disabled": {"reasoning_effort", "include_thoughts", "thinking_level"}
    }
)
```

### Benefits
- **API Safety**: Invalid parameters never reach provider APIs
- **Clear Documentation**: `allowed` lists show supported parameters explicitly
- **Provider Compatibility**: Cross-provider parameter contamination prevented
- **Better UX**: Users don't see cryptic API errors from invalid parameters

### Documentation Updates
- **README.md**: Added comprehensive parameter validation system documentation
- **MODEL_REGISTRY.md**: Updated with parameter policy examples and best practices
- **Examples**: Updated custom registry examples with proper parameter policies

### Migration Guide
- **Custom Registries**: Add `param_policy` with `allowed`/`disabled` lists to all models
- **Method Calls**: No changes needed for existing code (filtering is automatic)
- **Examples**: See `examples/custom_registry.py` for updated parameter policies

---

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
