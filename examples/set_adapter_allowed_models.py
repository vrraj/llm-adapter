"""Example: Set adapter allowed models.

This example demonstrates how to configure an LLMAdapter with an allowlist
of permitted model keys for security and control.

Usage:
  python set_adapter_allowed_models.py

Requirements:
  - Set OPENAI_API_KEY and/or GEMINI_API_KEY for testing
  - export OPENAI_API_KEY=...
  - export GEMINI_API_KEY=...
"""

import os
from llm_adapter import LLMAdapter

def main():
    print("=== LLMAdapter Allowed Models Example ===")
    print()
    
    # Example 1: Create adapter with specific allowed models
    print("1. Creating adapter with allowed model allowlist:")
    allowed_models = {"openai:gpt-4o-mini", "openai:embed_small"}
    llm = LLMAdapter(allowed_model_keys=allowed_models)
    
    print(f"   Allowed models: {allowed_models}")
    print(f"   Total models in registry: {len(llm.model_registry)}")
    print(f"   Allowed models count: {len(llm.allowed_model_keys)}")
    print()
    
    # Example 2: Test model lookup with allowlist
    print("2. Testing model lookups:")
    
    # This should work (in allowlist)
    try:
        model_info = llm._lookup_model_info_from_registry("openai:gpt-4o-mini")
        print(f"   ✅ openai:gpt-4o-mini found: {model_info.model}")
    except Exception as e:
        print(f"   ❌ openai:gpt-4o-mini error: {e}")
    
    # This should fail (not in allowlist)
    try:
        model_info = llm._lookup_model_info_from_registry("openai:gpt-4o")
        print(f"   ❌ openai:gpt-4o unexpectedly found: {model_info.model}")
    except Exception as e:
        print(f"   ✅ openai:gpt-4o correctly blocked: {type(e).__name__}")
    
    # This should fail (provider-native name without prefix)
    try:
        model_info = llm._lookup_model_info_from_registry("gpt-4o-mini")
        print(f"   ❌ gpt-4o-mini unexpectedly found: {model_info.model}")
    except Exception as e:
        print(f"   ✅ gpt-4o-mini correctly requires registry key: {type(e).__name__}")
    print()
    
    # Example 3: Environment variable fallback
    print("3. Environment variable configuration:")
    old_env = os.environ.get("LLM_ADAPTER_ALLOWED_MODELS")
    
    # Set environment variable
    os.environ["LLM_ADAPTER_ALLOWED_MODELS"] = "gemini:native-embed, openai:embed_large"
    
    # Create adapter without explicit allowlist (will use env var)
    llm_env = LLMAdapter()
    print(f"   From environment: {llm_env.allowed_model_keys}")
    
    # Restore original environment
    if old_env is None:
        os.environ.pop("LLM_ADAPTER_ALLOWED_MODELS", None)
    else:
        os.environ["LLM_ADAPTER_ALLOWED_MODELS"] = old_env
    
    print()
    print("✅ Allowed models configuration complete!")
    print()
    print("Key takeaways:")
    print("- Use allowed_model_keys to restrict which models can be used")
    print("- Always use full registry keys (e.g., 'openai:gpt-4o-mini')")
    print("- Can also set via LLM_ADAPTER_ALLOWED_MODELS environment variable")
    print("- Provider-native names without prefix are blocked for security")


if __name__ == "__main__":
    main()

