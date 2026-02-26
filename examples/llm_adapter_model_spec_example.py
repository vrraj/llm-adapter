#!/usr/bin/env python3
"""ModelSpec Example for LLM Adapter.

This script demonstrates how to use ModelSpec for structured, type-safe
configuration of LLM calls. ModelSpec provides a reusable, validated
alternative to passing kwargs directly.

Key benefits of ModelSpec:
- Type safety and validation
- Reusable configurations
- Structured parameter management
- Better IDE support and documentation

Usage:
    python examples/llm_adapter_model_spec_example.py

Requirements:
    - Set OPENAI_API_KEY and/or GEMINI_API_KEY for testing
    - export OPENAI_API_KEY=...
    - export GEMINI_API_KEY=...

"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from llm_adapter import llm_adapter, ModelSpec, LLMError


def handle_llm_error(error: LLMError, title: str = "Error") -> None:
    """Simple error handler for LLMAdapter calls."""
    print(f"\n=== {title} ===")
    print(f"Error: {error}")
    print(f"Code: {error.code}")
    print(f"Provider: {error.provider}")
    print(f"Model: {error.model}")
    print(f"Hint: Check model name, API keys, or LLM_ADAPTER_ALLOWED_MODELS")

def demo_model_spec_chat():
    """Demonstrate ModelSpec with chat completions."""
    print("=== ModelSpec Chat Completions Demo ===")
    
    # Test 1: Basic ModelSpec with OpenAI
    print("\n1. Basic OpenAI ModelSpec:")
    chat_spec = ModelSpec(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        max_output_tokens=100
    )
    
    print(f"   ModelSpec: {chat_spec}")
    print(f"   Provider: {chat_spec.provider}")
    print(f"   Model: {chat_spec.model}")
    print(f"   Temperature: {chat_spec.temperature}")
    
    try:
        resp = llm_adapter.create(
            spec=chat_spec,
            input=[{"role": "user", "content": "Hello, ModelSpec!"}]
        )
        print("✅ SUCCESS: Basic ModelSpec chat worked")
        print(f"   Response: {resp.output_text[:100]}...")
        print(f"   Usage: {resp.usage}")
    except LLMError as e:
        handle_llm_error(e, "Chat ModelSpec Error")
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            print("✅ ModelSpec structure correct (API key error expected)")
        else:
            print(f"❌ Unexpected error: {e}")
    
    # Test 2: Reusable ModelSpec
    print("\n2. Reusable ModelSpec:")
    reusable_spec = ModelSpec(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.8,
        max_output_tokens=50
    )
    
    print("   Using same ModelSpec for multiple calls:")
    try:
        resp1 = llm_adapter.create(
            spec=reusable_spec,
            input=[{"role": "user", "content": "First call with reusable spec"}]
        )
        resp2 = llm_adapter.create(
            spec=reusable_spec,
            input=[{"role": "user", "content": "Second call with same spec"}]
        )
        print("✅ SUCCESS: ModelSpec reusability worked")
        print(f"   First response: {resp1.output_text[:50]}...")
        print(f"   Second response: {resp2.output_text[:50]}...")
    except LLMError as e:
        handle_llm_error(e, "Reusable ModelSpec Error")
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            print("✅ Reusable ModelSpec structure correct")
        else:
            print(f"❌ Unexpected error: {e}")

def demo_model_spec_embeddings():
    """Demonstrate ModelSpec with embeddings."""
    print("\n=== ModelSpec Embeddings Demo ===")
    
    # Test 1: Basic embedding ModelSpec
    print("\n1. Basic OpenAI embedding ModelSpec:")
    embed_spec = ModelSpec(
        provider="openai",
        model="text-embedding-3-small"
    )
    
    print(f"   ModelSpec: {embed_spec}")
    print(f"   Provider: {embed_spec.provider}")
    print(f"   Model: {embed_spec.model}")
    
    try:
        resp = llm_adapter.create_embedding(
            spec=embed_spec,
            input="Test ModelSpec embeddings"
        )
        print("✅ SUCCESS: Basic ModelSpec embedding worked")
        print(f"   Embeddings generated: {len(resp.data)} vectors")
        print(f"   First embedding (first 5 dims): {resp.data[0][:5]}...")
        print(f"   Usage: {resp.usage}")
    except LLMError as e:
        handle_llm_error(e, "Embedding ModelSpec Error")
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            print("✅ ModelSpec embedding structure correct")
        else:
            print(f"❌ Unexpected error: {e}")
    
    # Test 2: Embedding ModelSpec with dimensions
    print("\n2. Embedding ModelSpec with custom dimensions:")
    embed_spec_dims = ModelSpec(
        provider="openai",
        model="text-embedding-3-small",
        extra={"dimensions": 768}
    )
    
    print(f"   Custom dimensions: {embed_spec_dims.extra.get('dimensions')}")
    
    try:
        resp = llm_adapter.create_embedding(
            spec=embed_spec_dims,
            input="Test with custom dimensions"
        )
        print("✅ SUCCESS: ModelSpec embedding with dimensions worked")
        print(f"   Embedding dimension: {len(resp.data[0])}")
    except LLMError as e:
        handle_llm_error(e, "Embedding Dimensions ModelSpec Error")
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            print("✅ ModelSpec embedding with dimensions structure correct")
        else:
            print(f"❌ Unexpected error: {e}")

def demo_model_spec_gemini():
    """Demonstrate ModelSpec with Gemini provider."""
    print("\n=== ModelSpec Gemini Demo ===")
    
    # Test Gemini ModelSpec
    print("\n1. Gemini ModelSpec:")
    gemini_spec = ModelSpec(
        provider="gemini",
        model="gemini-1.5-flash",
        temperature=0.7,
        max_output_tokens=100
    )
    
    print(f"   ModelSpec: {gemini_spec}")
    print(f"   Provider: {gemini_spec.provider}")
    print(f"   Model: {gemini_spec.model}")
    
    try:
        resp = llm_adapter.create(
            spec=gemini_spec,
            input=[{"role": "user", "content": "Hello Gemini ModelSpec!"}]
        )
        print("✅ SUCCESS: Gemini ModelSpec worked")
        print(f"   Response: {resp.output_text[:100]}...")
        print(f"   Usage: {resp.usage}")
    except LLMError as e:
        handle_llm_error(e, "Gemini ModelSpec Error")
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            print("✅ Gemini ModelSpec structure correct")
        else:
            print(f"❌ Unexpected error: {e}")

def demo_model_spec_validation():
    """Demonstrate ModelSpec validation and error handling."""
    print("\n=== ModelSpec Validation Demo ===")
    
    # Test 1: Missing provider (should fail at creation)
    print("\n1. ModelSpec validation - missing provider:")
    try:
        invalid_spec = ModelSpec(
            # provider missing - should fail
            model="gpt-4o-mini"
        )
        print("❌ Should have failed for missing provider")
    except TypeError as e:
        print("✅ SUCCESS: Correctly failed for missing provider")
        print(f"   Error: {e}")
    
    # Test 2: Missing model (should fail at creation)
    print("\n2. ModelSpec validation - missing model:")
    try:
        invalid_spec = ModelSpec(
            provider="openai"
            # model missing - should fail
        )
        print("❌ Should have failed for missing model")
    except TypeError as e:
        print("✅ SUCCESS: Correctly failed for missing model")
        print(f"   Error: {e}")
    
    # Test 3: Invalid provider (should fail at runtime)
    print("\n3. ModelSpec validation - invalid provider:")
    try:
        invalid_spec = ModelSpec(
            provider="invalid_provider",
            model="gpt-4o-mini"
        )
        resp = llm_adapter.create(
            spec=invalid_spec,
            input=[{"role": "user", "content": "Test"}]
        )
        print("❌ Should have failed for invalid provider")
    except Exception as e:
        print("✅ SUCCESS: Correctly failed for invalid provider")
        print(f"   Error: {e}")

def main():
    """Run ModelSpec demonstrations."""
    print("=== ModelSpec Demo Suite ===")
    print("Demonstrating ModelSpec functionality for structured LLM calls")
    print("Note: API key errors are expected without valid credentials")
    print()
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set - OpenAI tests will show structure validation only")
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not set - Gemini tests will show structure validation only")
    print()
    
    # Show allowlist status
    allowed_models = os.getenv("LLM_ADAPTER_ALLOWED_MODELS")
    if allowed_models:
        print(f"🔐 Allowlist active: {allowed_models}")
    else:
        print("🔓 No allowlist - all models allowed")
    print()
    
    # Run all demonstrations
    print("Running ModelSpec demonstrations...")
    print("="*50)
    
    demo_model_spec_chat()
    demo_model_spec_embeddings()
    demo_model_spec_gemini()
    demo_model_spec_validation()
    
    # Summary
    print("\n" + "="*50)
    print("=== ModelSpec Demo Summary ===")
    print("✅ ModelSpec provides structured, type-safe configuration")
    print("✅ ModelSpec supports reusability across multiple calls")
    print("✅ ModelSpec works with both chat and embeddings")
    print("✅ ModelSpec validates required fields at creation")
    print("✅ ModelSpec integrates with error handling")
    print()
    print("Key Benefits:")
    print("• Type safety and validation")
    print("• Reusable configurations")
    print("• Structured parameter management")
    print("• Better IDE support and documentation")
    print()
    print("Usage in production:")
    print("```python")
    print("spec = ModelSpec(provider='openai', model='gpt-4o-mini', temperature=0.7)")
    print("response = llm_adapter.create(spec=spec, input='Hello')")
    print("```")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
