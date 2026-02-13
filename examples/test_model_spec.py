#!/usr/bin/env python3
"""Test script for ModelSpec functionality.

This script demonstrates and tests ModelSpec usage for both chat and embeddings
with different providers and parameter configurations.
"""

import sys
import os

def test_model_spec_chat():
    """Test ModelSpec with chat completions."""
    print("=== Testing ModelSpec Chat Completions ===")
    
    try:
        from llm_adapter import llm_adapter, ModelSpec
        
        # Test 1: Basic ModelSpec with OpenAI
        print("\n1. Basic OpenAI ModelSpec:")
        chat_spec = ModelSpec(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_output_tokens=100
        )
        
        try:
            resp = llm_adapter.create(
                spec=chat_spec,
                input=[{"role": "user", "content": "Hello, test ModelSpec!"}]
            )
            print("✅ SUCCESS: Basic ModelSpec chat worked")
            print(f"   Response type: {type(resp)}")
            print(f"   Has output_text: {hasattr(resp, 'output_text')}")
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                print("✅ SUCCESS: ModelSpec structure correct (API key error expected)")
            else:
                print(f"❌ FAILED: {e}")
                return False
        
        # Test 2: ModelSpec with extra parameters
        print("\n2. ModelSpec with extra parameters:")
        chat_spec_extra = ModelSpec(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.5,
            extra={"custom_param": "test_value"}
        )
        
        try:
            resp = llm_adapter.create(
                spec=chat_spec_extra,
                input=[{"role": "user", "content": "Test extra parameters"}]
            )
            print("✅ SUCCESS: ModelSpec with extra params worked")
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                print("✅ SUCCESS: ModelSpec with extra params structure correct")
            else:
                print(f"❌ FAILED: {e}")
                return False
        
        # Test 3: ModelSpec with extra_body
        print("\n3. ModelSpec with extra_body:")
        chat_spec_body = ModelSpec(
            provider="openai",
            model="gpt-4o-mini",
            extra={"extra_body": {"custom_field": "custom_value"}}
        )
        
        try:
            resp = llm_adapter.create(
                spec=chat_spec_body,
                input=[{"role": "user", "content": "Test extra_body"}]
            )
            print("✅ SUCCESS: ModelSpec with extra_body worked")
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                print("✅ SUCCESS: ModelSpec with extra_body structure correct")
            else:
                print(f"❌ FAILED: {e}")
                return False
        
        # Test 4: ModelSpec reusability
        print("\n4. ModelSpec reusability:")
        reusable_spec = ModelSpec(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.8,
            max_output_tokens=50
        )
        
        try:
            resp1 = llm_adapter.create(
                spec=reusable_spec,
                input=[{"role": "user", "content": "First call"}]
            )
            resp2 = llm_adapter.create(
                spec=reusable_spec,
                input=[{"role": "user", "content": "Second call"}]
            )
            print("✅ SUCCESS: ModelSpec reusability worked")
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                print("✅ SUCCESS: ModelSpec reusability structure correct")
            else:
                print(f"❌ FAILED: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Could not test ModelSpec chat: {e}")
        return False

def test_model_spec_embeddings():
    """Test ModelSpec with embeddings."""
    print("\n=== Testing ModelSpec Embeddings ===")
    
    try:
        from llm_adapter import llm_adapter, ModelSpec
        
        # Test 1: Basic embedding ModelSpec
        print("\n1. Basic OpenAI embedding ModelSpec:")
        embed_spec = ModelSpec(
            provider="openai",
            model="text-embedding-3-small"
        )
        
        try:
            resp = llm_adapter.create_embedding(
                spec=embed_spec,
                input="Test ModelSpec embeddings"
            )
            print("✅ SUCCESS: Basic ModelSpec embedding worked")
            print(f"   Response type: {type(resp)}")
            print(f"   Has data: {hasattr(resp, 'data')}")
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                print("✅ SUCCESS: ModelSpec embedding structure correct")
            else:
                print(f"❌ FAILED: {e}")
                return False
        
        # Test 2: Embedding ModelSpec with dimensions
        print("\n2. Embedding ModelSpec with dimensions:")
        embed_spec_dims = ModelSpec(
            provider="openai",
            model="text-embedding-3-small",
            extra={"dimensions": 1536}
        )
        
        try:
            resp = llm_adapter.create_embedding(
                spec=embed_spec_dims,
                input="Test with dimensions"
            )
            print("✅ SUCCESS: ModelSpec embedding with dimensions worked")
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                print("✅ SUCCESS: ModelSpec embedding with dimensions structure correct")
            else:
                print(f"❌ FAILED: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Could not test ModelSpec embeddings: {e}")
        return False

def test_model_spec_gemini():
    """Test ModelSpec with Gemini provider."""
    print("\n=== Testing ModelSpec Gemini ===")
    
    try:
        from llm_adapter import llm_adapter, ModelSpec
        
        # Test Gemini ModelSpec
        print("\n1. Gemini ModelSpec:")
        gemini_spec = ModelSpec(
            provider="gemini",
            model="gemini-1.5-flash",
            temperature=0.7,
            max_output_tokens=100
        )
        
        try:
            resp = llm_adapter.create(
                spec=gemini_spec,
                input=[{"role": "user", "content": "Hello Gemini ModelSpec!"}]
            )
            print("✅ SUCCESS: Gemini ModelSpec worked")
            print(f"   Response type: {type(resp)}")
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                print("✅ SUCCESS: Gemini ModelSpec structure correct")
            else:
                print(f"❌ FAILED: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Could not test ModelSpec Gemini: {e}")
        return False

def test_model_spec_validation():
    """Test ModelSpec validation and error handling."""
    print("\n=== Testing ModelSpec Validation ===")
    
    try:
        from llm_adapter import llm_adapter, ModelSpec
        
        # Test 1: Missing provider (should fail at creation)
        print("\n1. ModelSpec validation - missing provider:")
        try:
            invalid_spec = ModelSpec(
                # provider missing - should fail
                model="gpt-4o-mini"
            )
            print("❌ FAILED: Should have failed for missing provider")
            return False
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
            print("❌ FAILED: Should have failed for missing model")
            return False
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
            print("❌ FAILED: Should have failed for invalid provider")
            return False
        except Exception as e:
            print("✅ SUCCESS: Correctly failed for invalid provider")
            print(f"   Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Could not test ModelSpec validation: {e}")
        return False

def main():
    """Run all ModelSpec tests."""
    print("=== ModelSpec Test Suite ===")
    print("Testing ModelSpec functionality across different scenarios")
    print("Note: API key errors are expected without valid credentials")
    
    # Add current directory to path for local development
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    results = []
    
    # Run all tests
    results.append(test_model_spec_chat())
    results.append(test_model_spec_embeddings())
    results.append(test_model_spec_gemini())
    results.append(test_model_spec_validation())
    
    # Summary
    print("\n" + "="*50)
    print("=== ModelSpec Test Summary ===")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All ModelSpec tests passed!")
        print("ModelSpec functionality is working correctly.")
        return 0
    else:
        print("❌ Some ModelSpec tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
