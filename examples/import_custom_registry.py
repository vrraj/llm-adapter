#!/usr/bin/env python3
"""
Test script to verify custom registry functionality.
This script tests importing and using a custom registry with LLMAdapter.
"""

import os
import sys
from pathlib import Path

# Add src to path so we can import llm_adapter
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add examples to path so we can import custom_registry
examples_path = str(Path(__file__).parent)
if examples_path not in sys.path:
    sys.path.insert(0, examples_path)

def test_custom_registry():
    """Test that LLMAdapter can be instantiated with a custom registry."""
    
    print("Testing custom registry functionality...")
    
    try:
        # Import the custom registry
        from custom_registry import REGISTRY as USER_REGISTRY
        print("✓ Successfully imported custom registry")
        
        # Import LLMAdapter
        from llm_adapter import LLMAdapter
        print("✓ Successfully imported LLMAdapter")
        
        # Test 1: Create adapter with custom registry
        llm = LLMAdapter(model_registry=USER_REGISTRY)
        print("✓ Successfully created LLMAdapter with custom registry")
        
        # Test 2: Verify custom models are available
        custom_models = ["openai:custom_reasoning_o3-mini", "openai:custom_reasoning_gpt-5-mini"]
        for model_key in custom_models:
            model_info = llm.model_registry.get(model_key)
            if model_info:
                print(f"✓ Found custom model: {model_key}")
            else:
                print(f"✗ Missing custom model: {model_key}")
                return False
        
        # Test 3: Verify default models are still available (merged registry)
        default_models = ["openai:gpt-4o-mini", "openai:embed_small"]
        for model_key in default_models:
            model_info = llm.model_registry.get(model_key)
            if model_info:
                print(f"✓ Found default model after merge: {model_key}")
            else:
                print(f"✗ Missing default model after merge: {model_key}")
                return False
        
        # Test 4: Test pricing lookup for custom models
        for model_key in custom_models:
            pricing = llm.get_pricing_for_model(model_key)
            if pricing:
                print(f"✓ Found pricing for custom model {model_key}: {pricing}")
            else:
                print(f"✗ Missing pricing for custom model: {model_key}")
                return False
        
        # Test 5: Validate the merged registry
        from llm_adapter.model_registry import validate_registry
        try:
            validate_registry(llm.model_registry, strict=False)
            print("✓ Merged registry validation passed")
        except Exception as e:
            print(f"✗ Merged registry validation failed: {e}")
            return False
        
        print("\n🎉 All tests passed! Custom registry functionality works correctly.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_model_resolution():
    """Test that model resolution works with custom registry."""
    
    print("\nTesting model resolution with custom registry...")
    
    try:
        from custom_registry import REGISTRY as USER_REGISTRY
        from llm_adapter import LLMAdapter
        
        llm = LLMAdapter(model_registry=USER_REGISTRY)
        
        # Test resolving custom model names
        test_cases = [
            ("openai:custom_reasoning_o3-mini", "o3-mini"),
            ("openai:custom_reasoning_gpt-5-mini", "gpt-5-mini"),
        ]
        
        for model_key, expected_model in test_cases:
            resolved = llm._resolve_provider_model_name(model_key)
            if resolved == expected_model:
                print(f"✓ Resolved {model_key} -> {resolved}")
            else:
                print(f"✗ Failed to resolve {model_key}. Expected: {expected_model}, Got: {resolved}")
                return False
        
        print("✓ Model resolution tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Model resolution test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Custom Registry Integration Test")
    print("=" * 60)
    
    success = test_custom_registry() and test_model_resolution()
    
    if success:
        print("\n✅ All tests passed! Ready to implement UI feature.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Fix issues before implementing UI feature.")
        sys.exit(1)
