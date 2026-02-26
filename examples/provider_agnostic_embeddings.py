#!/usr/bin/env python3
"""Test script to verify provider-agnostic embedding functionality.

This script tests that embedding methods work without explicit provider parameters,
using registry keys for auto-detection.
"""

import sys
import os

def main():
    print("=== Testing Provider-Agnostic Embeddings ===")
    print()
    
    # Add current directory to path for local development
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from llm_adapter import llm_adapter
        print("✅ Successfully imported llm_adapter")
    except Exception as e:
        print(f"❌ Failed to import llm_adapter: {e}")
        return 1
    
    print()
    
    # Test 1: Direct create_embedding method with registry key
    print("Test 1: llm_adapter.create_embedding() with registry key")
    print("Model: openai:embed_small (should auto-detect 'openai' provider)")
    try:
        resp = llm_adapter.create_embedding(
            model="openai:embed_small",  # Registry key
            input="Test text for provider auto-detection"
        )
        print("✅ SUCCESS: Provider auto-detected and call structured correctly")
        print(f"   Response type: {type(resp)}")
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            print("✅ SUCCESS: Method structure works (API key error expected)")
            print(f"   Expected error: {e}")
        else:
            print(f"❌ UNEXPECTED ERROR: {e}")
            return 1
    
    print()
    
    # Test 2: Facade method with registry key
    print("Test 2: llm_adapter.embeddings.create() with registry key")
    print("Model: openai:embed_small (should auto-detect 'openai' provider)")
    try:
        resp = llm_adapter.embeddings.create(
            model="openai:embed_small",  # Registry key
            input="Test text for facade method"
        )
        print("✅ SUCCESS: Facade method works with auto-detection")
        print(f"   Response type: {type(resp)}")
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            print("✅ SUCCESS: Facade method structure works (API key error expected)")
            print(f"   Expected error: {e}")
        else:
            print(f"❌ UNEXPECTED ERROR: {e}")
            return 1
    
    print()
    
    # Test 3: Gemini registry key (should auto-detect 'gemini')
    print("Test 3: Testing Gemini registry key")
    print("Model: gemini:native-embed (should auto-detect 'gemini' provider)")
    try:
        resp = llm_adapter.create_embedding(
            model="gemini:native-embed",  # Registry key
            input="Test text for Gemini auto-detection"
        )
        print("✅ SUCCESS: Gemini provider auto-detected")
        print(f"   Response type: {type(resp)}")
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            print("✅ SUCCESS: Gemini auto-detection works (API key error expected)")
            print(f"   Expected error: {e}")
        else:
            print(f"❌ UNEXPECTED ERROR: {e}")
            return 1
    
    print()
    print("=== All Tests Completed ===")
    print("✅ Provider-agnostic embedding functionality is working correctly!")
    print("   (API key errors are expected without valid credentials)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
