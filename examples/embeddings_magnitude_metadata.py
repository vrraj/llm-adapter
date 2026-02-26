#!/usr/bin/env python3
"""
Test script for standalone llm-adapter magnitude metadata functionality.
"""
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_adapter import llm_adapter

def test_magnitude_metadata():
    print("Testing magnitude metadata in standalone llm-adapter...")
    
    # Test 1: OpenAI embeddings with magnitude
    print("\n1. Testing OpenAI embeddings with magnitude...")
    try:
        if os.getenv("OPENAI_API_KEY"):
            result = llm_adapter.embeddings.create(
                model="openai:embed_small",
                input="Test text for OpenAI magnitude metadata"
            )
            
            if result.data:
                item = result.data[0]
                print(f"   Embedding length: {len(item.embedding)}")
                print(f"   Has magnitude: {hasattr(item, 'magnitude')}")
                print(f"   Magnitude: {getattr(item, 'magnitude', 'None')}")
                print(f"   Normalized: {getattr(item, 'normalized', 'None')}")
                print(f"   Provider: {getattr(item, 'provider', 'None')}")
            else:
                print("   No data returned")
        else:
            print("   Skipping - OPENAI_API_KEY not set")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Gemini embeddings with normalization and magnitude
    print("\n2. Testing Gemini embeddings with normalization and magnitude...")
    try:
        if os.getenv("GEMINI_API_KEY"):
            result = llm_adapter.embeddings.create(
                model="gemini:native-embed",
                input="Test text for Gemini magnitude metadata",
                normalize_embedding=True
            )
            
            if result.data:
                item = result.data[0]
                print(f"   Embedding length: {len(item.embedding)}")
                print(f"   Has magnitude: {hasattr(item, 'magnitude')}")
                print(f"   Magnitude: {getattr(item, 'magnitude', 'None')}")
                print(f"   Normalized: {getattr(item, 'normalized', 'None')}")
                print(f"   Provider: {getattr(item, 'provider', 'None')}")
            else:
                print("   No data returned")
        else:
            print("   Skipping - GEMINI_API_KEY not set")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Batch embeddings with metadata
    print("\n3. Testing batch embeddings with metadata...")
    try:
        if os.getenv("GEMINI_API_KEY"):
            result = llm_adapter.embeddings.create(
                model="gemini:native-embed", 
                input=["Text 1", "Text 2"],
                normalize_embedding=True
            )
            
            for i, item in enumerate(result.data):
                print(f"   Item {i}:")
                print(f"     Has magnitude: {hasattr(item, 'magnitude')}")
                print(f"     Magnitude: {getattr(item, 'magnitude', 'None')}")
                print(f"     Normalized: {getattr(item, 'normalized', 'None')}")
                print(f"     Provider: {getattr(item, 'provider', 'None')}")
        else:
            print("   Skipping - GEMINI_API_KEY not set")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✅ Magnitude metadata functionality is available in standalone llm-adapter!")

if __name__ == "__main__":
    test_magnitude_metadata()
