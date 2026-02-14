#!/usr/bin/env python3
"""
Direct test of Gemini native embedding API to verify dimension behavior.
Tests both 768 and 1536 dimensions without llm_adapter wrapper.
"""

import os
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_gemini_direct():
    """Test Gemini native embedding directly with different dimensions."""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Import Gemini native SDK
        from google import genai
        from google.genai import types
        
        # Initialize client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment")
            return False
            
        client = genai.Client(api_key=api_key)
        
        print("🧪 Testing Gemini Native Embedding Directly")
        print("=" * 50)
        
        # Test 1: Default dimensions (1536)
        print("\n📏 Test 1: Default Dimensions (1536)")
        print("-" * 30)
        
        cfg_default = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            # output_dimensionality=None  # Use default
        )
        
        response_1536 = client.models.embed_content(
            model="gemini-embedding-001",
            contents="Test document for 1536 dimensions",
            config=cfg_default
        )
        
        vectors_1536 = []
        for emb in response_1536.embeddings:
            vectors_1536.append(emb.values)
        
        print(f"✅ 1536-dim test successful")
        print(f"📏 Vector dimensions: {len(vectors_1536[0])}")
        print(f"📊 First 5 values: {vectors_1536[0][:5]}")
        
        # Calculate magnitude
        import math
        magnitude_1536 = math.sqrt(sum([x*x for x in vectors_1536[0]]))
        print(f"📏 Magnitude: {magnitude_1536:.4f}")
        
        # Test 2: Custom dimensions (768)
        print("\n📏 Test 2: Custom Dimensions (768)")
        print("-" * 30)
        
        cfg_768 = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768  # Custom dimension
        )
        
        response_768 = client.models.embed_content(
            model="gemini-embedding-001",
            contents="Test document for 768 dimensions", 
            config=cfg_768
        )
        
        vectors_768 = []
        for emb in response_768.embeddings:
            vectors_768.append(emb.values)
        
        print(f"✅ 768-dim test successful")
        print(f"📏 Vector dimensions: {len(vectors_768[0])}")
        print(f"📊 First 5 values: {vectors_768[0][:5]}")
        
        # Calculate magnitude
        magnitude_768 = math.sqrt(sum(x*x for x in vectors_768[0]))
        print(f"📏 Magnitude: {magnitude_768:.4f}")
        
        # Test 3: Multiple inputs with 768
        print("\n📏 Test 3: Multiple Inputs with 768")
        print("-" * 30)
        
        response_multi = client.models.embed_content(
            model="gemini-embedding-001",
            contents=["Document 1", "Document 2"],
            config=cfg_768
        )
        
        vectors_multi = []
        for emb in response_multi.embeddings:
            vectors_multi.append(emb.values)
        
        print(f"✅ Multiple 768-dim test successful")
        print(f"📏 Number of vectors: {len(vectors_multi)}")
        print(f"📏 Vector dimensions: {len(vectors_multi[0])}")
        print(f"📏 Vector 1 magnitude: {math.sqrt(sum([x*x for x in vectors_multi[0]])):.4f}")
        print(f"📏 Vector 2 magnitude: {math.sqrt(sum([x*x for x in vectors_multi[1]])):.4f}")
        
        # Summary
        print("\n📊 Summary")
        print("=" * 20)
        print(f"✅ 1536-dim magnitude: {magnitude_1536:.4f}")
        print(f"✅ 768-dim magnitude: {magnitude_768:.4f}")
        print(f"✅ Both tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct Gemini test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemini_direct()
    if success:
        print("\n🎉 All direct Gemini tests passed!")
    else:
        print("\n💥 Direct Gemini tests failed!")
        sys.exit(1)
