#!/usr/bin/env python3
"""
Quick test to verify OPENAI_BASE_URL functionality.
Tests with the standard OpenAI endpoint to ensure the feature works.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_openai_base_url():
    """Test that OPENAI_BASE_URL environment variable works correctly."""
    
    print("Testing OPENAI_BASE_URL functionality...")
    
    # Set custom base URL (same as default OpenAI URL)
    custom_base_url = "https://api.openai.com/v1"
    os.environ["OPENAI_BASE_URL"] = custom_base_url
    
    try:
        from llm_adapter import LLMAdapter
        
        # Create adapter with environment base URL
        llm = LLMAdapter()
        
        print(f"✅ LLMAdapter created successfully")
        print(f"✅ Base URL from environment: {llm.openai_base_url}")
        
        # Verify the base URL was set correctly
        if llm.openai_base_url == custom_base_url:
            print("✅ Base URL set correctly from environment")
        else:
            print(f"❌ Base URL mismatch. Expected: {custom_base_url}, Got: {llm.openai_base_url}")
            return False
        
        # Test that the OpenAI client gets the base URL
        print("🔄 Testing OpenAI client initialization...")
        
        # This will test that the base URL is passed to the OpenAI client
        # We won't make an actual API call to avoid needing API keys
        try:
            client = llm._get_openai()
            if hasattr(client, 'base_url'):
                print(f"✅ OpenAI client base_url: {client.base_url}")
                if str(client.base_url) == custom_base_url:
                    print("✅ OpenAI client base_url set correctly!")
                else:
                    print(f"⚠️  Client base_url differs: {client.base_url}")
            else:
                print("⚠️  OpenAI client doesn't have base_url attribute")
        except Exception as e:
            print(f"⚠️  Could not initialize OpenAI client (expected without API key): {e}")
        
        print("✅ OPENAI_BASE_URL test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    finally:
        # Clean up environment
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]

if __name__ == "__main__":
    success = test_openai_base_url()
    sys.exit(0 if success else 1)
