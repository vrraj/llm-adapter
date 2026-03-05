#!/usr/bin/env python3
"""
Test script to verify the tool sanitization fix works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_adapter.llm_adapter import LLMAdapter

def test_tool_sanitization():
    """Test that flat tool format gets converted to nested format."""
    
    adapter = LLMAdapter()
    
    # Test flat format (what our application sends)
    flat_tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location name"},
                    "unit": {"type": "string", "enum": ["C", "K", "F"], "default": "C"}
                },
                "required": ["location"],
                "additionalProperties": False  # This should be stripped
            }
        }
    ]
    
    print("Original flat tools:")
    import json
    print(json.dumps(flat_tools, indent=2))
    
    # Test the sanitization
    sanitized = adapter._sanitize_tools_for_gemini_adapter(flat_tools)
    
    print("\nSanitized tools:")
    print(json.dumps(sanitized, indent=2))
    
    # Verify the conversion
    assert len(sanitized) == 1, "Should have one tool"
    tool = sanitized[0]
    
    assert tool["type"] == "function", "Type should be function"
    assert "function" in tool, "Should have nested function"
    
    func = tool["function"]
    assert func["name"] == "get_weather", "Name should be preserved"
    assert func["description"] == "Get weather for a location", "Description should be preserved"
    assert "parameters" in func, "Should have parameters"
    
    params = func["parameters"]
    assert "additionalProperties" not in params, "additionalProperties should be stripped"
    assert params["required"] == ["location"], "Required should be preserved"
    
    print("\n✅ All tests passed! Tool sanitization works correctly.")

def test_with_real_model():
    """Test with a real model call (without actually calling API)."""
    
    adapter = LLMAdapter()
    
    # Test tools in flat format
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location name"}
                },
                "required": ["location"]
            }
        }
    ]
    
    # This should not raise an exception during tool processing
    try:
        # Just test the tool processing part
        sanitized_tools = adapter._sanitize_tools_for_gemini_adapter(tools)
        print("✅ Real model test: Tool processing successful")
        print(f"   Input: {len(tools)} flat tools")
        print(f"   Output: {len(sanitized_tools)} sanitized tools")
    except Exception as e:
        print(f"❌ Real model test failed: {e}")
        raise

if __name__ == "__main__":
    print("🧪 Testing llm-adapter tool sanitization fix...")
    print("=" * 60)
    
    test_tool_sanitization()
    print("\n" + "=" * 60)
    test_with_real_model()
    
    print("\n🎉 All tests passed! The fix is working correctly.")
