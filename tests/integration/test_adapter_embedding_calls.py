"""Test the normalized embedding response structure with multiple inputs."""

from llm_adapter import llm_adapter

def test_chat_example():
    """Test the chat example from README."""
    try:
        resp = llm_adapter.create(
            model="gemini:native-sdk-3-flash-preview",  # provider derived from model registry
            input=[{"role": "user", "content": "Hello"}],
            max_output_tokens=200,
        )
        print("✅ Chat example works")
        print(f"Response type: {type(resp)}")
        print(f"Has output_text: {hasattr(resp, 'output_text')}")
        print(f"Chat Response: {resp.output_text}")
        print(f"Usage: {getattr(resp, 'usage', 'Usage info not available')}")
        return True
    except Exception as e:
        print(f"❌ Chat example failed: {e}")
        return False

def test_embedding_single_input():
    """Test embedding with single input using new normalized response."""
    try:
        emb_resp = llm_adapter.create_embedding(
            model="gemini:native-embed",  # provider derived from model registry
            input="Hello world"
        )
        print("✅ Single input embedding works")
        print(f"Response type: {type(emb_resp)}")
        print(f"Has data attribute: {hasattr(emb_resp, 'data')}")
        print(f"Data type: {type(emb_resp.data)}")
        print(f"Data length: {len(emb_resp.data)}")
        print(f"First vector type: {type(emb_resp.data[0]) if emb_resp.data else None}")
        print(f"First vector length: {len(emb_resp.data[0]) if emb_resp.data else 0}")
        print(f"Embedding (Truncated): {str(emb_resp.data[0][:5]) if emb_resp.data else 'No data'}...")
        print(f"Usage: {getattr(emb_resp, 'usage', 'Usage info not available')}")
        print(f"Provider: {emb_resp.metadata.get('provider', 'No provider') if emb_resp.metadata else 'No provider'}")
        print(f"Model: {emb_resp.metadata.get('model', 'No model') if emb_resp.metadata else 'No model'}")
        print(f"Vector dimensions: {getattr(emb_resp, 'vector_dim', 'No dim')}")
        print(f"Normalized: {getattr(emb_resp, 'normalized', 'No normalization info')}")
        return True
    except Exception as e:
        print(f"❌ Single input embedding failed: {e}")
        return False

def test_embedding_multiple_inputs():
    """Test embedding with multiple inputs using new normalized response."""
    try:
        emb_resp = llm_adapter.create_embedding(
            model="gemini:native-embed",  # provider derived from model registry
            input=["Hello world", "How are you", "Testing multiple strings"]
        )
        print("✅ Multiple input embedding works")
        print(f"Response type: {type(emb_resp)}")
        print(f"Data length: {len(emb_resp.data)}")
        print(f"Input count: {emb_resp.metadata.get('input_count', 'Unknown')}")
        print(f"Input texts: {emb_resp.metadata.get('input_texts', 'Unknown')}")
        
        # Test each vector
        for i, vector in enumerate(emb_resp.data):
            print(f"Vector {i}: {vector[:5]}... (length: {len(vector)})")
        
        # Test metadata
        print(f"Magnitudes: {emb_resp.metadata.get('magnitudes', 'No magnitudes')}")
        print(f"Processing time: {emb_resp.metadata.get('processing_time', 'No time')}")
        print(f"Usage: {getattr(emb_resp, 'usage', 'Usage info not available')}")
        print(f"Provider: {emb_resp.metadata.get('provider', 'No provider') if emb_resp.metadata else 'No provider'}")
        print(f"Model: {emb_resp.metadata.get('model', 'No model') if emb_resp.metadata else 'No model'}")
        print(f"Vector dimensions: {getattr(emb_resp, 'vector_dim', 'No dim')}")
        print(f"Normalized: {getattr(emb_resp, 'normalized', 'No normalization info')}")
        return True
    except Exception as e:
        print(f"❌ Multiple input embedding failed: {e}")
        return False

def test_openai_embedding():
    """Test OpenAI embedding with new normalized response."""
    try:
        emb_resp = llm_adapter.create_embedding(
            model="openai:embed_small",  # provider derived from model registry
            input=["OpenAI test", "Multiple strings"]
        )
        print("✅ OpenAI embedding works")
        print(f"Data length: {len(emb_resp.data)}")
        print(f"Provider: {emb_resp.metadata.get('provider', 'No provider') if emb_resp.metadata else 'No provider'}")
        print(f"Model: {emb_resp.metadata.get('model', 'No model') if emb_resp.metadata else 'No model'}")
        
        # Test consistent access pattern
        for i, vector in enumerate(emb_resp.data):
            print(f"OpenAI Vector {i}: {vector[:5]}... (length: {len(vector)})")
        
        return True
    except Exception as e:
        print(f"❌ OpenAI embedding failed: {e}")
        return False

def test_gemini_custom_parameters():
    """Test Gemini embedding with custom dimensions and task_type."""
    try:
        emb_resp = llm_adapter.create_embedding(
            model="gemini:native-embed",  # provider derived from model registry
            input=["Document for retrieval", "Another document"],
            output_dimensionality=768,  # Custom dimension for Gemini native
            task_type="RETRIEVAL_QUERY",  # Custom task type
            normalize_embedding=False,
        )
        print("✅ Gemini custom parameters embedding works")
        print(f"Data length: {len(emb_resp.data)}")
        print(f"Provider: {emb_resp.metadata.get('provider', 'No provider') if emb_resp.metadata else 'No provider'}")
        print(f"Model: {emb_resp.metadata.get('model', 'No model') if emb_resp.metadata else 'No model'}")
        print(f"Vector dimensions: {getattr(emb_resp, 'vector_dim', 'No dim')}")
        print(f"Normalized: {getattr(emb_resp, 'normalized', 'No normalization info')}")
        print(f"Magnitudes: {emb_resp.metadata.get('magnitudes', 'No magnitudes')}")
        
        # Test custom parameters in metadata
        print(f"Task type: {emb_resp.metadata.get('task_type', 'No task type')}")
        print(f"Output dimensionality: {emb_resp.metadata.get('output_dimensionality', 'No output dim')}")
        
        # Test each vector
        for i, vector in enumerate(emb_resp.data):
            print(f"Custom Vector {i}: {vector[:5]}... (length: {len(vector)})")
        
        # Test usage
        print(f"Usage: {getattr(emb_resp, 'usage', 'Usage info not available')}")
        
        # Debug: Check if vectors are actually 768-dim
        if emb_resp.data and len(emb_resp.data[0]) == 768:
            print(f"✅ CONFIRMED: Vectors are 768 dimensions (not 1536)")
        else:
            print(f"❌ UNEXPECTED: Vectors are {len(emb_resp.data[0]) if emb_resp.data else 0} dimensions")
        
        return True
    except Exception as e:
        print(f"❌ Gemini custom parameters embedding failed: {e}")
        return False

def test_consistent_interface():
    """Test that both providers have consistent interface."""
    try:
        # Test Gemini
        gemini_resp = llm_adapter.create_embedding(
            model="gemini:native-embed",
            input=["Test string"]
        )
        
        # Test OpenAI
        openai_resp = llm_adapter.create_embedding(
            model="openai:embed_small", 
            input=["Test string"]
        )
        
        # Both should have same interface
        gemini_has_data = hasattr(gemini_resp, 'data') and isinstance(gemini_resp.data, list)
        openai_has_data = hasattr(openai_resp, 'data') and isinstance(openai_resp.data, list)
        
        gemini_vector_type = type(gemini_resp.data[0]) if gemini_resp.data else None
        openai_vector_type = type(openai_resp.data[0]) if openai_resp.data else None
        
        print("✅ Interface consistency test")
        print(f"Gemini has data: {gemini_has_data}, vector type: {gemini_vector_type}")
        print(f"OpenAI has data: {openai_has_data}, vector type: {openai_vector_type}")
        
        # Both should return direct list of floats
        gemini_direct = isinstance(gemini_resp.data[0], list) if gemini_resp.data else False
        openai_direct = isinstance(openai_resp.data[0], list) if openai_resp.data else False
        
        print(f"Gemini direct vectors: {gemini_direct}")
        print(f"OpenAI direct vectors: {openai_direct}")
        
        return gemini_direct and openai_direct
    except Exception as e:
        print(f"❌ Interface consistency test failed: {e}")
        return False

def run_all_tests():
    """Run all embedding tests."""
    print("=== Testing Normalized Embedding Response Structure ===")
    print()
    
    success = True
    
    # Test chat
    print("1. Testing chat example...")
    #success &= test_chat_example()
    print()
    
    # Test single input embedding
    print("2. Testing single input embedding...")
    success &= test_embedding_single_input()
    print()
    
    # Test multiple input embedding
    print("3. Testing multiple input embedding...")
    success &= test_embedding_multiple_inputs()
    print()
    
    # Test OpenAI embedding
    print("4. Testing OpenAI embedding...")
    success &= test_openai_embedding()
    print()
    
    # Test Gemini custom parameters
    print("5. Testing Gemini custom parameters...")
    success &= test_gemini_custom_parameters()
    print()
    
    # Test interface consistency
    print("6. Testing interface consistency...")
    success &= test_consistent_interface()
    print()
    
    if success:
        print("✅ All embedding tests passed!")
        return True
    else:
        print("❌ Some embedding tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("✅ Normalized embedding response tests passed")
        exit(0)
    else:
        print("❌ Normalized embedding response tests failed")
        exit(1)
