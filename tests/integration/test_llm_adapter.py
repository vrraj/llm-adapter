import pytest
from llm_adapter import llm_adapter

@pytest.mark.integration
def test_chat_and_embeddings():
    """Test chat and embeddings functionality."""
    # Chat
    resp = llm_adapter.create(
        model="openai:gpt-4o-mini",
        input=[{"role": "user", "content": "Hello"}],
        max_output_tokens=200,
    )
    print("Chat Response:")
    print(resp.output_text)
    print(f"Usage: {getattr(resp, 'usage', 'Usage info not available')}")

    # Embeddings
    emb_resp = llm_adapter.create_embedding(
        model="openai:embed_small",
        input=["Hello world", "How are you?", "This is a test"]
    )
    print("Embedding Response:")
    for i, emb in enumerate(emb_resp.data):
        print(f"Embedding {i+1} (First 7 vectors): {emb[:7]}...")
    print(f"Usage: {getattr(emb_resp, 'usage', 'Usage info not available')}")
