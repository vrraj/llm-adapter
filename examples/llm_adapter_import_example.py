"""
Quick smoke test for vrraj-llm-adapter.

Make sure you set:
  export OPENAI_API_KEY=...

Then run:
  python llm_adapter_import_example.py
"""

import os
from dataclasses import asdict, is_dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from llm_adapter import llm_adapter, LLMError


def handle_llm_error(error: LLMError, title: str = "Error") -> None:
    """Simple error handler for LLMAdapter calls.
    
    Args:
        error: The LLMError exception
        title: Context title for the error (e.g., "Chat Error", "Embedding Error")
    """
    print(f"\n=== {title} ===")
    print(f"Error: {error}")
    print(f"Code: {error.code}")
    print(f"Provider: {error.provider}")
    print(f"Model: {error.model}")
    print(f"Hint: Check model name, API keys, or LLM_ADAPTER_ALLOWED_MODELS")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first: export OPENAI_API_KEY=... (or set it in your environment)")
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY first: export GEMINI_API_KEY=... (or set it in your environment)")

    # Show allowed models if LLM_ADAPTER_ALLOWED_MODELS is set
    allowed_models = os.getenv("LLM_ADAPTER_ALLOWED_MODELS")
    if allowed_models:
        print(f"LLM_ADAPTER_ALLOWED_MODELS is set to: {allowed_models}")
        models_list = [m.strip() for m in allowed_models.split(",") if m.strip()]
        print(f"Allowed models ({len(models_list)}):")
        for model in models_list:
            print(f"  - {model}")
        print()
    else:
        print("LLM_ADAPTER_ALLOWED_MODELS is not set - all models are allowed")
        print()

    # ---- Chat ----
    try:
        resp = llm_adapter.create(
            model="openai:gpt-4o-mini",
            input=[{"role": "user", "content": "Hello"}],
            max_output_tokens=200,
        )

        print("\n=== Chat Response ===")
        print(resp.output_text)
        print("Usage:", resp.usage)
    except LLMError as e:
        handle_llm_error(e, "Chat Error")

    # ---- Chat (Reasoning Effort example) ----
    try:
        resp2 = llm_adapter.create(
            model="openai:reasoning_o3-mini",
            input="Explain why the sky is blue in 3 bullets.",
            reasoning_effort="low",
            max_output_tokens=250,
        )

        print("\n=== Chat Response (reasoning_effort) ===")
        print(resp2.output_text)
        print("Usage:", resp2.usage)
    except LLMError as e:
        handle_llm_error(e, "Chat (reasoning_effort) Error")

    # ---- Chat (Gemini sdk Reasoning Effort example) ----
    try:
        resp3 = llm_adapter.create(
            model="gemini:native-sdk-reasoning-2.5-flash",
            input="Explain why the sky is blue in 3 bullets.",
            reasoning_effort="low",
            max_output_tokens=250,
        )

        print("\n=== Gemini SDK Response (reasoning_effort) ===")
        print(resp3.output_text)
        print("Usage:", resp3.usage)
    except LLMError as e:
        handle_llm_error(e, "Gemini SDK Error")

    # ---- Embeddings ----
    try:
        emb_resp = llm_adapter.create_embedding(
            model="openai:embed_small",
            input=["Hello world", "How are you?", "This is a test"],
        )

        print("\n=== Embedding Response ===")
        print(f"Embeddings generated: {len(emb_resp.data)} vectors")
        for i, emb in enumerate(emb_resp.data[:3]):  # Show first 3
            print(f"Embedding {i+1} (first 7 dims): {emb[:7]}...")
        
        usage_obj = emb_resp.usage
        if is_dataclass(usage_obj):
            usage_obj = asdict(usage_obj)
        elif hasattr(usage_obj, "__dict__"):
            usage_obj = dict(usage_obj.__dict__)
        print("Usage:", usage_obj)
    except LLMError as e:
        handle_llm_error(e, "Embedding Error")


if __name__ == "__main__":
    main()
