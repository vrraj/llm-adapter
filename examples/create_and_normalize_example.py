#!/usr/bin/env python3
"""
Example: Create and normalize.

This example shows the recommended two-step flow:
- `llm_adapter.create()` returns an `AdapterResponse` (provider boundary)
- `llm_adapter.normalize_adapter_response()` returns an `LLMResult` (stable app schema)

Usage:
  python examples/create_and_normalize_example.py

Requirements:
  - OPENAI_API_KEY and/or GEMINI_API_KEY
"""

import os
from typing import Any, Dict

from llm_adapter import llm_adapter, LLMError


def create_result(**kwargs) -> Dict[str, Any]:
    """Create via `llm_adapter.create()` and normalize to `LLMResult`."""
    resp = llm_adapter.create(**kwargs)
    return llm_adapter.normalize_adapter_response(resp)


def main() -> None:
    print("Create → Normalize Example")

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))

    if not has_openai and not has_gemini:
        print("Missing API keys. Set at least one of OPENAI_API_KEY or GEMINI_API_KEY.")
        return

    model_key = "openai:gpt-4o-mini" if has_openai else "gemini:openai-2.5-flash-lite"

    # 1) Recommended path: stable app-facing contract
    try:
        result = create_result(
            model=model_key,
            input="Write a one-sentence summary of artificial intelligence."
        )

        print(f"Model: {result['model']}")
        print(f"Text: {result['text']}")
        if result.get("reasoning"):
            print("Reasoning: (present)")
        print(f"Usage: {result.get('usage')}")
        tool_calls = result.get("tool_calls") or []
        print(f"Tool calls: {len(tool_calls)}")

    except LLMError as e:
        print(f"LLMError: {e.code} - {e}")
        return

    # 2) Provider boundary: show raw vs display-safe text
    try:
        resp = llm_adapter.create(model=model_key, input="Say 'Hello world!' and nothing else.")
        normalized = llm_adapter.normalize_adapter_response(resp)

        print("\nRaw vs normalized:")
        print(f"Raw output_text: {resp.output_text}")
        print(f"Normalized text: {normalized['text']}")

    except LLMError as e:
        print(f"LLMError: {e.code} - {e}")


if __name__ == "__main__":
    main()
