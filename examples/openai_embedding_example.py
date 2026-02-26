"""Simple CLI test for llm_adapter embeddings via OpenAI.

Usage:
  python examples/test_openai_embeddings.py "Text to embed"

Requirements:
  - OPENAI_API_KEY must be set in the environment.
  - The llm-adapter package must be installed (e.g., `pip install -e .`).

Note: This example specifically requires OPENAI_API_KEY for OpenAI embeddings.
"""

from __future__ import annotations

import os
import sys
from typing import Any


def get_text_from_argv() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return input("Enter text to embed via llm_adapter (OpenAI): ")


def main() -> None:
    try:
        from llm_adapter import llm_adapter  # type: ignore[import]
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Could not import llm_adapter: {e}")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[WARNING] OPENAI_API_KEY is not set; OpenAI calls will fail.")

    model = os.getenv("TEST_EMBEDDING_MODEL", "openai:embed_small")
    text = get_text_from_argv().strip()
    if not text:
        print("[ERROR] Empty text; nothing to embed.")
        sys.exit(1)

    print("=== llm_adapter.embeddings.create (model registry key) ===")
    print(f"Model: {model}")
    print(f"Text: {text}")
    print("----------------------------------------")

    try:
        resp = llm_adapter.embeddings.create(
            model=model,
            input=text,
        )
    except Exception as e:
        print(f"[ERROR] llm_adapter.embeddings.create failed: {e}")
        sys.exit(1)

    try:
        data = getattr(resp, "data", None) or []
        if not data:
            print(f"[ERROR] No data field on embedding response: {resp!r}")
            sys.exit(1)
        embedding = getattr(data[0], "embedding", None)
        if embedding is None:
            print(f"[ERROR] No embedding field on first data item: {data[0]!r}")
            sys.exit(1)

        print(f"Embedding length: {len(embedding)}")
        sample_size = min(8, len(embedding))
        print(f"Embedding sample (first {sample_size} values): {embedding[:sample_size]}")
    except Exception as e:
        print(f"[ERROR] Failed to inspect embedding response: {e}")
        sys.exit(1)

    usage = getattr(resp, "usage", None)
    if usage is not None:
        print("\n=== Usage (best-effort) ===")
        if isinstance(usage, dict):
            for k, v in usage.items():
                print(f"{k}: {v}")
        else:
            for field in ("prompt_tokens", "total_tokens"):
                val = getattr(usage, field, None)
                if val is not None:
                    print(f"{field}: {val}")


if __name__ == "__main__":  # pragma: no cover
    main()
