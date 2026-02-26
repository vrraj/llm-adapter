"""Simple CLI test for llm_adapter chat via OpenAI.

Usage:
  python examples/test_openai_chat.py "Your prompt here"

Requirements:
  - OPENAI_API_KEY must be set in the environment (or via a .env you load yourself).
  - The llm-adapter package must be installed (e.g., `pip install -e .`).

Note: This example specifically requires OPENAI_API_KEY for OpenAI chat.
"""

from __future__ import annotations

import os
import sys
from typing import Any


def get_prompt_from_argv() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return input("Enter a prompt for llm_adapter (OpenAI): ")


def main() -> None:
    try:
        from llm_adapter import llm_adapter  # type: ignore[import]
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Could not import llm_adapter: {e}")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[WARNING] OPENAI_API_KEY is not set; OpenAI calls will fail.")

    prompt = get_prompt_from_argv().strip()
    if not prompt:
        print("[ERROR] Empty prompt; nothing to send.")
        sys.exit(1)

    print("=== llm_adapter.create (model='openai:gpt-4o-mini') ===")
    print("Model: openai:gpt-4o-mini")
    print(f"Prompt: {prompt}")
    print("----------------------------------------")

    try:
        resp = llm_adapter.create(
            model="openai:gpt-4o-mini",
            input=prompt,
            stream=False,
        )
    except Exception as e:
        print(f"[ERROR] llm_adapter.create failed: {e}")
        sys.exit(1)

    text: str = getattr(resp, "output_text", "") or ""
    print("\n=== Response ===")
    print(text or "<no text returned>")

    usage = getattr(resp, "usage", None)
    if usage is not None:
        print("\n=== Usage (best-effort) ===")
        if isinstance(usage, dict):
            for k, v in usage.items():
                print(f"{k}: {v}")
        else:
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = getattr(usage, field, None)
                if val is not None:
                    print(f"{field}: {val}")


if __name__ == "__main__":  # pragma: no cover
    main()
