"""Simple CLI test for llm_adapter chat via NVIDIA NIM (Nemotron).

Usage:
  python examples/nvidia_nemotron_example.py "Your prompt here"

Requirements:
  - NVIDIA_API_KEY must be set in the environment (or via a .env you load yourself).
    Get one at https://build.nvidia.com (NVIDIA hosted) or point NVIDIA_BASE_URL
    at a self-hosted NIM container.
  - The llm-adapter package must be installed (e.g., `pip install -e .`).

Note: This example specifically requires NVIDIA_API_KEY for NVIDIA NIM chat.
"""

from __future__ import annotations

import os
import sys


def get_prompt_from_argv() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return input("Enter a prompt for llm_adapter (NVIDIA NIM): ")


def main() -> None:
    try:
        from llm_adapter import llm_adapter  # type: ignore[import]
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Could not import llm_adapter: {e}")
        sys.exit(1)

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("[WARNING] NVIDIA_API_KEY is not set; NVIDIA calls will fail.")

    prompt = get_prompt_from_argv().strip()
    if not prompt:
        print("[ERROR] Empty prompt; nothing to send.")
        sys.exit(1)

    print("=== llm_adapter.create (model='nvidia:nemotron-3-super-120b') ===")
    print("Model: nvidia:nemotron-3-super-120b")
    print(f"Prompt: {prompt}")
    print("----------------------------------------")

    try:
        resp = llm_adapter.create(
            model="nvidia:nemotron-3-super-120b",
            input=prompt,
            stream=False,
        )
    except Exception as e:
        print(f"[ERROR] llm_adapter.create failed: {e}")
        sys.exit(1)

    print("\n=== Response ===")
    print(getattr(resp, "output_text", "") or "<no text returned>")

    usage = getattr(resp, "usage", None)
    if isinstance(usage, dict):
        print("\n=== Usage (best-effort) ===")
        for k, v in usage.items():
            print(f"{k}: {v}")

    # Reasoning example: budget-based thinking via reasoning_effort knob.
    print("\n=== llm_adapter.create (model='nvidia:nemotron-3.5-lightning-30b', reasoning_effort='medium') ===")
    try:
        resp = llm_adapter.create(
            model="nvidia:nemotron-3.5-lightning-30b",
            input=prompt,
            stream=False,
            reasoning_effort="medium",
        )
        result = llm_adapter.normalize_adapter_response(resp, provider="nvidia")
        print("\n=== Reasoning ===")
        print(result.get("reasoning") or "<no reasoning returned>")
        print("\n=== Answer ===")
        print(result.get("text") or "<no text returned>")
        print("\n=== Usage (normalized) ===")
        print(result.get("usage"))
    except Exception as e:
        print(f"[ERROR] reasoning call failed: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
