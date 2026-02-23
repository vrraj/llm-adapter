"""Example: Get model pricing information.

Usage:
  python get_model_pricing_example.py <model_key>

Examples:
  python get_model_pricing_example.py openai:gpt-4o-mini
  python get_model_pricing_example.py gemini:native-embed
"""

import sys
from llm_adapter import LLMAdapter

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_model_pricing_example.py <model_key>")
        print("\nAvailable models:")
        adapter = LLMAdapter()
        for key in sorted(adapter.model_registry.keys()):
            print(f"  {key}")
        return

    model_key = sys.argv[1]
    adapter = LLMAdapter()

    pricing = adapter.get_pricing_for_model(model_key)

    if pricing:
        print(f"Pricing for {model_key}:")
        if isinstance(pricing, dict):
            print(f"  Input: ${pricing.get('input_per_mm', 'N/A')}/1M tokens")
            print(f"  Output: ${pricing.get('output_per_mm', 'N/A')}/1M tokens")
            if pricing.get('cached_input_per_mm', 0) > 0:
                print(f"  Cached Input: ${pricing.get('cached_input_per_mm')}/1M tokens")
        else:
            print(f"  Input: ${pricing.input_per_mm}/1M tokens")
            print(f"  Output: ${pricing.output_per_mm}/1M tokens")
            if pricing.cached_input_per_mm > 0:
                print(f"  Cached Input: ${pricing.cached_input_per_mm}/1M tokens")
    else:
        print(f"No pricing found for {model_key}")
        print("\nAvailable models:")
        for key in sorted(adapter.model_registry.keys()):
            print(f"  {key}")

if __name__ == "__main__":
    main()
