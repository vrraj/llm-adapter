
import sys
from llm_adapter import LLMAdapter

def main():
    if len(sys.argv) < 2:
        print("Usage: python pricing_cli.py <model_name>")
        return

    model = sys.argv[1]
    adapter = LLMAdapter()

    pricing = adapter.get_pricing_for_model(model)

    if pricing:
        print(f"Pricing for {model}:")
        print(pricing)
    else:
        print(f"No pricing found for {model}")

if __name__ == "__main__":
    main()
