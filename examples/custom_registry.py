
# user_custom_registry.py

from llm_adapter.model_registry import ModelInfo, Pricing, validate_registry

REGISTRY = {
    "openai:custom_reasoning_o3-mini": ModelInfo(
        key="openai:custom_reasoning_o3-mini",
        provider="openai",
        model="o3-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=1.10, output_per_mm=4.40),
        limits={"max_output_tokens": 2000},
        capabilities={
            "assistant_role": "assistant",
            "reasoning_effort": True,
        },
        param_policy={"disabled": {"stream", "temperature", "top_p"}},
        reasoning_policy={
            "mode": "openai_effort",
            "default": "low",
        },
        reasoning_parameter=("reasoning_effort", "low"),
    ),
    "openai:custom_reasoning_gpt-5-mini": ModelInfo(
        key="openai:custom_reasoning_gpt-5-mini",
        provider="openai",
        model="gpt-5-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=0.25, output_per_mm=2.00),
        limits={"max_output_tokens": 2000},
        capabilities={
            "assistant_role": "assistant",
            "reasoning_effort": True,
        },
        param_policy={"disabled": {"stream", "temperature", "top_p"}},
        reasoning_policy={
            "mode": "openai_effort",
            "default": "minimal",
        },
        reasoning_parameter=("reasoning_effort", "minimal"),
    ),
}

# Optional (recommended in dev/CI)
validate_registry(REGISTRY, strict=False)
