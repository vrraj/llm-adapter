import pytest
import os
import pprint

from llm_adapter.llm_adapter import LLMAdapter, AdapterResponse


class _Obj:
    """Tiny helper to create attribute-style objects."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _dbg(title: str, obj: object) -> None:
    print(f"\n==== {title} ==== ")
    try:
        if hasattr(obj, "__dict__"):
            pprint.pprint(getattr(obj, "__dict__"))
        else:
            pprint.pprint(obj)
    except Exception:
        print(obj)


@pytest.mark.integration
def test_nvidia_chat_completions_create_populates_adapter_tool_calls():
    if not os.getenv("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY not set")

    model_key = "nvidia:nemotron-3-super-120b"

    adapter = LLMAdapter()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    resp = adapter.create(
        model=model_key,
        input="Call get_weather with city='Paris' and do not answer normally.",
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        temperature=0,
    )

    print("Response from NVIDIA NIM (chat.completions):", resp.model_response)
    print("Tool calls:", resp.tool_calls)
    assert isinstance(resp.tool_calls, list), f"Expected tool_calls to be a list, got {type(resp.tool_calls)}"
    assert len(resp.tool_calls) >= 1, f"Expected at least one tool call, got {len(resp.tool_calls)}"
    assert resp.tool_calls[0].get("name") == "get_weather", f"Expected tool call name to be 'get_weather', got {resp.tool_calls[0].get('name')}"


@pytest.mark.integration
def test_nvidia_reasoning_model_returns_reasoning_and_answer():
    if not os.getenv("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY not set")

    model_key = "nvidia:nemotron-3.5-lightning-30b"

    adapter = LLMAdapter()

    resp = adapter.create(
        model=model_key,
        input="Which number is larger: 9.11 or 9.8? Explain briefly.",
        reasoning_effort="low",
        max_output_tokens=1000,
    )

    assert isinstance(resp, AdapterResponse)
    result = adapter.normalize_adapter_response(resp, provider="nvidia")
    _dbg("normalized LLMResult", result)

    assert result["status"] == "completed"
    assert result["text"], "Expected a non-empty answer"
    usage = result.get("usage") or {}
    assert usage.get("total_tokens", 0) > 0
