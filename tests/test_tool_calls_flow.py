import pytest
from llm_adapter.llm_adapter import LLMAdapter, AdapterResponse


def test_tool_calls_preferred_over_raw():
    adapter = LLMAdapter()

    fake_resp = AdapterResponse(
        output_text="Calling tool...",
        model="test-model",
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
        tool_calls=[
            {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}
        ],
    )

    result = adapter.build_llm_result_from_response(
        fake_resp,
        provider="openai",
    )

    assert result["tool_calls"] is not None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "get_weather"
    assert result["tool_calls"][0]["args"]["city"] == "Paris"
