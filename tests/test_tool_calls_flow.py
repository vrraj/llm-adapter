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
    print(f"\n==== {title} ====")
    try:
        if hasattr(obj, "__dict__"):
            pprint.pprint(getattr(obj, "__dict__"))
        else:
            pprint.pprint(obj)
    except Exception:
        print(obj)


def test_extract_chatcompletion_tool_calls_from_object_shape():
    adapter = LLMAdapter()

    # Simulate OpenAI/Gemini OpenAI-compatible chat.completions shape:
    # resp.choices[0].message.tool_calls = [{type:'function', id:'...', function:{name:'x', arguments:'{...}'}}]
    resp = _Obj(
        choices=[
            _Obj(
                message=_Obj(
                    tool_calls=[
                        _Obj(
                            type="function",
                            id="call_1",
                            function=_Obj(name="get_weather", arguments='{"city":"Paris"}'),
                        )
                    ]
                )
            )
        ]
    )
    _dbg("chatcompletion resp", resp)

    tool_calls = adapter._extract_chatcompletion_tool_calls(resp)
    _dbg("extracted tool_calls", tool_calls)

    assert tool_calls == [
        {"name": "get_weather", "args": '{"city":"Paris"}', "id": "call_1"}
    ]


def test_extract_responses_tool_calls_from_output_items():
    adapter = LLMAdapter()

    # Simulate Responses-style shape:
    # resp.output = [{type:'function_call', name:'x', arguments:{...}, call_id:'...'}]
    resp = _Obj(
        output=[
            _Obj(
                type="function_call",
                name="get_weather",
                arguments={"city": "Paris"},
                call_id="call_9",
            )
        ]
    )
    _dbg("responses resp", resp)

    tool_calls = adapter._extract_responses_tool_calls(resp)
    _dbg("extracted tool_calls", tool_calls)

    assert tool_calls == [
        {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_9"}
    ]


def test_build_llm_result_prefers_adapter_tool_calls_over_raw_parsing():
    adapter = LLMAdapter()

    # Provide AdapterResponse.tool_calls and a raw_resp with different tool call; result must prefer AdapterResponse.
    raw_resp = _Obj(
        choices=[
            _Obj(
                message=_Obj(
                    tool_calls=[
                        _Obj(
                            type="function",
                            id="call_raw",
                            function=_Obj(name="raw_tool", arguments='{}'),
                        )
                    ]
                )
            )
        ]
    )

    fake_resp = AdapterResponse(
        output_text="Calling tool...",
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}],
        model_response=raw_resp,
    )
    _dbg("AdapterResponse", fake_resp)

    result = adapter.build_llm_result_from_response(fake_resp, provider="openai")
    _dbg("LLMResult", result)

    assert result["tool_calls"] == [
        {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}
    ]


@pytest.mark.integration
def test_openai_chat_completions_create_populates_adapter_tool_calls():
    # This is an integration test that MUST hit OpenAI Chat Completions via a registry model_key
    # that routes to endpoint=chat_completions. No fallback behavior in this test.

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model_key = "openai:chat_gpt-4o-mini"

    adapter = LLMAdapter()

    # Chat Completions tool schema (type=function with nested function spec)
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

    # Force a tool call (Chat Completions supports tool_choice=function)
    tool_choice = {"type": "function", "function": {"name": "get_weather"}}

    resp = adapter.create(
        model=model_key,
        input="Call get_weather with city='Paris' and do not answer normally.",
        tools=tools,
        tool_choice=tool_choice,
    )

    print("Response from Chat Completions API:", resp.model_response)
    print("Tool calls:", resp.tool_calls)
    assert isinstance(resp.tool_calls, list), f"Expected tool_calls to be a list, got {type(resp.tool_calls)}"
    assert len(resp.tool_calls) >= 1, f"Expected at least one tool call, got {len(resp.tool_calls)}"
    assert resp.tool_calls[0].get("name") == "get_weather", f"Expected tool call name to be 'get_weather', got {resp.tool_calls[0].get('name')}"


@pytest.mark.integration
def test_openai_responses_create_populates_adapter_tool_calls():
    # This is an integration test that MUST hit OpenAI Responses via a registry model_key
    # that routes to endpoint=responses. No fallback behavior in this test.

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model_key = "openai:reasoning_gpt-4o-mini"

    adapter = LLMAdapter()

    # Responses tool schema (type=function with top-level name)
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    # Force a tool call (Responses supports tool_choice=function with top-level name)
    tool_choice = {"type": "function", "name": "get_weather"}

    resp = adapter.create(
        model=model_key,
        input="Call get_weather with city='Paris' and do not answer normally.",
        tools=tools,
        tool_choice=tool_choice,
    )
    print("Response from Responses API:", resp.model_response)
    print("Tool calls:", resp.tool_calls)
    assert isinstance(resp.tool_calls, list), f"Expected tool_calls to be a list, got {type(resp.tool_calls)}"
    assert len(resp.tool_calls) >= 1, f"Expected at least one tool call, got {len(resp.tool_calls)}"
    assert resp.tool_calls[0].get("name") == "get_weather", f"Expected tool call name to be 'get_weather', got {resp.tool_calls[0].get('name')}"
