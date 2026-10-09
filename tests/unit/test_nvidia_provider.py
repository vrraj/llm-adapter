"""Unit tests for the NVIDIA NIM (Nemotron) provider in llm_adapter."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest

from llm_adapter.llm_adapter import LLMAdapter, AdapterResponse, LLMError
from llm_adapter.model_registry import REGISTRY, validate_registry, get_model_info


class _Obj:
    """Tiny helper to create attribute-style objects."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate tests from any real NVIDIA/allowlist env configuration."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_ADAPTER_ALLOWED_MODELS", raising=False)


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeNvidiaClient:
    """Mimics the OpenAI SDK client pointed at a NVIDIA NIM endpoint."""

    def __init__(self, response):
        self.chat = _FakeChat(_FakeCompletions(response))
        self.completions = self.chat.completions


def _chat_response(content="Hello!", reasoning_content=None, tool_calls=None, finish_reason="stop"):
    message = _Obj(role="assistant", content=content)
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    if tool_calls is not None:
        message.tool_calls = [
            _Obj(type="function", id="call_1", function=_Obj(name=t["name"], arguments=t["arguments"]))
            for t in tool_calls
        ]
    return _Obj(
        id="resp_1",
        created=1234567890,
        model="nvidia/nemotron-3-super-120b-a12b",
        choices=[_Obj(index=0, message=message, finish_reason=finish_reason)],
        usage=_Obj(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


# ----------------------------
# Registry
# ----------------------------

def test_registry_contains_nvidia_models():
    nvidia_keys = sorted(k for k, v in REGISTRY.items() if v.provider == "nvidia")
    assert nvidia_keys == [
        "nvidia:nemotron-3-super-120b",
        "nvidia:nemotron-3.5-lightning-30b",
        "nvidia:nemotron-nano-30b",
        "nvidia:nemotron-super-49b",
    ]


def test_registry_nvidia_model_info_fields():
    mi = get_model_info("nvidia:nemotron-3-super-120b")
    assert mi.provider == "nvidia"
    assert mi.model == "nvidia/nemotron-3-super-120b-a12b"
    assert mi.endpoint == "chat_completions"
    assert mi.capabilities["assistant_role"] == "assistant"
    assert mi.pricing is not None
    assert mi.pricing.input_per_mm == 0.10
    assert mi.pricing.output_per_mm == 0.50


def test_registry_lightning_model_has_nvidia_budget_reasoning_policy():
    mi = get_model_info("nvidia:nemotron-3.5-lightning-30b")
    assert mi.reasoning_policy["mode"] == "nvidia_budget"
    assert mi.reasoning_policy["param"] == "reasoning_budget"
    assert mi.reasoning_policy["default"] == "low"
    assert mi.reasoning_policy["budget_map"]["low"] == 2048


def test_validate_registry_accepts_nvidia_entries():
    validate_registry(REGISTRY)


# ----------------------------
# Client factory
# ----------------------------

def test_get_nvidia_missing_key_raises():
    adapter = LLMAdapter()
    with pytest.raises(LLMError) as exc_info:
        adapter._get_nvidia()
    assert exc_info.value.provider == "nvidia"
    assert exc_info.value.code == "missing_api_key"


def test_get_nvidia_default_base_url():
    adapter = LLMAdapter(nvidia_api_key="nvapi-test")
    client = adapter._get_nvidia()
    assert str(client.base_url).startswith("https://integrate.api.nvidia.com/v1")


def test_get_nvidia_custom_base_url():
    adapter = LLMAdapter(nvidia_api_key="nvapi-test", nvidia_base_url="http://localhost:8000/v1")
    client = adapter._get_nvidia()
    assert str(client.base_url).startswith("http://localhost:8000/v1")


def test_get_nvidia_env_var_base_url(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("NVIDIA_BASE_URL", "http://nim.internal:8000/v1")
    adapter = LLMAdapter()
    client = adapter._get_nvidia()
    assert str(client.base_url).startswith("http://nim.internal:8000/v1")


def test_get_nvidia_injected_client_wins():
    sentinel = object()
    adapter = LLMAdapter(nvidia_client=sentinel)
    assert adapter._get_nvidia() is sentinel


# ----------------------------
# Reasoning policy
# ----------------------------

def test_nvidia_reasoning_policy_maps_effort_to_budget():
    adapter = LLMAdapter()
    out = adapter._apply_nvidia_reasoning_policy(
        "nvidia:nemotron-3.5-lightning-30b",
        {"reasoning_effort": "medium"},
    )
    assert out["extra_body"]["reasoning_budget"] == 4096
    assert out["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_nvidia_reasoning_policy_none_disables_thinking():
    adapter = LLMAdapter()
    out = adapter._apply_nvidia_reasoning_policy(
        "nvidia:nemotron-3.5-lightning-30b",
        {"reasoning_effort": "none"},
    )
    assert out["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "reasoning_budget" not in out["extra_body"]


def test_nvidia_reasoning_policy_defaults_to_low():
    adapter = LLMAdapter()
    out = adapter._apply_nvidia_reasoning_policy("nvidia:nemotron-3.5-lightning-30b", {})
    assert out["extra_body"]["reasoning_budget"] == 2048
    assert out["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_nvidia_reasoning_policy_clamps_budget_to_output_cap():
    adapter = LLMAdapter()
    out = adapter._apply_nvidia_reasoning_policy(
        "nvidia:nemotron-3.5-lightning-30b",
        {"reasoning_effort": "high", "max_output_tokens": 1000},
    )
    # Keeps at least 100 tokens for the visible answer.
    assert out["extra_body"]["reasoning_budget"] == 900


def test_nvidia_reasoning_policy_preserves_caller_extra_body():
    adapter = LLMAdapter()
    out = adapter._apply_nvidia_reasoning_policy(
        "nvidia:nemotron-3.5-lightning-30b",
        {"reasoning_effort": "low", "extra_body": {"custom_flag": True}},
    )
    assert out["extra_body"]["custom_flag"] is True
    assert out["extra_body"]["reasoning_budget"] == 2048


def test_nvidia_reasoning_policy_ignores_non_nvidia_models():
    adapter = LLMAdapter()
    kwargs = {"reasoning_effort": "high"}
    out = adapter._apply_nvidia_reasoning_policy("openai:reasoning_o3-mini", dict(kwargs))
    assert out == kwargs


def test_nvidia_reasoning_policy_ignores_non_reasoning_nvidia_models():
    adapter = LLMAdapter()
    kwargs = {"temperature": 0.5}
    out = adapter._apply_nvidia_reasoning_policy("nvidia:nemotron-3-super-120b", dict(kwargs))
    assert out == kwargs


# ----------------------------
# create() dispatch + call path
# ----------------------------

def _make_adapter_with_fake_client(response):
    fake = _FakeNvidiaClient(response)
    adapter = LLMAdapter(nvidia_client=fake)
    return adapter, fake


def test_create_dispatches_nvidia_by_registry_inference():
    adapter, fake = _make_adapter_with_fake_client(_chat_response(content="Hi from Nemotron"))
    resp = adapter.create(
        model="nvidia:nemotron-3-super-120b",
        input="Say hi",
        max_output_tokens=256,
        temperature=0.2,
    )
    assert isinstance(resp, AdapterResponse)
    assert resp.output_text == "Hi from Nemotron"
    sent = fake.chat.completions.last_kwargs
    assert sent["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert sent["messages"] == [{"role": "user", "content": "Say hi"}]
    # Canonical max_output_tokens maps to NIM's OpenAI `max_tokens`.
    assert sent["max_tokens"] == 256
    assert sent["temperature"] == 0.2


def test_create_dispatches_nvidia_with_explicit_provider():
    adapter, fake = _make_adapter_with_fake_client(_chat_response())
    resp = adapter.create(provider="nvidia", model="nvidia:nemotron-nano-30b", input="Hello")
    assert isinstance(resp, AdapterResponse)
    assert fake.chat.completions.last_kwargs["model"] == "nvidia/nemotron-3-nano-30b-a3b"


def test_create_nvidia_wraps_usage_and_metadata():
    adapter, _ = _make_adapter_with_fake_client(_chat_response())
    resp = adapter.create(model="nvidia:nemotron-3-super-120b", input="Hello")
    assert resp.usage["prompt_tokens"] == 10
    assert resp.usage["output_tokens"] == 5
    assert resp.metadata["provider"] == "nvidia"
    assert resp.status == "completed"
    assert resp.finish_reason == "stop"


def test_create_nvidia_tool_calls_normalized():
    adapter, _ = _make_adapter_with_fake_client(
        _chat_response(content="", tool_calls=[{"name": "get_weather", "arguments": '{"city":"Paris"}'}])
    )
    resp = adapter.create(model="nvidia:nemotron-3-super-120b", input="What's the weather in Paris?")
    assert resp.tool_calls == [{"name": "get_weather", "args": '{"city":"Paris"}', "id": "call_1"}]


def test_create_nvidia_reasoning_content_split_on_normalize():
    adapter, _ = _make_adapter_with_fake_client(
        _chat_response(content="9.8 is larger.", reasoning_content="Let me compare them...")
    )
    resp = adapter.create(model="nvidia:nemotron-3.5-lightning-30b", input="Which is larger: 9.11 or 9.8?")
    result = adapter.normalize_adapter_response(resp, provider="nvidia")
    assert result["reasoning"] == "Let me compare them..."
    assert result["text"] == "9.8 is larger."


def test_create_nvidia_reasoning_effort_becomes_extra_body():
    adapter, fake = _make_adapter_with_fake_client(_chat_response())
    adapter.create(
        model="nvidia:nemotron-3.5-lightning-30b",
        input="Think step by step",
        reasoning_effort="high",
    )
    sent = fake.chat.completions.last_kwargs
    assert sent["extra_body"]["reasoning_budget"] == 8192
    assert sent["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_create_nvidia_sanitizes_flat_tools():
    adapter, fake = _make_adapter_with_fake_client(_chat_response())
    flat_tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    adapter.create(model="nvidia:nemotron-3-super-120b", input="Weather?", tools=flat_tools)
    sent_tools = fake.chat.completions.last_kwargs["tools"]
    assert sent_tools[0]["function"]["name"] == "get_weather"


def test_create_unknown_provider_raises():
    adapter = LLMAdapter()
    with pytest.raises(LLMError) as exc_info:
        adapter.create(provider="anthropic", model="claude-x", input="Hello")
    assert exc_info.value.code == "unsupported_provider"


def test_prepare_nvidia_kwargs_drops_model_spec_marker():
    adapter = LLMAdapter()
    out = adapter._prepare_nvidia_adapter_kwargs(
        "nvidia:nemotron-3-super-120b",
        {"__model_spec": object(), "temperature": 0.1},
    )
    assert "__model_spec" not in out
    assert out["temperature"] == 0.1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
