import asyncio
import contextlib
import copy
import json

import httpx
import pytest

from src import agent_loop, llm_core
from src.agent_tools import admin_tools
from src import settings
from routes.chat_routes import _resolve_prompt_thinking_mode


def test_kimi_thinking_off_translates_to_provider_payload():
    assert llm_core._detect_provider("https://api.kimi.com/coding/v1") == "kimi-code"
    payload = {}
    llm_core._apply_hosted_thinking_mode(
        payload, "kimi-code", "kimi-for-coding", "off"
    )
    assert payload["thinking"] == {"type": "disabled"}


def test_kimi_thinking_default_does_not_add_provider_field():
    payload = {}
    llm_core._apply_hosted_thinking_mode(
        payload, "kimi-code", "kimi-for-coding", None
    )
    assert "thinking" not in payload


def test_openrouter_thinking_off_uses_unified_reasoning_control():
    payload = {}
    llm_core._apply_hosted_thinking_mode(
        payload, "openrouter", "moonshotai/kimi-k3", "off"
    )
    assert payload["reasoning"] == {"effort": "none"}


_OPENROUTER_URL = "https://openrouter.ai/api/v1"
_MANDATORY_REASONING_BODY = json.dumps({"error": {
    "message": "Reasoning is mandatory for this endpoint and cannot be disabled.", "code": 400,
}})
_TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]


@pytest.fixture(autouse=True)
def _isolated_learned_reasoning_models(monkeypatch):
    monkeypatch.setattr(llm_core, "_OPENROUTER_LEARNED_REASONING_REQUIRED", set())


class _RecordingStreamClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.payloads = []

    @contextlib.asynccontextmanager
    async def stream(self, method, url, **kwargs):
        # Copy: production mutates the payload before retrying.
        self.payloads.append(copy.deepcopy(kwargs["json"]))
        yield self.responses.pop(0)


def _ok_stream(text):
    chunk = json.dumps({"choices": [{"delta": {"content": text}}]})
    return httpx.Response(200, text=f"data: {chunk}\n\ndata: [DONE]\n\n")


def _run_openrouter_stream(monkeypatch, client, model, thinking_mode, tools=None):
    """Stream one request and return ``(text, error_payloads)``."""
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda _url: False)

    async def collect():
        return [chunk async for chunk in llm_core.stream_llm(
            _OPENROUTER_URL, model, [{"role": "user", "content": "ping"}],
            tools=tools, thinking_mode=thinking_mode,
        )]

    text, errors = "", []
    for chunk in asyncio.run(collect()):
        if "data: {" not in chunk:
            continue
        data = json.loads(chunk.split("data: ", 1)[1])
        if chunk.startswith("event: error"):
            errors.append(data)
        elif isinstance(data.get("delta"), str):
            text += data["delta"]
    return text, errors


@pytest.mark.parametrize("tools", [None, _TOOLS], ids=["chat", "agent"])
def test_openrouter_mandatory_reasoning_stream_self_heals(monkeypatch, tools):
    client = _RecordingStreamClient(
        httpx.Response(400, text=_MANDATORY_REASONING_BODY), _ok_stream("pong"),
    )

    result = _run_openrouter_stream(monkeypatch, client, "acme/always-thinks:free", "off", tools)

    assert result == ("pong", [])
    first, retry = client.payloads
    assert first["reasoning"] == {"effort": "none"}
    assert "reasoning" not in retry
    if tools:
        assert first["tools"] == retry["tools"] == tools

    # The model is remembered (across variants): later requests skip the control.
    client = _RecordingStreamClient(_ok_stream("again"))
    result = _run_openrouter_stream(monkeypatch, client, "acme/always-thinks", "off", tools)

    assert result == ("again", [])
    assert len(client.payloads) == 1
    assert "reasoning" not in client.payloads[0]

    other = {}
    llm_core._apply_hosted_thinking_mode(other, "openrouter", "moonshotai/kimi-k3", "off")
    assert other["reasoning"] == {"effort": "none"}


@pytest.mark.parametrize(("thinking_mode", "tools", "bodies"), [
    ("off", None, [_MANDATORY_REASONING_BODY, _MANDATORY_REASONING_BODY]),
    ("off", None, [json.dumps({"error": {"message": "Invalid request", "code": 400}})]),
    (None, _TOOLS, [_MANDATORY_REASONING_BODY]),
], ids=["one-shot", "unrelated-400", "agent-default"])
def test_openrouter_400_surfaced_without_self_heal(monkeypatch, thinking_mode, tools, bodies):
    model = "acme/rejects"
    client = _RecordingStreamClient(*(httpx.Response(400, text=body) for body in bodies))

    _, errors = _run_openrouter_stream(monkeypatch, client, model, thinking_mode, tools)

    assert len(client.payloads) == len(bodies)
    assert [error["status"] for error in errors] == [400]
    if thinking_mode is None:
        assert "reasoning" not in client.payloads[0]
    if len(bodies) == 1:
        # Nothing was retried, so thinking-off must still be sent next time.
        payload = {}
        llm_core._apply_hosted_thinking_mode(payload, "openrouter", model, "off")
        assert payload["reasoning"] == {"effort": "none"}


def test_openrouter_mandatory_reasoning_async_call_self_heals(monkeypatch):
    sent = []
    responses = [
        httpx.Response(400, text=_MANDATORY_REASONING_BODY),
        httpx.Response(200, json={"choices": [{"message": {"content": "pong"}}]}),
    ]

    async def fake_post(client, url, headers, **kwargs):
        sent.append(copy.deepcopy(kwargs["json"]))
        return responses.pop(0)

    monkeypatch.setattr(llm_core, "_get_http_client", lambda: None)
    monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", fake_post)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda _url: False)
    monkeypatch.setattr(llm_core, "_response_cache", {})
    monkeypatch.setattr(llm_core, "_response_model_cache", {})

    # max_retries=1: the self-heal must not consume the transport retry budget.
    result = asyncio.run(llm_core.llm_call_async(
        _OPENROUTER_URL, "acme/always-thinks", [{"role": "user", "content": "ping"}],
        max_tokens=0, max_retries=1, thinking_mode="off",
    ))

    assert result == "pong"
    assert [payload.get("reasoning") for payload in sent] == [{"effort": "none"}, None]


def test_stream_transport_only_keeps_required_protocol_reasoning_when_off():
    import ast
    source = open(llm_core.__file__, encoding="utf-8").read()
    condition = next(node.test for node in ast.walk(ast.parse(source))
                     if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
                     and isinstance(node.test.values[0], ast.Name)
                     and node.test.values[0].id == "reasoning")
    expression = compile(ast.Expression(condition), "reasoning-policy", "eval")
    for model, mode, expected in [("generic", "off", False), ("generic", "on", True),
                                  ("deepseek-v4", "off", True)]:
        assert bool(eval(expression, {"reasoning": "analysis", "thinking_mode": mode,
                                      "model": model, "_normalize_thinking_mode": llm_core._normalize_thinking_mode})) is expected


def test_response_cache_is_partitioned_by_thinking_mode():
    args = ("https://api.kimi.com/coding/v1", "kimi-for-coding", [{"role": "user", "content": "Hi"}], 1.0, 100)
    assert llm_core._get_cache_key(*args, thinking_mode="off") != llm_core._get_cache_key(*args, thinking_mode="on")


def test_chat_backend_reads_thinking_mode_from_active_prompt_preset():
    manager = type("PresetManager", (), {"presets": {"custom": {"enabled": True, "thinking_mode": "off"}}})()
    assert _resolve_prompt_thinking_mode(None, "custom", manager) == "off"
    assert _resolve_prompt_thinking_mode("on", "custom", manager) == "on"


def test_email_writing_style_is_agent_manageable_and_routable():
    assert settings.DEFAULT_SETTINGS["email_writing_style"] == ""
    assert settings.DEFAULT_SETTINGS["document_writing_style"] == ""
    source = open(admin_tools.__file__, encoding="utf-8").read()
    assert '"writing style": "document_writing_style"' in source
    assert '"email writing style": "email_writing_style"' in source
    assert any(
        "writing style" in triggers and "manage_settings" in tools
        for triggers, tools in agent_loop._QWEN38_ROUTER_KEYWORD_TOOLS
    )
    index_source = open("src/tool_index.py", encoding="utf-8").read()
    assert '"writing style", "reply style", "email writing style"' in index_source
