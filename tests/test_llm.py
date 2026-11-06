import json

import httpx
import pytest

from app.llm import ANSWER_SCHEMA, LLMError, LLMTimeout, OllamaLLM


def make_llm(handler) -> OllamaLLM:
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return OllamaLLM("http://ollama", "gemma3", timeout_s=30, max_tokens=300, client=client)


def test_sends_model_messages_schema_and_limits():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '{"ok": 1}'}})

    assert make_llm(handler).complete("SYS", "USER") == '{"ok": 1}'
    assert sent["model"] == "gemma3" and sent["stream"] is False
    assert sent["messages"] == [{"role": "system", "content": "SYS"},
                                {"role": "user", "content": "USER"}]
    assert sent["format"] == ANSWER_SCHEMA
    assert sent["options"] == {"temperature": 0, "num_predict": 300}


def _timeout(request):
    raise httpx.ReadTimeout("too slow", request=request)


def _refused(request):
    raise httpx.ConnectError("connection refused", request=request)


def test_timeout_becomes_llm_timeout():
    with pytest.raises(LLMTimeout):
        make_llm(_timeout).complete("s", "u")


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(404, json={"error": "model 'gemma3' not found"}),
    lambda r: httpx.Response(200, json={"unexpected": 1}),
    lambda r: httpx.Response(200, text="not json"),
    _refused,
], ids=["model-missing", "bad-shape", "not-json", "connection-refused"])
def test_failures_become_llm_error(handler):
    with pytest.raises(LLMError) as err:
        make_llm(handler).complete("s", "u")
    assert not isinstance(err.value, LLMTimeout)
