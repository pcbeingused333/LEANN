"""
Tests for how OllamaChat sends the thinking budget to Ollama.

These tests run without LEANN's compiled backends and without an Ollama server.
"""

import json
import os
import sys
from unittest.mock import patch

import pytest

_LEANN_SRC = os.path.join(os.path.dirname(__file__), "..", "packages", "leann-core", "src")
if _LEANN_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_LEANN_SRC))

if "leann" not in sys.modules:
    import types

    _stub = types.ModuleType("leann")
    _stub.__path__ = [os.path.join(os.path.abspath(_LEANN_SRC), "leann")]
    sys.modules["leann"] = _stub

pytest.importorskip("requests")


def _sent_payload(model, **kwargs):
    from leann.chat import OllamaChat

    with (
        patch("requests.get"),
        patch("leann.chat.validate_model_and_suggest", return_value=None),
        patch("requests.post") as mock_post,
    ):
        mock_post.return_value.text = json.dumps({"response": "ok", "done": True})
        llm = OllamaChat(model=model, host="http://ollama.invalid")
        assert llm.ask("hi", **kwargs) == "ok"

    return json.loads(mock_post.call_args.kwargs["data"])


@pytest.mark.parametrize("budget", ["low", "medium", "high"])
def test_thinking_budget_is_sent_as_think(budget):
    payload = _sent_payload("gpt-oss:20b", thinking_budget=budget)

    assert payload["think"] == budget
    assert "reasoning" not in payload["options"]
    assert "thinking_budget" not in payload["options"]


@pytest.mark.parametrize("model", ["llama3:8b", "deepseek-coder:6.7b"])
def test_think_is_not_sent_to_models_without_thinking(model):
    # Ollama rejects the request with HTTP 400 if such a model gets `think`.
    payload = _sent_payload(model, thinking_budget="high")

    assert "think" not in payload
    assert "thinking_budget" not in payload["options"]


def test_think_is_not_sent_without_a_budget():
    assert "think" not in _sent_payload("gpt-oss:20b")
