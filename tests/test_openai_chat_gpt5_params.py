"""
Tests for the request parameters OpenAIChat sends to reasoning models.

These tests exercise OpenAIChat without triggering LEANN's compiled backend
imports, and without any network access.
"""

import os
import sys
from types import SimpleNamespace
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

pytest.importorskip("openai")


def _sent_params(mock_openai_cls, model, **kwargs):
    from leann.chat import OpenAIChat

    create = mock_openai_cls.return_value.chat.completions.create
    create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=SimpleNamespace(total_tokens=2, prompt_tokens=1, completion_tokens=1),
    )

    assert OpenAIChat(model=model, api_key="test-key").ask("hi", **kwargs) == "ok"
    return create.call_args.kwargs


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5.4", "openai/gpt-5.5"])
@patch("openai.OpenAI")
def test_gpt5_uses_max_completion_tokens_and_default_temperature(mock_openai_cls, model):
    params = _sent_params(mock_openai_cls, model, max_tokens=256)

    assert "max_tokens" not in params
    assert params["max_completion_tokens"] == 256
    assert params["temperature"] == 1.0


@patch("openai.OpenAI")
def test_non_reasoning_model_keeps_max_tokens(mock_openai_cls):
    params = _sent_params(mock_openai_cls, "gpt-4o", max_tokens=256)

    assert params["max_tokens"] == 256
    assert "max_completion_tokens" not in params
    assert params["temperature"] == 0.7
