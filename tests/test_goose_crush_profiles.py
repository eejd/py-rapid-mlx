"""Current Goose and Crush configurations for a local OpenAI server."""

import json

import pytest

from rapid_mlx.agents import get_profile
from rapid_mlx.agents.adapter import setup_agent_config


@pytest.mark.parametrize("keyed", [False, True])
def test_goose_provider_setup(tmp_path, monkeypatch, keyed):
    monkeypatch.setenv("HOME", str(tmp_path))
    if keyed:
        monkeypatch.setenv("RAPID_MLX_API_KEY", "local-secret")
    else:
        monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    profile = get_profile("goose")
    assert profile is not None
    summary = setup_agent_config(
        profile, "http://127.0.0.1:18740/v1", "qwen3.5-4b-4bit", context_length=32768
    )
    path = tmp_path / ".config/goose/custom_providers/custom_rapid_mlx.json"
    assert str(path) in summary
    data = json.loads(path.read_text())
    assert data["base_url"] == "http://127.0.0.1:18740/v1/chat/completions"
    assert data["models"] == [{"name": "qwen3.5-4b-4bit", "context_limit": 32768}]
    assert data["requires_auth"] is keyed
    assert data["api_key_env"] == ("RAPID_MLX_API_KEY" if keyed else "")
    assert "local-secret" not in path.read_text()


@pytest.mark.parametrize("keyed", [False, True])
def test_crush_provider_setup(tmp_path, monkeypatch, keyed):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CRUSH_GLOBAL_CONFIG", str(tmp_path / "crush-config"))
    if keyed:
        monkeypatch.setenv("RAPID_MLX_API_KEY", "local-secret")
    else:
        monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    profile = get_profile("crush")
    assert profile is not None
    setup_agent_config(
        profile, "http://127.0.0.1:18741/v1", "qwen3.5-4b-4bit", context_length=32768
    )
    path = tmp_path / "crush-config/crush.json"
    data = json.loads(path.read_text())
    provider = data["providers"]["rapid-mlx"]
    assert provider["type"] == "openai-compat"
    assert provider["base_url"] == "http://127.0.0.1:18741/v1"
    assert provider["api_key"] == ("$RAPID_MLX_API_KEY" if keyed else "not-needed")
    assert provider["models"][0]["context_window"] == 32768
    assert data["models"]["large"] == {
        "provider": "rapid-mlx",
        "model": "qwen3.5-4b-4bit",
    }
    assert "local-secret" not in path.read_text()
