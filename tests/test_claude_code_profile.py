"""Regression coverage for the Claude Code agent discovery surface (#1531)."""

from types import SimpleNamespace
from unittest.mock import patch

from rapid_mlx.agents import get_profile, list_profiles, load_profiles
from rapid_mlx.agents.adapter import setup_agent_config


def setup_function():
    # Keep these tests independent of registry state left by profile-loader tests.
    load_profiles()


def test_claude_code_is_listed_once_and_claude_is_an_alias():
    profile = get_profile("claude-code")

    assert profile is not None
    assert get_profile("claude") is profile
    from rapid_mlx.agents import get_profile_or_generic

    assert get_profile_or_generic("claude") is profile
    assert [p.name for p in list_profiles()].count("claude-code") == 1


def test_claude_code_setup_uses_bare_anthropic_base_url():
    profile = get_profile("claude-code")
    assert profile is not None

    summary = setup_agent_config(
        profile,
        base_url="http://localhost:8000/v1",
        model_id="qwen3.5-9b-4bit",
    )

    assert "export ANTHROPIC_BASE_URL=http://localhost:8000" in summary
    assert "export ANTHROPIC_API_KEY=not-needed" in summary
    assert "export ANTHROPIC_MODEL=qwen3.5-9b-4bit" in summary
    assert "ANTHROPIC_BASE_URL=http://localhost:8000/v1" not in summary


def test_claude_code_profile_has_runnable_test_command():
    profile = get_profile("claude-code")
    assert profile is not None
    assert profile.testing.binary == "claude"
    assert profile.testing.query_cmd == (
        "claude --allowedTools 'Read(pyproject.toml)' "
        "'Bash(echo rapidmlx_claude-code_test)' -p '{query}'"
    )


def test_claude_code_e2e_receives_rendered_environment(tmp_path):
    profile = get_profile("claude-code")
    assert profile is not None
    from rapid_mlx.agents.testing import AgentTestRunner, TestResult, TestStatus

    isolated_home = tmp_path / "isolated-claude-home"
    isolated_home.mkdir()
    temporary_home = SimpleNamespace(
        name=str(isolated_home),
        cleanup=lambda: None,
    )

    with (
        patch(
            "rapid_mlx.agents.testing.tempfile.TemporaryDirectory",
            return_value=temporary_home,
        ),
        patch(
            "rapid_mlx.agents.testing.AgentTestRunner._server_available",
            return_value=True,
        ),
        patch(
            "rapid_mlx.agents.testing.AgentTestRunner._agent_binary_available",
            return_value=True,
        ),
        patch("rapid_mlx.agents.testing._test_plain_chat") as plain_chat,
        patch(
            "rapid_mlx.agents.testing._test_single_tool_call",
            return_value=TestResult("_test_single_tool_call", TestStatus.PASS),
        ),
        patch(
            "rapid_mlx.agents.testing._test_tool_choice",
            return_value=TestResult("_test_tool_choice", TestStatus.PASS),
        ),
        patch(
            "rapid_mlx.agents.testing._test_multi_turn_tool",
            return_value=TestResult("_test_multi_turn_tool", TestStatus.PASS),
        ),
        patch(
            "rapid_mlx.agents.testing._test_no_tool_leak",
            return_value=TestResult("_test_no_tool_leak", TestStatus.PASS),
        ),
        patch(
            "rapid_mlx.agents.testing._test_no_tool_needed",
            return_value=TestResult("_test_no_tool_needed", TestStatus.PASS),
        ),
        patch(
            "rapid_mlx.agents.testing._test_streaming_tool_call",
            return_value=TestResult("_test_streaming_tool_call", TestStatus.PASS),
        ),
        patch(
            "rapid_mlx.agents.testing._test_many_tools",
            return_value=TestResult("_test_many_tools", TestStatus.PASS),
        ),
        patch(
            "rapid_mlx.agents.testing._test_streaming_basic",
            return_value=TestResult("_test_streaming_basic", TestStatus.PASS),
        ),
        patch(
            "rapid_mlx.agents.testing._test_stress_no_leak",
            return_value=TestResult("_test_stress_no_leak", TestStatus.PASS),
        ),
        patch("rapid_mlx.agents.testing._test_e2e_chat") as e2e_chat,
        patch(
            "rapid_mlx.agents.testing._test_e2e_file_read",
            return_value=TestResult("_test_e2e_file_read", TestStatus.PASS),
        ),
        patch(
            "rapid_mlx.agents.testing._test_e2e_terminal",
            return_value=TestResult("_test_e2e_terminal", TestStatus.PASS),
        ),
    ):
        plain_chat.return_value = TestResult("plain_chat", TestStatus.PASS)
        e2e_chat.return_value = TestResult("e2e_chat", TestStatus.PASS)
        AgentTestRunner(
            profile,
            base_url="http://localhost:8000/v1",
            model_id="qwen3.5-9b-4bit",
        ).run()

    env = e2e_chat.call_args.kwargs["env_overrides"]
    assert {
        key: env[key]
        for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL")
    } == {
        "ANTHROPIC_BASE_URL": "http://localhost:8000",
        "ANTHROPIC_API_KEY": "not-needed",
        "ANTHROPIC_MODEL": "qwen3.5-9b-4bit",
    }
    assert env["HOME"] == str(isolated_home)
    assert env["CLAUDE_CONFIG_DIR"] == f"{env['HOME']}/.claude"
