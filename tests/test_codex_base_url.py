# SPDX-License-Identifier: Apache-2.0
"""Codex server-root input must reach the same API in setup and tests."""

from argparse import Namespace

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from rapid_mlx.agents import get_profile
from rapid_mlx.agents.adapter import setup_agent_config
from rapid_mlx.agents.testing import AgentTestRunner


@pytest.mark.parametrize(
    "supplied,expected",
    [
        ("http://127.0.0.1:8899", "http://127.0.0.1:8899/v1"),
        ("http://localhost:8899/", "http://localhost:8899/v1"),
        ("https://inference.example", "https://inference.example/v1"),
        ("http://[::1]:8899/", "http://[::1]:8899/v1"),
        ("http://localhost:8899/v1", "http://localhost:8899/v1"),
        ("http://localhost:8899/v1/", "http://localhost:8899/v1/"),
        ("https://inference.example/local/v1", "https://inference.example/local/v1"),
        ("https://inference.example/api", "https://inference.example/api"),
        ("http://localhost:1", "http://localhost:1/v1"),
        ("http://localhost:65535/v1", "http://localhost:65535/v1"),
    ],
)
def test_setup_and_runner_agree_on_api_base(tmp_path, monkeypatch, supplied, expected):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    profile = get_profile("codex")
    setup_agent_config(profile, supplied, "local-qwen")
    config = tomllib.loads((tmp_path / "config.toml").read_text())
    assert config["model_providers"]["rapid-mlx"]["base_url"] == expected
    runner = AgentTestRunner(profile, base_url=supplied, model_id="local-qwen")
    assert runner.base_url == expected
    assert profile.normalize_base_url(expected) == expected


def test_runner_discovers_model_at_normalized_api(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return type(
            "Response", (), {"json": lambda _: {"data": [{"id": "local-qwen"}]}}
        )()

    monkeypatch.setattr("rapid_mlx.agents.testing.httpx.get", get)
    runner = AgentTestRunner(get_profile("codex"), "http://127.0.0.1:8899")
    assert runner.model_id == "local-qwen"
    assert calls == ["http://127.0.0.1:8899/v1/models"]


@pytest.mark.parametrize(
    "suffix", ["?token=secret", "/?token=secret", "#fragment", "?", "#", "/?#"]
)
def test_root_query_or_fragment_rejected_before_setup_or_discovery(
    tmp_path, monkeypatch, suffix
):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    profile = get_profile("codex")
    supplied = "http://localhost:8899" + suffix
    with pytest.raises(ValueError, match="must not contain a query or fragment"):
        setup_agent_config(profile, supplied, "local-qwen")
    with pytest.raises(ValueError, match="must not contain a query or fragment"):
        AgentTestRunner(profile, supplied)
    assert not (tmp_path / "config.toml").exists()


def test_cli_rejects_root_query_without_echoing_secret(tmp_path, monkeypatch, capsys):
    from rapid_mlx.cli import agents_command

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        agents_command(
            Namespace(
                agent_name="codex",
                base_url="http://localhost:8899?token=secret",
                setup=True,
                test=False,
            )
        )
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "must not contain a query or fragment" in output
    assert "secret" not in output
    assert not (tmp_path / "config.toml").exists()


@pytest.mark.parametrize(
    "supplied",
    [
        "http://user:credential-canary-4436＠localhost:8899",
        "http://user:credential-canary-4436@localhost：8899/v1",
        "http://user:credential-canary-4436@[::1",
        "http://user:credential-canary-4436@localhost:abc",
        "http://user:credential-canary-4436@localhost:0",
        "http://user:credential-canary-4436@localhost:65536/v1",
        "http://user:credential-canary-4436@",
        "http:///v1",
    ],
)
def test_malformed_url_rejected_without_echoing_secret(
    tmp_path, monkeypatch, capsys, supplied
):
    import traceback

    from rapid_mlx.cli import agents_command

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    profile = get_profile("codex")
    for operation in (
        lambda: setup_agent_config(profile, supplied, "local-qwen"),
        lambda: AgentTestRunner(profile, supplied),
    ):
        with pytest.raises(ValueError) as exc:
            operation()
        assert "credential-canary-4436" not in "".join(
            traceback.format_exception(exc.value)
        )
        assert "valid HTTP(S)" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        agents_command(
            Namespace(agent_name="codex", base_url=supplied, setup=True, test=False)
        )
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "credential-canary-4436" not in output
    assert "valid HTTP(S)" in output
    assert not (tmp_path / "config.toml").exists()


def test_cli_normalizes_before_discovery_verification_and_write(tmp_path, monkeypatch):
    from rapid_mlx.cli import agents_command

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    calls = []

    def discover(url):
        calls.append(("discover", url))
        return "local-qwen", 32768

    def verify(url, model, **kwargs):
        calls.append(("verify", url))
        assert model == "local-qwen"
        return model

    monkeypatch.setattr("rapid_mlx.agents.adapter._detect_running_model", discover)
    monkeypatch.setattr("rapid_mlx.agents.setup.verify_server", verify)
    agents_command(
        Namespace(
            agent_name="codex",
            base_url="http://127.0.0.1:8899/",
            model=None,
            setup=True,
            test=False,
            agent_version=None,
            dry_run=False,
            no_check=False,
            yes=True,
        )
    )
    assert calls == [
        ("discover", "http://127.0.0.1:8899/v1"),
        ("verify", "http://127.0.0.1:8899/v1"),
    ]
    config = tomllib.loads((tmp_path / "config.toml").read_text())
    assert (
        config["model_providers"]["rapid-mlx"]["base_url"] == "http://127.0.0.1:8899/v1"
    )


@pytest.mark.parametrize("name", ["claude-code", "hermes", "qwen-code"])
def test_other_agent_url_contracts_unchanged(name):
    assert (
        get_profile(name).normalize_base_url("http://localhost:8899/")
        == "http://localhost:8899/"
    )


@pytest.fixture(autouse=True)
def isolated_agent_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))


@pytest.fixture
def cli_runner(monkeypatch):
    from unittest.mock import Mock

    runner = Mock()
    runner.return_value._server_available.return_value = True
    runner.return_value.run.return_value.print_summary.return_value = True
    monkeypatch.setattr("rapid_mlx.agents.testing.AgentTestRunner", runner)
    return runner


def _cli_args(*flags):
    from rapid_mlx.cli import build_parser

    return build_parser().parse_args(["agents", *flags])


def _run_cli_test(cli_runner, *flags, exit_code=0):
    from rapid_mlx.cli import agents_command

    with pytest.raises(SystemExit) as raised:
        agents_command(_cli_args(*flags, "--test"))
    assert raised.value.code == exit_code
    return cli_runner.call_args.kwargs["base_url"]


def test_cli_setup_then_test_uses_saved_endpoint(monkeypatch, cli_runner, capsys):
    from rapid_mlx.cli import agents_command

    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.fetch_context_window", lambda *_: 32768
    )
    agents_command(
        _cli_args(
            "codex",
            "--setup",
            "--base-url",
            "http://127.0.0.1:8581",
            "--model",
            "local-qwen",
            "--no-check",
        )
    )
    capsys.readouterr()
    assert _run_cli_test(cli_runner, "codex") == "http://127.0.0.1:8581/v1"
    assert (
        "Using saved Codex endpoint: http://127.0.0.1:8581/v1"
        in capsys.readouterr().out
    )


def test_cli_explicit_endpoint_wins(cli_runner, tmp_path):
    config = tmp_path / "codex" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        'model_provider = "rapid-mlx"\n'
        '[model_providers.rapid-mlx]\nbase_url = "http://127.0.0.1:8581/v1"\n'
    )
    assert (
        _run_cli_test(cli_runner, "codex", "--base-url", "http://localhost:8000/v1")
        == "http://localhost:8000/v1"
    )


@pytest.mark.parametrize(
    "contents",
    [
        None,
        "invalid toml = [",
        "",
        "model_provider = 123\n",
        'model_provider = "missing"\n',
        'model_provider = "rapid-mlx"\nmodel_providers = 1\n',
        'model_provider = "rapid-mlx"\n[model_providers]\nrapid-mlx = 1\n',
        'model_provider = "rapid-mlx"\n[model_providers.rapid-mlx]\n',
        'model_provider = "rapid-mlx"\n[model_providers.rapid-mlx]\nbase_url = 123\n',
        'model_provider = "rapid-mlx"\n[model_providers.rapid-mlx]\nbase_url = "http://localhost:bad"\n',
    ],
)
def test_cli_unusable_saved_endpoint_falls_back(cli_runner, tmp_path, contents, capsys):
    if contents is not None:
        config = tmp_path / "codex" / "config.toml"
        config.parent.mkdir()
        config.write_text(contents)
    assert _run_cli_test(cli_runner, "codex") == "http://localhost:8000/v1"
    assert "Using saved Codex endpoint" not in capsys.readouterr().out


@pytest.mark.parametrize("available", [True, False])
def test_cli_saved_endpoint_credentials_are_not_printed(
    cli_runner, tmp_path, capsys, available
):
    config = tmp_path / "codex" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        'model_provider = "custom"\n[model_providers.custom]\n'
        'base_url = "http://user:saved-password@127.0.0.1:8581"\n'
    )
    cli_runner.return_value._server_available.return_value = available
    assert _run_cli_test(cli_runner, "codex", exit_code=0 if available else 1) == (
        "http://user:saved-password@127.0.0.1:8581/v1"
    )
    output = capsys.readouterr().out
    assert "Using saved Codex endpoint: http://127.0.0.1:8581/v1" in output
    assert "user" not in output
    assert "saved-password" not in output
    if not available:
        assert "Server not running at http://127.0.0.1:8581/v1" in output


def test_cli_other_agent_keeps_default_without_readback(cli_runner, monkeypatch):
    def unexpected_read(*_):
        pytest.fail("Other agents must not read Codex config")

    monkeypatch.setattr(
        "rapid_mlx.agents.adapter._resolve_config_path", unexpected_read
    )
    assert _run_cli_test(cli_runner, "aider") == "http://localhost:8000/v1"


def test_cli_setup_none_endpoint_defaults_before_discovery(tmp_path, monkeypatch):
    from rapid_mlx.cli import agents_command

    calls = []

    def discover(url):
        calls.append(url)
        return "local-qwen", 32768

    monkeypatch.setattr("rapid_mlx.agents.adapter._detect_running_model", discover)
    args = _cli_args("codex", "--setup", "--no-check")
    args.base_url = None
    agents_command(args)
    assert calls == ["http://localhost:8000/v1"]
    config = tomllib.loads((tmp_path / "codex" / "config.toml").read_text())
    assert (
        config["model_providers"]["rapid-mlx"]["base_url"] == "http://localhost:8000/v1"
    )


def test_cli_instructions_none_endpoint_defaults(monkeypatch):
    from unittest.mock import Mock

    from rapid_mlx.cli import agents_command

    instructions = Mock(return_value="setup guide")
    monkeypatch.setattr("rapid_mlx.agents.adapter.get_setup_instructions", instructions)
    args = _cli_args("codex")
    args.base_url = None
    agents_command(args)
    assert instructions.call_args.args[1] == "http://localhost:8000/v1"


def test_agents_parser_preserves_omitted_endpoint():
    assert _cli_args("codex", "--test").base_url is None
