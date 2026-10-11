# SPDX-License-Identifier: Apache-2.0
"""SDK startup failures are validation failures, never baseline evidence."""

import sys
from contextlib import contextmanager
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts.pr_validate.base import Step, StepResult
from scripts.pr_validate.context import Context
from scripts.pr_validate.steps import sdk_env_check
from scripts.pr_validate.steps import stress_e2e_bench as matrix


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.chdir(tmp_path)
    context = Context(pr_number=123)
    context.work_dir = tmp_path / "artifacts"
    context.work_dir.mkdir()
    context.files_changed = ["rapid_mlx/server.py"]
    return context


@pytest.fixture
def agent(ctx):
    (ctx.repo_root / "sdk.py").touch()
    return {
        "name": "toy_sdk",
        "script": "sdk.py",
        "result_prefix": "Toy SDK",
        "prerequisite_imports": ["toy_sdk"],
    }


@pytest.mark.parametrize(
    "code,stdout,stderr,executed,status",
    [
        (1, "", "ModuleNotFoundError: No module named 'anthropic'", False, "fail"),
        (1, "", "ImportError: cannot import name 'Agent'", False, "fail"),
        (1, "", "RuntimeError: setup failed", False, "fail"),
        (0, "", "", False, "fail"),
        (0, "Toy SDK: 0/0 passed\n", "", False, "fail"),
        (0, "Toy SDK: 0/2 passed, 2 skipped\n", "", False, "fail"),
        (0, "Toy SDK: 3/2 passed\n", "", False, "fail"),
        (0, "Toy SDK: 1/2 passed\n", "", False, "fail"),
        (1, "Toy SDK: 2/2 passed\n", "", False, "fail"),
        (
            1,
            "Toy SDK: 1/2 passed\n",
            "Traceback (most recent call last)",
            False,
            "fail",
        ),
        (0, "Toy SDK: 2/2 passed\nToy SDK: 2/2 passed\n", "", False, "fail"),
        (0, "Toy SDK: 2/2 passed\n", "", True, "pass"),
        (0, "Toy SDK: 1/2 passed, 1 skipped\n", "", True, "pass"),
        (1, "Toy SDK: 1/2 passed\n", "", True, "fail"),
        (1, "Toy SDK: 0/2 passed\n", "", True, "fail"),
    ],
)
def test_sdk_requires_completed_nonempty_results(
    ctx, agent, monkeypatch, code, stdout, stderr, executed, status
):
    monkeypatch.setattr(
        matrix.subprocess,
        "run",
        lambda *a, **kw: CompletedProcess(a, code, stdout, stderr),
    )
    result = matrix._run_agent(
        ctx, matrix.ModelChoice("toy", "toy/model", 1, "golden", []), agent
    )
    assert result["executed"] is executed
    assert result["status"] == status
    assert Path(result["artifact"]).read_text() == stdout + stderr


def test_missing_registered_script_blocks(ctx, agent):
    (ctx.repo_root / "sdk.py").unlink()
    result = matrix._run_agent(
        ctx, matrix.ModelChoice("toy", "toy/model", 1, "golden", []), agent
    )
    assert result["status"] == "fail"
    assert result["executed"] is False


@pytest.mark.parametrize(
    "pr_executed,base_executed,expected",
    [(False, True, "fail"), (True, False, "fail"), (True, True, "pass")],
)
def test_only_real_test_failures_can_be_baseline_dispositioned(
    ctx, agent, monkeypatch, pr_executed, base_executed, expected
):
    registry = {
        "families": [
            {
                "family": "toy",
                "candidates": [
                    {"id": "toy/model", "ram_gb_required": 1, "quality_tier": "golden"}
                ],
            }
        ],
        "agents": [agent],
    }
    monkeypatch.setattr(matrix, "_load_registry", lambda: registry)
    monkeypatch.setattr(matrix, "_available_ram_gb", lambda: 64)

    @contextmanager
    def server(*args):
        yield "server.log"

    monkeypatch.setattr(matrix, "_server", server)
    monkeypatch.setattr(
        matrix, "_run_bench", lambda *a: {"status": "pass", "summary": "ok"}
    )
    monkeypatch.setattr(
        matrix, "_run_stress", lambda *a: {"status": "pass", "summary": "ok"}
    )
    monkeypatch.setattr(
        matrix,
        "_run_agent",
        lambda *a: {"status": "fail", "summary": "1/2 passed", "executed": pr_executed},
    )
    calls = []

    def base(*args):
        calls.append(args)
        return {"status": "fail", "summary": "1/2 passed", "executed": base_executed}

    monkeypatch.setattr(matrix, "_run_base_check", base)
    result = matrix.StressE2EBenchStep().run(ctx)
    assert result.status == expected
    assert bool(calls) is pr_executed


@pytest.mark.parametrize("returncode", [0, 1])
def test_sdk_prerequisites_use_matrix_interpreter_without_running_scripts(
    ctx, agent, monkeypatch, returncode
):
    monkeypatch.setattr(sdk_env_check, "_load_registry", lambda: {"agents": [agent]})
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return CompletedProcess(
            argv, returncode, "missing toy_sdk" if returncode else "", ""
        )

    monkeypatch.setattr(sdk_env_check.subprocess, "run", run)
    result = sdk_env_check.SDKEnvCheckStep().run(ctx)
    assert result.status == ("pass" if returncode == 0 else "fail")
    argv, kwargs = calls[0]
    assert argv[:2] == ["python3.12", "-c"]
    assert argv[-1] == "toy_sdk"
    assert "sdk.py" not in argv
    assert kwargs["cwd"] == str(ctx.repo_root)
    assert kwargs["timeout"] == 60


def test_sdk_env_failure_stops_before_costly_tests_without_fail_fast(ctx):
    from scripts.pr_validate.runner import run_pipeline

    class MissingSDK(Step):
        name = "sdk_env_check"

        def run(self, ctx):
            return StepResult(self.name, "fail", "missing SDK")

    class Costly(Step):
        def run(self, ctx):
            pytest.fail("must not reach model/full-unit work")

    assert run_pipeline(123, steps=[MissingSDK(), Costly()], fail_fast=False) == 1


def test_sdk_preflight_follows_supply_chain_and_precedes_full_unit():
    from scripts.pr_validate.runner import STEPS

    names = [step.name for step in STEPS]
    assert (
        names.index("supply_chain")
        < names.index("sdk_env_check")
        < names.index("full_unit")
    )


@pytest.mark.parametrize("missing", ["script", "prerequisite_imports", "result_prefix"])
def test_incomplete_registration_blocks_without_running_scripts(
    ctx, agent, monkeypatch, missing
):
    if missing == "script":
        (ctx.repo_root / agent[missing]).unlink()
    else:
        del agent[missing]
    monkeypatch.setattr(sdk_env_check, "_load_registry", lambda: {"agents": [agent]})
    monkeypatch.setattr(
        sdk_env_check.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("invalid registry must not launch"),
    )
    assert sdk_env_check.SDKEnvCheckStep().run(ctx).status == "fail"


@pytest.mark.parametrize(
    "missing,error",
    [
        (
            "_harbor_sdk_startup_negative_control_missing_module_0161",
            "ModuleNotFoundError",
        ),
        ("json:_harbor_missing_sdk_symbol_0161", "AttributeError"),
    ],
)
def test_prerequisite_probe_reports_real_import_failure_without_sdk_or_network(
    ctx, agent, monkeypatch, missing, error
):
    agent["prerequisite_imports"] = ["json", missing]
    monkeypatch.setattr(sdk_env_check, "_load_registry", lambda: {"agents": [agent]})
    real_run = sdk_env_check.subprocess.run

    def isolated_run(argv, **kwargs):
        return real_run([sys.executable, "-I", *argv[1:]], **kwargs)

    monkeypatch.setattr(sdk_env_check.subprocess, "run", isolated_run)
    result = sdk_env_check.SDKEnvCheckStep().run(ctx)
    assert result.status == "fail"
    log = Path(result.artifacts[0]).read_text()
    assert f"{missing}: {error}" in log

    # Exercise the real failing probe through the pipeline too: a module may
    # exist while a required class is missing, and costly tests must not run.
    from scripts.pr_validate import runner

    monkeypatch.setattr(runner, "Context", lambda **kwargs: ctx)

    class Costly(Step):
        name = "full_unit"

        def run(self, ctx):
            pytest.fail("missing prerequisite must stop before costly work")

    assert (
        runner.run_pipeline(123, steps=[sdk_env_check.SDKEnvCheckStep(), Costly()]) == 1
    )


@pytest.mark.parametrize(
    "files,override,expected",
    [
        (["rapid_mlx/server.py"], False, True),
        (["README.md"], False, False),
        (["tests/test_example.py"], False, False),
        (["rapid_mlx/server.py"], True, False),
    ],
)
def test_sdk_preflight_matches_model_matrix_gating(
    ctx, monkeypatch, files, override, expected
):
    ctx.files_changed = files
    monkeypatch.setenv("PR_VALIDATE_NO_STRESS", "1" if override else "0")
    assert sdk_env_check.SDKEnvCheckStep().should_run(ctx) is expected


def test_cpu_only_partial_pipeline_omits_matrix_prerequisites(ctx):
    from scripts.pr_validate.runner import run_pipeline

    class SDKProbe(Step):
        name = "sdk_env_check"

        def run(self, ctx):
            pytest.fail("CPU-only pipeline must not require the omitted SDK matrix")

    class CPUCheck(Step):
        name = "cpu_check"

        def run(self, ctx):
            return StepResult(self.name, "pass", "CPU check ran")

    assert (
        run_pipeline(
            123,
            steps=[SDKProbe(), CPUCheck()],
            skip_steps=("stress_e2e_bench",),
        )
        == 0
    )
