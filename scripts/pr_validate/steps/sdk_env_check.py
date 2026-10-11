# SPDX-License-Identifier: Apache-2.0
"""Verify registered SDK imports without starting a server or installing packages."""

from __future__ import annotations

import json
import subprocess

from ..base import Step, StepResult
from ..context import Context
from .stress_e2e_bench import StressE2EBenchStep, _load_registry, _repo_python_env


class SDKEnvCheckStep(Step):
    name = "sdk_env_check"
    description = "verify registered SDK prerequisites in matrix Python"

    def should_run(self, ctx: Context) -> bool:
        return StressE2EBenchStep().should_run(ctx)

    def run(self, ctx: Context) -> StepResult:
        agents = _load_registry()["agents"]
        modules: set[str] = set()
        for agent in agents:
            imports = agent.get("prerequisite_imports")
            if (
                not isinstance(imports, list)
                or not imports
                or not all(isinstance(module, str) and module for module in imports)
                or not isinstance(agent.get("result_prefix"), str)
                or not agent["result_prefix"]
                or not (ctx.repo_root / agent["script"]).is_file()
            ):
                return StepResult(
                    name=self.name,
                    status="fail",
                    summary=f"incomplete SDK registration/script: {agent['name']}",
                )
            modules.update(imports)

        # Import modules only, never the integration scripts: their module-level
        # setup can contact the API. Match _run_agent's interpreter/cwd/env.
        probe = (
            "import importlib, sys\n"
            "failed = []\n"
            "for prerequisite in sys.argv[1:]:\n"
            "    try:\n"
            "        module, sep, symbol = prerequisite.partition(':')\n"
            "        imported = importlib.import_module(module)\n"
            "        if sep: getattr(imported, symbol)\n"
            "    except Exception as exc:\n"
            "        failed.append(prerequisite)\n"
            "        print(f'{prerequisite}: {type(exc).__name__}: {exc}')\n"
            "sys.exit(bool(failed))\n"
        )
        proc = subprocess.run(  # noqa: S603
            ["python3.12", "-c", probe, *sorted(modules)],
            cwd=str(ctx.repo_root),
            env=_repo_python_env(ctx.repo_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        log = ctx.artifact_path("sdk-env-check.log")
        log.write_text(
            f"interpreter: python3.12\nmodules: {json.dumps(sorted(modules))}\n"
            + (proc.stdout or "")
            + (proc.stderr or "")
        )
        return StepResult(
            name=self.name,
            status="pass" if proc.returncode == 0 else "fail",
            summary=(
                "registered SDK prerequisites import successfully"
                if proc.returncode == 0
                else "SDK prerequisites failed; repair the validator environment before validation"
            ),
            artifacts=[str(log)],
        )
