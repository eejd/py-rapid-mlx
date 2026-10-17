"""Keep Linux runner routing opt-in and fork-isolated across PR/main workflows."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
ROUTED = (
    "ci.yml",
    "rapid-mac-ci.yml",
    "version-check.yml",
    "pr-validate.yml",
    "merge-on-green.yml",
    "main-guard.yml",
    "authorize-merge-ready.yml",
)
MAC_JOBS = {
    "ci.yml": {"test-apple-silicon", "l1-smoke"},
    "rapid-mac-ci.yml": {"build", "gui-app-build", "gui-golden-flows"},
}
RUNNER_EXPRESSION = (
    "${{ fromJSON((vars.CI_LINUX_RUNNER != '' && (!github.event.pull_request || "
    "github.event.pull_request.head.repo.full_name == github.repository) && "
    "(!github.event.workflow_run || "
    "github.event.workflow_run.head_repository.full_name == github.repository)) && "
    """vars.CI_LINUX_RUNNER || '"ubuntu-latest"') }}"""
)


@pytest.mark.parametrize("filename", ROUTED)
def test_every_linux_job_uses_exact_fork_safe_runner_expression(filename):
    source = (WORKFLOWS / filename).read_text()
    jobs = yaml.safe_load(source)["jobs"]
    mac_jobs = MAC_JOBS.get(filename, set())
    assert mac_jobs <= jobs.keys()
    for name, job in jobs.items():
        if name in mac_jobs:
            if filename == "ci.yml":
                assert job["runs-on"] == "macos-14"
            else:
                assert job["runs-on"] == (
                    "${{ vars.MACOS_RUNNER == 'macos-26' && 'macos-26' "
                    "|| 'manzanita-standard' }}"
                )
        else:
            assert job["runs-on"] == RUNNER_EXPRESSION, name
    assert "runs-on: ubuntu-latest" not in source
    assert "gh variable delete CI_LINUX_RUNNER" in source
    assert (
        "Fork heads (pull_request, pull_request_target, workflow_run) always use "
        "ubuntu-latest." in source
    )


@pytest.mark.parametrize(
    "workflow",
    sorted(
        {
            WORKFLOWS / name
            for name in (
                "auto-release.yml",
                "rapid-mac-release.yml",
                "publish.yml",
                "pages.yml",
                "aggregate-bench.yml",
                "post-dmg-engine-recovery.yml",
                "mirror-drift-check.yml",
                "validate-community-submission.yml",
                "agent-gate.yml",
            )
        }
        | set(WORKFLOWS.glob("release-*.yml"))
    ),
    ids=lambda path: path.name,
)
def test_excluded_workflows_do_not_reference_linux_runner_variable(workflow):
    assert "CI_LINUX_RUNNER" not in workflow.read_text()
