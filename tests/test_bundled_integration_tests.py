# SPDX-License-Identifier: Apache-2.0
"""Keep bundled integration tests in sync with agent profile declarations."""

import fnmatch
import sys
from pathlib import Path

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 floor
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def _declared_specific_tests(profile: dict) -> list[str]:
    """Return every ``specific_tests`` name, including version overrides."""
    specs = [profile.get("testing") or {}]
    specs.extend((v or {}).get("testing") or {} for v in profile.get("versions") or [])
    return [s["specific_tests"] for s in specs if s.get("specific_tests")]


def test_bundled_integration_tests_match_profiles():
    package = REPO_ROOT / "rapid_mlx"
    bundled = package / "_integration_tests"
    referenced = set()
    for profile_path in sorted((package / "agents" / "profiles").glob("*.yaml")):
        profile = yaml.safe_load(profile_path.read_text())
        for name in _declared_specific_tests(profile):
            assert (bundled / name).is_file(), f"{profile_path.name}: missing {name}"
            referenced.add(name)

    bundled_names = {
        path.name
        for path in bundled.iterdir()
        if path.name != "__init__.py" and (path.is_file() or path.is_symlink())
    }
    assert bundled_names == referenced


def test_declared_specific_tests_reads_version_overrides():
    profile = {
        "testing": {"specific_tests": "a.py"},
        "versions": [
            {"range": ">=1", "testing": {"specific_tests": "b.py"}},
            {"range": "<1"},
        ],
    }
    assert _declared_specific_tests(profile) == ["a.py", "b.py"]


def test_package_data_ships_every_bundled_integration_test():
    """The wheel must carry each bundled test, not just the source tree."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    globs = pyproject["tool"]["setuptools"]["package-data"]["rapid_mlx"]
    bundled = REPO_ROOT / "rapid_mlx" / "_integration_tests"
    for path in bundled.iterdir():
        if path.name == "__init__.py" or not (path.is_file() or path.is_symlink()):
            continue
        rel = f"_integration_tests/{path.name}"
        assert any(fnmatch.fnmatch(rel, g) for g in globs), f"{rel} not packaged"
