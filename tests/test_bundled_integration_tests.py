# SPDX-License-Identifier: Apache-2.0
"""Keep bundled integration tests in sync with agent profile declarations."""

from pathlib import Path

import yaml


def _declared_specific_tests(profile: dict) -> list[str]:
    """Return every ``specific_tests`` name, including version overrides."""
    specs = [profile.get("testing") or {}]
    specs.extend((v or {}).get("testing") or {} for v in profile.get("versions") or [])
    return [s["specific_tests"] for s in specs if s.get("specific_tests")]


def test_bundled_integration_tests_match_profiles():
    package = Path(__file__).resolve().parents[1] / "rapid_mlx"
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
