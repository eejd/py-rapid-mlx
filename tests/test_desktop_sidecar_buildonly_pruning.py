import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
SIDECAR_BUILD = ROOT / "apps" / "rapid-mac" / "scripts" / "build-sidecar.sh"


def test_torch_pruning_removes_only_build_payload() -> None:
    script = SIDECAR_BUILD.read_text()
    removals = re.findall(r"^rm -rf \\\n(?:[^\n]*\\\n)*[^\n]*", script, re.M)
    torch_removals = [block for block in removals if "/torch/" in block]

    assert len(torch_removals) == 1
    assert re.findall(
        r'"\$STAGE/site-packages/torch/([^"\n]+)"(\*)?', torch_removals[0]
    ) == [
        ("include", ""),
        ("share", ""),
        ("bin/protoc", ""),
        ("bin/protoc-", "*"),
    ]
    assert "torch_shm_manager" not in torch_removals[0]
    assert "torch/testing" not in torch_removals[0]
    assert not re.search(r'^rm -rf.*"\$STAGE/site-packages/torch"', script, re.M)


def test_test_suite_sweep_is_guarded_and_scoped_to_five_packages() -> None:
    script = SIDECAR_BUILD.read_text()
    sweep = re.search(r"^for package in (.+); do\n(.*?)^done$", script, re.M | re.S)

    assert sweep is not None
    assert sweep[1].split() == [
        "sympy",
        "networkx",
        "matplotlib",
        "mpmath",
        "mpl_toolkits",
    ]
    assert 'if [ -d "$STAGE/site-packages/$package" ]; then' in sweep[2]
    assert (
        'find "$STAGE/site-packages/$package" -type d '
        r"\( -name tests -o -name test \) -prune -exec rm -rf {} +"
    ) in sweep[2]
    assert "-name testing" not in sweep[2]


def test_build_payload_pruning_precedes_compileall() -> None:
    script = SIDECAR_BUILD.read_text()
    compileall = script.index('"$STAGE/python/bin/python3.12" -m compileall')

    assert script.index('"$STAGE/site-packages/torch/bin/protoc-"*') < compileall
    assert (
        script.index("for package in sympy networkx matplotlib mpmath mpl_toolkits;")
        < compileall
    )


def test_arm64_thinning_runs_after_inventory_before_counting_and_signing() -> None:
    script = SIDECAR_BUILD.read_text()
    inventory = script.index(
        '"$STAGE/python/bin/python3.12" "$ENGINE_ROOT/scripts/sidecar_macho_inventory.py" "$STAGE" "$MACHOS_LIST"'
    )
    thinning = script.index('lipo "$f" -thin arm64 -output "$f.thin.$$"')
    count = script.index('MACHO_COUNT="$(wc -l')
    skip_codesign = script.index('if [ "$SKIP_CODESIGN" = "1" ]; then', inventory)
    signing = script.index('echo "==> codesigning')

    assert inventory < thinning < count < skip_codesign < signing
    assert 'done < "$MACHOS_LIST"' in script[thinning:count]
    assert 'echo "==> thinned $THINNED_COUNT multi-arch Mach-Os to arm64"' in script


def test_arm64_thinning_fails_closed_and_preserves_modes() -> None:
    script = SIDECAR_BUILD.read_text()
    thinning = script.split("THINNED_COUNT=0", 1)[1].split('MACHO_COUNT="$(wc -l', 1)[0]

    assert 'if ! arches="$(lipo -archs "$f")"; then' in thinning
    assert 'if [ "${#MACHO_ARCHS[@]}" -gt 1 ]; then' in thinning
    assert '*" arm64 "*)' in thinning
    assert "ERR: multi-arch Mach-O lacks arm64:" in thinning
    assert "ERR: lipo could not inspect" in thinning
    assert "ERR: could not thin Mach-O to arm64:" in thinning
    assert "exit 1" in thinning
    assert "stat -f '%Lp' \"$f\"" in thinning
    assert thinning.index('chmod "$mode" "$f.thin.$$"') < thinning.index(
        'mv "$f.thin.$$" "$f"'
    )
