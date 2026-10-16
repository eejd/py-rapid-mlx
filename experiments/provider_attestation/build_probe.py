"""Build an isolated, ad-hoc hardened app. Never modifies Rapid.app or Keychain."""

import argparse
import pathlib
import plistlib
import subprocess


def build(destination: pathlib.Path) -> pathlib.Path:
    contents = destination / "RapidProviderProbe.app" / "Contents"
    executable = contents / "MacOS" / "RapidProviderProbe"
    executable.parent.mkdir(parents=True, exist_ok=True)
    (contents / "Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": "io.rapidmlx.attestation.feasibility",
                "CFBundleExecutable": executable.name,
                "CFBundleName": "Rapid Provider Feasibility Probe",
                "CFBundlePackageType": "APPL",
                "CFBundleVersion": "1",
                "CFBundleShortVersionString": "0.0.1",
                "LSUIElement": True,
            }
        )
    )
    source = pathlib.Path(__file__).with_name("ProviderProbe.swift")
    subprocess.run(
        [
            "xcrun",
            "swiftc",
            "-parse-as-library",
            "-O",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=120,
    )
    subprocess.run(
        [
            "codesign",
            "--force",
            "--sign",
            "-",
            "--options",
            "runtime",
            str(contents.parent),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run(
        ["codesign", "--verify", "--strict", "--deep", str(contents.parent)],
        check=True,
        timeout=30,
    )
    return executable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    print(build(parser.parse_args().output.resolve()))
