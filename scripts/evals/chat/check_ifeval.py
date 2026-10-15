#!/usr/bin/env python3
"""Programmatic strict IFEval using Google's pinned reference implementation."""
import argparse
import importlib
import json
import pathlib
import sys
import urllib.request

REV = "3ef52111c1fabc301880818f5a547570e926a76b"
MODULES = ("evaluation_lib.py", "instructions.py", "instructions_registry.py", "instructions_util.py")
CACHE = pathlib.Path.home() / ".cache/rapid-mlx/chat-eval"


def official_checker():
    package = CACHE / ("ifeval-code-" + REV)
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").touch()
    for name in MODULES:
        target = package / name
        if not target.exists():
            url = f"https://raw.githubusercontent.com/google-research/google-research/{REV}/instruction_following_eval/{name}"
            target.write_bytes(urllib.request.urlopen(url, timeout=20).read())
    # Official modules import their canonical package name.
    sys.path.insert(0, str(package.parent))
    sys.modules["instruction_following_eval"] = type(sys)("instruction_following_eval")
    sys.modules["instruction_following_eval"].__path__ = [str(package)]
    return importlib.import_module("instruction_following_eval.evaluation_lib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    official = official_checker()
    inputs = {str(row.key): row for row in official.read_prompt_list(str(CACHE / "ifeval.jsonl"))}
    allowed = set(json.loads((pathlib.Path(__file__).parent / "subset.json").read_text())["ids"]["ifeval"])
    output = {}
    for line in args.results.read_text().splitlines():
        row = json.loads(line)
        if row["benchmark"] != "ifeval" or str(row["id"]) not in allowed:
            continue
        inp = inputs[str(row["id"])]
        checked = official.test_instruction_following_strict(inp, {inp.prompt: row["answer"]})
        key = f'ifeval:{row["id"]}:{row["path"]}:{str(bool(row["web"])).lower()}'
        output[key] = "correct" if checked.follow_all_instructions else "incorrect"
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Scored {len(output)} strict IFEval responses")

if __name__ == "__main__":
    main()
