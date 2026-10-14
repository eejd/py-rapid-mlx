#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTENTS="$1"
HELPERS="$CONTENTS/Resources/Helpers"
mkdir -p "$HELPERS/local-documents" "$CONTENTS/Resources/Licenses"
cp "$ROOT/Work/local-documents/SKILL.md" "$HELPERS/local-documents/SKILL.md"
ARCHIVE="$ROOT/build/goose-aarch64-apple-darwin-v1.54.0.tar.gz"
EXPECTED=feda29744cd4e13f9a5211a968ed47c67fb8fa79647f4d271d92d564998a393c
if [[ ! -f "$ARCHIVE" ]]; then
    curl -fL --retry 3 -o "$ARCHIVE" \
        'https://github.com/aaif-goose/goose/releases/download/v1.54.0/goose-aarch64-apple-darwin.tar.gz'
fi
ACTUAL="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo "goose archive SHA-256 mismatch" >&2; exit 1; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tar -xzf "$ARCHIVE" -C "$TMP"
GOOSE_BIN="$(find "$TMP" -type f -name goose -print -quit)"
[[ -n "$GOOSE_BIN" ]] || { echo "goose executable missing" >&2; exit 1; }
cp "$GOOSE_BIN" "$HELPERS/goose"
chmod 755 "$HELPERS/goose"
cp "$ROOT/Work/work-inference-proxy.py" "$HELPERS/work-inference-proxy.py"
cp "$ROOT/Work/work-helper-watchdog.py" "$HELPERS/work-helper-watchdog.py"
curl -fLsS 'https://raw.githubusercontent.com/aaif-goose/goose/v1.54.0/LICENSE' \
    -o "$CONTENTS/Resources/Licenses/goose-APACHE-2.0.txt"
PYTHON="$CONTENTS/Resources/rapid-mlx/python/bin/python3.12"
SITE="$CONTENTS/Resources/rapid-mlx/site-packages"
[[ -x "$PYTHON" && -d "$SITE" ]] || { echo "Work MVP needs the bundled Python sidecar" >&2; exit 1; }
if [[ "${WORK_FULL:-0}" == "1" ]]; then
    python3 -m pip download --disable-pip-version-check --no-deps --only-binary=:all: \
        --dest "$HELPERS" 'pip==25.2'
    mv "$HELPERS"/pip-25.2-py3-none-any.whl "$HELPERS/pip.whl"
fi
python3 -m pip install --disable-pip-version-check --no-deps --target "$SITE" \
    'pypdf==6.1.3' 'openpyxl==3.1.5' 'et_xmlfile==2.0.0'
PYTHONPATH="$SITE" "$PYTHON" -c 'import pypdf, openpyxl, csv, decimal'
