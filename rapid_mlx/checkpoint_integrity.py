# SPDX-License-Identifier: Apache-2.0
"""Detect checkpoints whose safetensors set is provably incomplete.

The checks use only the checkpoint's own manifest and file headers, never
model-class knowledge:

* ``model.safetensors.index.json`` names a shard file that is absent;
* a shard lacks tensors the index assigns to it;
* a ``model*.safetensors`` header points past the end of the file.

Only the small JSON headers are read; tensor data is never touched.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path, PurePosixPath

INDEX_FILENAME = "model.safetensors.index.json"
# A safetensors header is a JSON tensor table. Real headers are at most a few
# MiB even for thousand-tensor shards; refuse to parse anything larger.
_MAX_HEADER_BYTES = 64 * 1024 * 1024


class _UnreadableHeaderError(Exception):
    pass


def _read_header(path: Path) -> dict:
    """Return the tensor table or raise ``_UnreadableHeaderError``."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise _UnreadableHeaderError
            (header_len,) = struct.unpack("<Q", prefix)
            if header_len > _MAX_HEADER_BYTES or 8 + header_len > size:
                raise _UnreadableHeaderError
            header = json.loads(handle.read(header_len))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise _UnreadableHeaderError from exc
    if not isinstance(header, dict):
        raise _UnreadableHeaderError
    header.pop("__metadata__", None)
    data_end = 8 + header_len
    for entry in header.values():
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(value) is int for value in offsets)
            or not 0 <= offsets[0] <= offsets[1]
        ):
            raise _UnreadableHeaderError
        data_end = max(data_end, 8 + header_len + offsets[1])
    if data_end > size:
        raise _UnreadableHeaderError
    return header


def _safe_shard_name(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    return path.as_posix()


def incomplete_checkpoint_reason(model_path: str | Path) -> str | None:
    """Describe why ``model_path`` is an incomplete checkpoint, or ``None``.

    ``None`` also covers paths that are not directories and indexes that do
    not parse: those are left to the loader's own error reporting.
    """
    root = Path(model_path)
    try:
        if not root.is_dir():
            return None
    except OSError:
        return None

    checked: set[Path] = set()
    index_path = root / INDEX_FILENAME
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        index = None
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if isinstance(weight_map, dict):
        declared: dict[str, set[str]] = {}
        for tensor, raw_shard in weight_map.items():
            shard = _safe_shard_name(raw_shard)
            if shard is not None and isinstance(tensor, str):
                declared.setdefault(shard, set()).add(tensor)
        for shard, tensors in sorted(declared.items()):
            shard_path = root.joinpath(*PurePosixPath(shard).parts)
            checked.add(shard_path)
            if not shard_path.is_file():
                return f"{INDEX_FILENAME} lists {shard}, which is missing"
            try:
                header = _read_header(shard_path)
            except _UnreadableHeaderError:
                return f"{shard} is truncated or unreadable"
            absent = tensors.difference(header)
            if absent:
                return (
                    f"{shard} is missing {len(absent)} of the {len(tensors)} "
                    f"tensors {INDEX_FILENAME} assigns to it"
                )

    try:
        shards = sorted(root.glob("model*.safetensors"))
    except OSError:
        shards = []
    for shard_path in shards:
        if shard_path in checked:
            continue
        try:
            _read_header(shard_path)
        except _UnreadableHeaderError:
            return f"{shard_path.name} is truncated or unreadable"
    return None
