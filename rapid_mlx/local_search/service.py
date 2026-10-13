# SPDX-License-Identifier: Apache-2.0
"""Event-loop-owned indexer; disk extraction alone is offloaded to asyncio.

MLX engine construction, loading, embedding, and list materialisation run
synchronously on the same loop thread as routes/embeddings.py. No MLX tensors
are handed to a worker. Small batches yield so chat admission can proceed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from . import MODEL_ID
from .chunking import Chunk, chunk_units
from .extraction import (
    AUDIO_SUFFIXES,
    LEGACY_OFFICE_SUFFIXES,
    MAX_EMBEDDED_CHUNKS_PER_FILE,
    SKIP_REASONS,
    SkippedError,
    excluded,
    skip_reason,
    extract,
    file_kind,
    media_frames,
    next_media_batch,
    read_safe,
    scan_batches,
    validate_folder,
)
from .eligibility import semantic_chunk
from .search import KINDS, fuse, grouped_rankings, query_statistics, query_terms, semantic_ranking, snippet, top_files
from .store import Store

logger = logging.getLogger(__name__)


def local_model_path(model_id: str) -> str | None:
    """Same fail-closed runnability probe as model_downloads.require_local.

    require_local itself is conditional on the global download flag, so it
    cannot alone enforce this feature's unconditional no-download policy.
    Resolve a runnable snapshot locally and hand the loader a filesystem path.
    """
    from huggingface_hub import snapshot_download

    from ..cli import _cache_runnability
    from ..model_metadata import (
        resolve_offline_cached_snapshot,
        resolve_unreferenced_cached_snapshot,
    )

    try:
        if _cache_runnability(model_id) is not True:
            return None
        resolved = resolve_unreferenced_cached_snapshot(
            model_id
        ) or resolve_offline_cached_snapshot(model_id)
        if resolved is not None:
            return str(resolved)
        return str(snapshot_download(model_id, local_files_only=True))
    except (OSError, ValueError):
        return None


def make_embedder(path: str):
    from ..embedding import EmbeddingEngine

    return EmbeddingEngine(
        path, max_length="auto", overflow_policy="error", vision=True
    )


class LocalSearch:
    def __init__(
        self,
        directory: Path,
        *,
        model_id=MODEL_ID,
        resolve_model=local_model_path,
        embedder_factory=make_embedder,
        batch_size=4,
        yield_seconds=0.01,
    ):
        self.model_id = model_id
        self.store = Store(directory / "index.sqlite3", model_id)
        self.resolve_model = resolve_model
        self.embedder_factory = embedder_factory
        self.engine = None
        self.batch_size = batch_size
        self.yield_seconds = yield_seconds
        self.pending: list[str] = []
        self.task: asyncio.Task | None = None
        self.progress = self._idle()
        self.closed = False

    @staticmethod
    def _idle():
        return {
            "active": False,
            "folder_id": None,
            "current_path": None,
            "files_done": 0,
            "files_total": 0,
        }

    def start(self):
        for folder in self.store.folders():
            self.queue(folder["id"])

    def queue(self, folder_id: str):
        if self.closed or not self.store.folder(folder_id):
            return
        if folder_id not in self.pending:
            self.pending.append(folder_id)
        if self.progress["folder_id"] != folder_id:
            self.store.set_folder(folder_id, "queued")
        self._start_worker()

    def _start_worker(self):
        if self.pending and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self._run())

    def status(self) -> dict:
        model_path = self.resolve_model(self.model_id)
        downloaded = model_path is not None
        # A status poll after an explicit model pull resumes waiting folders.
        if downloaded:
            for folder in self.store.folders():
                if (
                    folder["state"] == "waiting_for_model"
                    and self.progress["folder_id"] != folder["id"]
                ):
                    self.queue(folder["id"])
        folders = self.store.folders()
        for folder in folders:
            if not Path(folder["path"]).is_dir():
                folder["state"] = "missing"
        return {
            "enabled": True,
            "model": {
                "id": self.model_id,
                "downloaded": downloaded,
                "loaded": bool(self.engine and self.engine.is_loaded),
                "vision": self._vision_available(model_path),
            },
            "folders": folders,
            "indexing": dict(self.progress),
            "totals": self.store.totals(),
        }

    def _vision_available(self, model_path):
        if self.engine is not None and self.engine.is_loaded:
            return bool(getattr(self.engine, "vision", False))
        if model_path is None:
            return False
        try:
            config = json.loads((Path(model_path) / "config.json").read_text())
            return config.get("model_type") == "embedding_gemma2" and bool(
                config.get("vision_config")
            )
        except (OSError, ValueError):
            return False

    def add_folder(self, value: str) -> tuple[dict, bool]:
        path = validate_folder(value)
        folder_id, created = self.store.add_folder(path)
        self.pending = [i for i in self.pending if self.store.folder(i)]
        active_id = self.progress["folder_id"]
        if (
            active_id
            and not self.store.folder(active_id)
            and self.task
            and not self.task.cancelling()
        ):
            # No await between absorption and cancellation: the old indexer
            # cannot commit against its removed root before cancellation lands.
            self.task.cancel()
        self.queue(folder_id)
        return self.store.folder(folder_id), created

    async def delete_folder(self, folder_id: str):
        if self.progress["folder_id"] == folder_id:
            await self._cancel_worker()
        self.pending = [i for i in self.pending if i != folder_id]
        self.store.delete_folder(folder_id)
        self._start_worker()

    async def _cancel_worker(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None

    async def close(self):
        self.closed = True
        await self._cancel_worker()
        # Release the separate engine on its owning event-loop thread. There is
        # no compatible embedding residency lane in the server today.
        self.engine = None
        self.store.close()

    async def wait_idle(self):
        if self.task:
            await self.task

    def _embed(self, inputs: list, *, images=False):
        if self.engine is None:
            path = self.resolve_model(self.model_id)
            if path is None:
                return None
            self.engine = self.embedder_factory(path)
        method = self.engine.embed_images if images else self.engine.embed
        vectors = np.asarray(method(inputs), dtype=np.float32)
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(inputs)
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("invalid embedding output")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("zero embedding output")
        return vectors / norms

    async def _index_media(self, path, root, available):
        frames = media_frames(path, root)
        chunks, vectors = [], []
        error = None
        try:
            while True:
                # Finish the CPU read before closing its pinned descriptor on cancel.
                read = asyncio.create_task(
                    asyncio.to_thread(next_media_batch, frames, 4)
                )
                try:
                    batch = await asyncio.shield(read)
                except asyncio.CancelledError:
                    try:
                        await read
                    except Exception:
                        pass
                    raise
                if not batch:
                    break
                chunks.extend(
                    Chunk("", None, None, file_kind(path), frame.timestamp)
                    for frame in batch
                )
                if available:
                    try:
                        embedded = self._embed(
                            [frame.image for frame in batch], images=True
                        )
                        if embedded is None:
                            available = False
                        else:
                            vectors.extend(embedded)
                    except Exception as exc:
                        logger.warning("Local-search image model unavailable: %s", exc)
                        self.engine = None
                        available = False
                        error = str(exc)
                del batch
                await asyncio.sleep(self.yield_seconds)
        finally:
            frames.close()
        return chunks, vectors if available else None, available, error

    async def _run(self):
        try:
            while self.pending:
                folder_id = self.pending.pop(0)
                folder = self.store.folder(folder_id)
                if folder is None:
                    continue
                self.progress = {**self._idle(), "active": True, "folder_id": folder_id}
                self.store.set_folder(folder_id, "indexing")
                try:
                    await self._index(folder)
                except asyncio.CancelledError:
                    if not self.closed and self.store.folder(folder_id) is None:
                        # An absorbed root resumes through the ordinary queue;
                        # committed files now belong to the containing root.
                        continue
                    self.store.set_folder(folder_id, "queued")
                    raise
                except (FileNotFoundError, NotADirectoryError):
                    self.store.set_folder(folder_id, "missing")
                except Exception as exc:
                    logger.exception("Local-search indexing failed")
                    self.store.set_folder(folder_id, "error", error=str(exc))
                finally:
                    self.progress = self._idle()
        finally:
            self.progress = self._idle()

    async def _index(self, folder: dict):
        root = validate_folder(folder["path"])
        if str(root) != folder["path"]:
            raise ValueError(
                "folder was replaced by a symlink; remove and add it again"
            )
        old = self.store.files(folder["id"])
        existing = set()
        skipped = 0
        batches = scan_batches(root)
        index_directory = self.store.path.parent.resolve()

        async def discovered_paths():
            nonlocal skipped
            while True:
                # next(..., None) avoids propagating StopIteration through a Future.
                read = asyncio.create_task(asyncio.to_thread(next, batches, None))
                try:
                    batch = await asyncio.shield(read)
                except asyncio.CancelledError:
                    await read
                    raise
                if batch is None:
                    break
                paths, count = batch
                skipped += count
                self.store.set_folder(folder["id"], "indexing", skipped=skipped)
                for path in paths:
                    if path.parent.resolve() == index_directory:
                        continue
                    existing.add(str(path))
                    yield path
                await asyncio.sleep(self.yield_seconds)

        # Zero means discovery is ongoing; committed files are already searchable.
        self.progress["files_total"] = 0
        available = self.resolve_model(self.model_id) is not None
        model_error = None
        async for path in discovered_paths():
            self.progress["current_path"] = str(path)
            previous = old.get(str(path))
            try:
                info = path.lstat()
                if (
                    previous
                    and previous["size"] == info.st_size
                    and previous["mtime_ns"] == info.st_mtime_ns
                    and previous["status"] != "pending"
                    and not (available and previous["status"] == "keyword")
                ):
                    if previous["status"].startswith("skipped"):
                        skipped += 1
                    continue
                if excluded(path):
                    raise SkippedError(SKIP_REASONS["private"])
                if file_kind(path) != "text":
                    chunks, vectors, available, error = await self._index_media(
                        path, root, available
                    )
                    model_error = error or model_error
                    after = path.lstat()
                    if (info.st_size, info.st_mtime_ns, info.st_ino) != (
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ino,
                    ):
                        raise OSError(
                            "file changed during extraction; reindex to retry"
                        )
                    self.store.replace(
                        folder["id"],
                        path,
                        info,
                        None,
                        chunks,
                        vectors,
                        "indexed" if available else "keyword",
                    )
                    continue
                if path.suffix.lower() in LEGACY_OFFICE_SUFFIXES:
                    raise SkippedError(SKIP_REASONS["unsupported"])
                if path.suffix.lower() in AUDIO_SUFFIXES:
                    raise SkippedError(SKIP_REASONS["audio"])
                data, info = await asyncio.to_thread(read_safe, path, root)
                digest = hashlib.sha256(data).hexdigest()
                if (
                    previous
                    and previous["content_hash"] == digest
                    and previous["status"] == "indexed"
                ):
                    self.store.touch(previous["id"], info)
                    continue
                try:
                    units = await asyncio.to_thread(extract, path, data)
                except SkippedError:
                    raise
                except Exception as exc:
                    raise SkippedError("extraction failed") from exc
                chunks = chunk_units(units)
                if not chunks:
                    raise SkippedError(SKIP_REASONS["empty"])
                vectors = None
                eligible = [i for i, chunk in enumerate(chunks)
                            if semantic_chunk(path, chunk.text)]
                if available and eligible:
                    order = sorted(
                        eligible[:MAX_EMBEDDED_CHUNKS_PER_FILE],
                        key=lambda i: len(chunks[i].text),
                    )
                    ordered_vectors = []
                    try:
                        for start in range(0, len(order), self.batch_size):
                            batch = order[start : start + self.batch_size]
                            embedded = self._embed(
                                [
                                    f"title: {path.stem} | text: {chunks[i].text}"
                                    for i in batch
                                ]
                            )
                            if embedded is None:
                                available = False
                                break
                            ordered_vectors.extend(embedded)
                            await asyncio.sleep(self.yield_seconds)
                        if available:
                            vectors = [None] * len(chunks)
                            for i, vector in zip(order, ordered_vectors):
                                vectors[i] = vector
                    except Exception as exc:
                        logger.warning("Local-search model unavailable: %s", exc)
                        self.engine = None
                        available = False
                        model_error = str(exc)
                self.store.replace(
                    folder["id"],
                    path,
                    info,
                    digest,
                    chunks,
                    vectors,
                    "indexed" if available or not eligible else "keyword",
                )
            except SkippedError as exc:
                skipped += 1
                try:
                    skipped_info = path.lstat()
                except FileNotFoundError:
                    if previous:
                        self.store.remove_file(previous["id"])
                    continue
                self.store.replace(
                    folder["id"],
                    path,
                    skipped_info,
                    None,
                    [],
                    None,
                    "skipped: " + skip_reason(str(exc)),
                )
            except FileNotFoundError:
                if previous:
                    self.store.remove_file(previous["id"])
            except (OSError, ValueError) as exc:
                # Unreadable documents do not abort the folder.
                skipped += 1
                try:
                    skipped_info = path.lstat()
                except OSError:
                    if previous:
                        self.store.remove_file(previous["id"])
                else:
                    self.store.replace(folder["id"], path, skipped_info, None, [], None,
                                       "skipped: " + SKIP_REASONS["read"])
                logger.debug("Local-search skipped %s: %s", path, exc)
            finally:
                self.progress["files_done"] += 1
                self.store.set_folder(folder["id"], "indexing", skipped=skipped)
                await asyncio.sleep(self.yield_seconds)
        self.progress["files_total"] = self.progress["files_done"]
        for path, row in old.items():
            if path not in existing:
                self.store.remove_file(row["id"])
        self.store.set_folder(
            folder["id"],
            "idle" if available else "waiting_for_model",
            error=model_error,
            skipped=skipped,
            completed=True,
        )

    async def search(
        self,
        query: str,
        limit: int,
        folder_ids: list[str] | None,
        kinds: list[str] | None = None,
        group_limit: int | None = None,
    ) -> dict:
        start = time.perf_counter()
        terms = query_terms(query)
        rows = self.store.search_rows(folder_ids, kinds)
        distinctive = self.store.distinctive_terms(terms)
        highlight_terms = distinctive or [query.strip()]
        evidence = {}
        keyword = top_files(
            self.store.keyword(terms, set(rows), query=query, distinctive=distinctive,
                               evidence=evidence),
            rows,
            max(100, limit * 5),
        )
        semantic_legs = []
        semantic_scores = None
        semantic_standardized = {}
        mode = "keyword"
        try:
            vector = self._embed([f"task: search result | query: {query}"])
            if vector is not None:
                pooled_scores = self.store.semantic_scores(vector[0], set(rows))
                semantic_scores = dict(pooled_scores)
                background = self.store.semantic_background(rows)
                by_kind = {kind: [] for kind in KINDS}
                for i, score in pooled_scores:
                    by_kind[rows[i]["kind"]].append((i, score))
                for kind in KINDS:
                    scores = by_kind[kind]
                    statistics = query_statistics(scores, rows)
                    ids, mean, deviation = statistics
                    semantic_standardized.update({
                        i: (semantic_scores[i] - mean) / deviation if deviation > 0 else 0.0
                        for i in ids
                    })
                    semantic_legs.append(semantic_ranking(scores, rows, background[kind], statistics=statistics))
                mode = "hybrid"
        except Exception as exc:
            logger.warning("Local-search keyword fallback: %s", exc)
        include_groups = group_limit is not None and len(set(KINDS if kinds is None else kinds)) > 1
        folders = {f["id"]: Path(f["path"]) for f in self.store.folders()}
        valid_rows = {}

        def admitted(ranking):
            seen = set()
            for chunk_id, score, matched in ranking:
                row = rows[chunk_id]
                path = Path(row["path"])
                root = folders[row["folder_id"]]
                if row["path"] in seen or not path.is_file():
                    continue
                # Do not expose stale results after a source/root has been replaced
                # by a link pointing outside the registered folder.
                if root.resolve() != root or not path.resolve().is_relative_to(root):
                    continue
                seen.add(row["path"])
                valid_rows[chunk_id] = row
                yield chunk_id, score, matched

        def result(hit):
            chunk_id, score, matched = hit
            row = rows[chunk_id]
            text, highlights = (
                snippet(self.store.chunk_text(chunk_id), highlight_terms)
                if row["kind"] == "text" else ("", [])
            )
            locator = json.loads(row["locator"])
            return {
                "kind": row["kind"],
                "timestamp": locator.get("timestamp") if row["kind"] == "video" else None,
                "path": row["path"], "folder_id": row["folder_id"],
                "title": row["title"], "snippet": text, "highlights": highlights,
                "score": score, "matched": matched,
                "page": locator.get("page"), "line": locator.get("line"),
                "modified_at": row["mtime"], "size": row["size"],
            }

        ranking = fuse(
            [], keyword, rows=rows, semantic_legs=semantic_legs,
            semantic_scores=semantic_scores, keyword_evidence=evidence,
            semantic_standardized=semantic_standardized, limit=limit,
        )
        hits = []
        for hit in admitted(ranking):
            hits.append(hit)
            if not include_groups and len(hits) == limit:
                break
        response = {"results": [result(hit) for hit in hits[:limit]], "mode": mode}
        if include_groups:
            # Preserve all safe chunks of admitted files for lexical/semantic
            # representative selection, including the best video frame.
            paths = {row["path"] for row in valid_rows.values()}
            group_rows = {i: row for i, row in rows.items() if row["path"] in paths}
            groups = grouped_rankings(
                [i for i in keyword if i in group_rows], rows=group_rows,
                semantic_legs=[[i for i in leg if i in group_rows] for leg in semantic_legs],
                semantic_scores=semantic_scores,
                semantic_standardized={i: z for i, z in semantic_standardized.items() if i in group_rows},
                keyword_evidence=evidence,
            )
            response["groups"] = [
                {"kind": kind, "results": [result(hit) for hit in group[:group_limit]], "total": len(group)}
                for kind, group in groups
            ]
        response["took_ms"] = (time.perf_counter() - start) * 1000
        return response
