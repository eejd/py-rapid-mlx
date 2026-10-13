# SPDX-License-Identifier: Apache-2.0
"""One WAL database and a lazily materialised float32 search matrix.

All methods are owned by the server event loop. No SQLite/MLX objects cross
threads. Each replacement is an atomic per-file transaction.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np

from .chunking import CHUNKING_VERSION, Chunk
from .eligibility import ELIGIBILITY_VERSION, semantic_chunk
from .extraction import (
    AUDIO_SUFFIXES, EXTRACTION_VERSION, MAX_EMBEDDED_CHUNKS_PER_FILE,
    SKIP_REASONS, skip_reason,
)

SCHEMA_VERSION = "2"
DISTINCTIVE_FRACTION = 0.10
DISTINCTIVE_MIN_FILES = 1
# Retain at most a small vocabulary (roughly hundreds of KiB), not an unbounded
# history of user queries. Eviction affects computation only, never ranking.
FREQUENCY_CACHE_TERMS = 1024


class Store:
    def __init__(self, path: Path, model_id: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS folders (
                id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                state TEXT NOT NULL DEFAULT 'idle', files_skipped INTEGER NOT NULL DEFAULT 0,
                last_indexed_at REAL, error TEXT);
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY, folder_id TEXT NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
                path TEXT NOT NULL, title TEXT NOT NULL, size INTEGER NOT NULL,
                mtime REAL NOT NULL, mtime_ns INTEGER NOT NULL, content_hash TEXT,
                status TEXT NOT NULL, UNIQUE(folder_id, path));
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                text TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'text',
                locator TEXT NOT NULL, vector BLOB);
            CREATE INDEX IF NOT EXISTS chunks_file_id ON chunks(file_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, filename, tokenize='trigram');
            CREATE TRIGGER IF NOT EXISTS chunks_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid=old.id;
            END;
        """)
        version = self.db.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if version and version[0] not in ("1", SCHEMA_VERSION):
            self.db.close()
            raise ValueError("unsupported local-search schema version")
        previous = self.db.execute(
            "SELECT value FROM meta WHERE key='model_id'"
        ).fetchone()
        chunking = self.db.execute(
            "SELECT value FROM meta WHERE key='chunking_version'"
        ).fetchone()
        extraction = self.db.execute(
            "SELECT value FROM meta WHERE key='extraction_version'"
        ).fetchone()
        with self.db:
            if extraction is None or extraction[0] != EXTRACTION_VERSION:
                # Additive upgrade: keep unrelated chunks/vectors and refresh only
                # newly supported Office files, expanded DOCX, and empty results.
                self.db.execute("""UPDATE files SET status='pending'
                    WHERE lower(path) LIKE '%.docx' OR lower(path) LIKE '%.pptx'
                    OR lower(path) LIKE '%.xlsx'
                    OR (status IN ('indexed','keyword') AND NOT EXISTS (
                        SELECT 1 FROM chunks WHERE file_id=files.id))
                    OR (lower(path) LIKE '%.pdf' AND status LIKE 'skipped:%')""")
                for row in self.db.execute("SELECT id,path,status FROM files WHERE status LIKE 'skipped:%'").fetchall():
                    reason = skip_reason(row["status"].split(":", 1)[1].strip())
                    if Path(row["path"]).suffix.lower() in AUDIO_SUFFIXES:
                        reason = SKIP_REASONS["audio"]
                    self.db.execute("UPDATE files SET status=? WHERE id=?",
                                    ("skipped: " + reason, row["id"]))
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('extraction_version',?)",
                                (EXTRACTION_VERSION,))
            eligibility = self.db.execute(
                "SELECT value FROM meta WHERE key='eligibility_version'"
            ).fetchone()
            if eligibility is None or eligibility[0] != ELIGIBILITY_VERSION:
                # One streaming pass, no source reads or model work. Keep IDs,
                # FTS and locators; intentional keyword-only files are complete.
                cursor = self.db.execute("""SELECT chunks.id,text,files.path
                    FROM chunks JOIN files ON files.id=file_id
                    WHERE kind='text' AND vector IS NOT NULL""")
                for row in cursor:
                    if not semantic_chunk(Path(row["path"]), row["text"]):
                        self.db.execute("UPDATE chunks SET vector=NULL WHERE id=?", (row["id"],))
                for file in self.db.execute("SELECT id,path FROM files WHERE status='keyword'").fetchall():
                    texts = self.db.execute("SELECT text,kind FROM chunks WHERE file_id=?", (file["id"],)).fetchall()
                    if texts and all(r["kind"] == "text" and not semantic_chunk(Path(file["path"]), r["text"]) for r in texts):
                        self.db.execute("UPDATE files SET status='indexed' WHERE id=?", (file["id"],))
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('eligibility_version',?)",
                                (ELIGIBILITY_VERSION,))
            if (
                (version and version[0] != SCHEMA_VERSION)
                or (chunking is None or chunking[0] != CHUNKING_VERSION)
            ):
                self.db.execute("UPDATE chunks SET vector=NULL")
                self.db.execute(
                    "UPDATE files SET status='pending' WHERE status IN ('indexed','keyword') OR status LIKE 'skipped:%'"
                )
            elif previous and previous[0] != model_id:
                # Keyword-only files/chunks have no model-dependent state.
                # Ordinary keyword-to-vector upgrades still run when available.
                self.db.execute(
                    """UPDATE files SET status='pending' WHERE id IN (
                        SELECT file_id FROM chunks WHERE vector IS NOT NULL
                    )"""
                )
                self.db.execute("UPDATE chunks SET vector=NULL WHERE vector IS NOT NULL")
            self.db.executemany(
                "INSERT OR REPLACE INTO meta VALUES (?,?)",
                [
                    ("schema_version", SCHEMA_VERSION),
                    ("model_id", model_id),
                    ("chunking_version", CHUNKING_VERSION),
                ],
            )
            self.db.execute(
                "UPDATE folders SET state='queued', error=NULL WHERE state IN ('indexing','queued')"
            )
            # Apply the embedding budget to older indexes without rechunking or
            # re-embedding. Chunk IDs follow source order; FTS/locators stay intact.
            self.db.execute(
                """UPDATE chunks SET vector=NULL WHERE vector IS NOT NULL AND id IN (
                    SELECT id FROM (
                        SELECT id, row_number() OVER (PARTITION BY file_id ORDER BY id) AS position
                        FROM chunks WHERE kind='text'
                    ) WHERE position > ?
                )""",
                (MAX_EMBEDDED_CHUNKS_PER_FILE,),
            )
        self.vectors = {
            r["id"]: np.frombuffer(r["vector"], dtype="<f2").astype(np.float32)
            for r in self.db.execute(
                "SELECT id,vector FROM chunks WHERE vector IS NOT NULL"
            )
        }
        self._matrix = None
        self._ids: list[int] = []
        self._search_cache = {}
        self._frequency_cache = {}
        self._background_cache = None

    def _invalidate_search(self):
        self._search_cache.clear()
        self._frequency_cache.clear()
        self._background_cache = None

    def search_rows(self, folder_ids=None, kinds=None) -> dict[int, dict]:
        """Cache metadata, never the corpus bodies; callers treat rows as read-only.

        Invalidation follows committed index mutations. Scope caches are bounded
        to one view so arbitrarily many folder filters cannot retain the index.
        """
        key = (None if folder_ids is None else tuple(sorted(folder_ids)),
               None if kinds is None else tuple(sorted(kinds)))
        if key not in self._search_cache:
            # File strings are shared by all of a file's chunks. Fetching them
            # once avoids allocating thousands of repeated paths/metadata rows.
            file_sql = "SELECT id,path,folder_id,title,size,mtime,content_hash FROM files"
            file_params = []
            if folder_ids is not None:
                file_sql += " WHERE folder_id IN (" + ",".join("?" for _ in folder_ids) + ")"
                file_params = folder_ids
            files = {r["id"]: dict(r) for r in self.db.execute(file_sql, file_params)}
            sql = """SELECT chunks.id, chunks.file_id, chunks.kind, chunks.locator
                FROM chunks JOIN files ON file_id=files.id"""
            clauses, params = [], []
            for column, values in (("folder_id", folder_ids), ("kind", kinds)):
                if values is not None:
                    clauses.append(column + " IN (" + ",".join("?" for _ in values) + ")")
                    params.extend(values)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            self._search_cache.clear()
            self._search_cache[key] = {
                r["id"]: {**files[r["file_id"]], **dict(r)}
                for r in self.db.execute(sql, params)
            }
        return self._search_cache[key]

    def chunk_text(self, chunk_id: int) -> str:
        row = self.db.execute("SELECT text FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return row[0] if row else ""

    def close(self):
        self.db.close()

    def _forget(self, chunk_ids: list[int]):
        self._invalidate_search()
        for chunk_id in chunk_ids:
            self.vectors.pop(chunk_id, None)
        self._matrix = None

    def add_folder(self, path: Path) -> tuple[str, bool]:
        nested = []
        for row in self.db.execute("SELECT id,path FROM folders"):
            existing = Path(row["path"])
            if path.is_relative_to(existing):
                return row["id"], False
            if existing.is_relative_to(path):
                nested.append(row["id"])
        folder_id = uuid.uuid4().hex
        with self.db:
            self.db.execute(
                "INSERT INTO folders(id,path,state) VALUES (?,?,'queued')",
                (folder_id, str(path)),
            )
            # Preserve file/chunk IDs, fingerprints, FTS entries and cached vectors.
            for absorbed_id in nested:
                self.db.execute(
                    "UPDATE files SET folder_id=? WHERE folder_id=?",
                    (folder_id, absorbed_id),
                )
                self.db.execute("DELETE FROM folders WHERE id=?", (absorbed_id,))
        self._invalidate_search()
        return folder_id, True

    def folder(self, folder_id: str) -> dict | None:
        return next((f for f in self.folders() if f["id"] == folder_id), None)

    def folders(self) -> list[dict]:
        folders = [dict(r) for r in self.db.execute("""
            SELECT folders.*, (SELECT count(*) FROM files WHERE folder_id=folders.id AND status IN ('indexed','keyword','pending')) AS files_indexed,
            (SELECT count(*) FROM chunks JOIN files ON file_id=files.id WHERE folder_id=folders.id) AS chunks
            FROM folders ORDER BY path
        """)]
        reasons = {folder["id"]: {} for folder in folders}
        for row in self.db.execute("""SELECT folder_id,status,count(*) AS count
            FROM files WHERE status LIKE 'skipped:%' GROUP BY folder_id,status"""):
            reason = skip_reason(row["status"].split(":", 1)[1].strip())
            counts = reasons[row["folder_id"]]
            counts[reason] = counts.get(reason, 0) + row["count"]
        for folder in folders:
            folder["skip_reasons"] = reasons[folder["id"]]
        return folders

    def set_folder(
        self, folder_id: str, state: str, *, error=None, skipped=None, completed=False
    ):
        with self.db:
            self.db.execute(
                "UPDATE folders SET state=?,error=? WHERE id=?",
                (state, error, folder_id),
            )
            if skipped is not None:
                self.db.execute(
                    "UPDATE folders SET files_skipped=? WHERE id=?",
                    (skipped, folder_id),
                )
            if completed:
                self.db.execute(
                    "UPDATE folders SET last_indexed_at=? WHERE id=?",
                    (time.time(), folder_id),
                )

    def delete_folder(self, folder_id: str):
        chunk_ids = [
            r[0]
            for r in self.db.execute(
                "SELECT chunks.id FROM chunks JOIN files ON file_id=files.id WHERE folder_id=?",
                (folder_id,),
            )
        ]
        with self.db:
            self.db.execute("DELETE FROM folders WHERE id=?", (folder_id,))
        self._forget(chunk_ids)

    def remove_file(self, file_id: int):
        chunk_ids = [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM chunks WHERE file_id=?",
                (file_id,),
            )
        ]
        with self.db:
            self.db.execute("DELETE FROM files WHERE id=?", (file_id,))
        self._forget(chunk_ids)

    def files(self, folder_id: str) -> dict[str, dict]:
        return {
            r["path"]: dict(r)
            for r in self.db.execute(
                "SELECT * FROM files WHERE folder_id=?", (folder_id,)
            )
        }

    def touch(self, file_id: int, info):
        with self.db:
            self.db.execute(
                "UPDATE files SET size=?,mtime=?,mtime_ns=? WHERE id=?",
                (info.st_size, info.st_mtime, info.st_mtime_ns, file_id),
            )

        self._invalidate_search()

    def replace(
        self,
        folder_id: str,
        path: Path,
        info,
        digest: str | None,
        chunks: list[Chunk],
        vectors,
        status: str,
    ):
        old = self.db.execute(
            "SELECT id FROM files WHERE folder_id=? AND path=?", (folder_id, str(path))
        ).fetchone()
        additions = {}
        # Evict only after a successful commit; a failed transaction preserves
        # both old keyword results and the corresponding in-memory vectors.
        old_ids = (
            [
                r[0]
                for r in self.db.execute(
                    "SELECT id FROM chunks WHERE file_id=?", (old[0],)
                )
            ]
            if old
            else []
        )
        with self.db:
            if old:
                self.db.execute("DELETE FROM files WHERE id=?", (old[0],))
            file_id = self.db.execute(
                """INSERT INTO files(folder_id,path,title,size,mtime,mtime_ns,content_hash,status)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    folder_id,
                    str(path),
                    path.stem,
                    info.st_size,
                    info.st_mtime,
                    info.st_mtime_ns,
                    digest,
                    status,
                ),
            ).lastrowid
            for index, chunk in enumerate(chunks):
                vector = (
                    None
                    if vectors is None or vectors[index] is None
                    else np.asarray(vectors[index], dtype="<f2")
                )
                chunk_id = self.db.execute(
                    "INSERT INTO chunks(file_id,text,kind,locator,vector) VALUES (?,?,?,?,?)",
                    (
                        file_id,
                        chunk.text,
                        chunk.kind,
                        json.dumps(
                            {
                                "line": chunk.line,
                                "page": chunk.page,
                                "timestamp": chunk.timestamp,
                            }
                        ),
                        None if vector is None else vector.tobytes(),
                    ),
                ).lastrowid
                self.db.execute(
                    "INSERT INTO chunks_fts(rowid,text,filename) VALUES (?,?,?)",
                    (chunk_id, chunk.text, path.name),
                )
                if vector is not None:
                    additions[chunk_id] = vector.astype(np.float32)
        for chunk_id in old_ids:
            self.vectors.pop(chunk_id, None)
        self.vectors.update(additions)
        self._invalidate_search()
        self._matrix = None

    def rows(self, folder_ids: list[str] | None = None) -> dict[int, dict]:
        if folder_ids == []:
            return {}
        sql = "SELECT chunks.id, chunks.file_id, chunks.text, chunks.kind, chunks.locator, files.path, files.folder_id, files.title, files.size, files.mtime, files.id AS source_id FROM chunks JOIN files ON file_id=files.id"
        params = []
        if folder_ids is not None:
            sql += " WHERE folder_id IN (" + ",".join("?" for _ in folder_ids) + ")"
            params = folder_ids
        return {r["id"]: dict(r) for r in self.db.execute(sql, params)}

    def _ensure_matrix(self):
        if not self.vectors:
            self._ids = []
            self._matrix = None
            return
        if self._matrix is None:
            self._ids = list(self.vectors)
            self._matrix = np.stack([self.vectors[i] for i in self._ids])
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            self._matrix /= np.maximum(norms, 1e-12)

    def _score_matrix(self, query) -> np.ndarray:
        self._ensure_matrix()
        vector = np.asarray(query, dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        return self._matrix @ vector

    def semantic_scores(self, query, allowed: set[int]) -> list[tuple[int, float]]:
        if not self.vectors or not allowed:
            return []
        scores = self._score_matrix(query)
        return [
            (self._ids[int(i)], float(scores[int(i)]))
            for i in np.argsort(-scores, kind="stable")
            if self._ids[int(i)] in allowed
        ]

    def semantic(self, query, allowed: set[int]) -> list[int]:
        return [i for i, _ in self.semantic_scores(query, allowed)]

    def semantic_background(self, rows: dict[int, dict]) -> dict[str, float]:
        """Cross-file cosine backgrounds, capped by an equal-file shared pool.

        Remove within-file pairs: a file's own overlapping chunks/frames are
        not independent background evidence. Cache only the current scope.
        """
        if self._background_cache is not None and self._background_cache[0] is rows:
            return self._background_cache[1]
        self._ensure_matrix()
        sums = {}
        counts = {}
        for i, vector in zip(self._ids, self._matrix if self._matrix is not None else []):
            if i not in rows:
                continue
            key = (rows[i]["kind"], rows[i]["file_id"])
            counts[key] = counts.get(key, 0) + 1
            if key not in sums:
                sums[key] = np.zeros(self._matrix.shape[1], dtype=np.float64)
            sums[key] += vector
        # A shared embedding space supplies an independent pooled background
        # for a narrow/coherent kind. Give each file one centroid so long files
        # and repeated video frames cannot dominate the pool. Include every kind
        # in the same folders even when the request filters the displayed kinds.
        folders = sorted({r["folder_id"] for r in rows.values()})
        pool_sums, pool_counts = {}, {}
        if folders:
            metadata = self.db.execute(
                "SELECT chunks.id,file_id FROM chunks JOIN files ON files.id=file_id "
                "WHERE folder_id IN (" + ",".join("?" for _ in folders) + ")",
                folders,
            )
            matrix_positions = {i: n for n, i in enumerate(self._ids)}
            for r in metadata:
                if r["id"] not in matrix_positions:
                    continue
                file_id = r["file_id"]
                vector = self._matrix[matrix_positions[r["id"]]]
                if file_id not in pool_sums:
                    pool_sums[file_id] = np.zeros(len(vector), dtype=np.float64)
                pool_sums[file_id] += vector
                pool_counts[file_id] = pool_counts.get(file_id, 0) + 1
        pooled = 1.0
        if len(pool_sums) > 1:
            centroids = [v / pool_counts[i] for i, v in pool_sums.items()]
            total = np.sum(centroids, axis=0)
            pairs = len(centroids) * (len(centroids) - 1)
            pooled = max(0.0, min(1.0, (float(total @ total)
                         - sum(float(v @ v) for v in centroids)) / pairs))
        background = {}
        for kind in ("text", "image", "video"):
            keys = [key for key in sums if key[0] == kind]
            count = sum(counts[key] for key in keys)
            pairs = count * count - sum(counts[key] ** 2 for key in keys)
            if pairs:
                total = np.sum([sums[key] for key in keys], axis=0)
                value = (float(total @ total) - sum(float(sums[key] @ sums[key]) for key in keys)) / pairs
                background[kind] = max(0.0, min(pooled, value))
            else:
                # No independent background exists in a one-file scope.
                background[kind] = 0.0
        self._background_cache = (rows, background)
        return background

    @staticmethod
    def _match_expression(terms: list[str]) -> str:
        return " AND ".join('"' + t.replace('"', '""') + '"' for t in terms)

    def distinctive_terms(self, terms: list[str]) -> list[str]:
        """File frequency, independent of chunk count and embedding coverage."""
        count = self.db.execute("SELECT count(DISTINCT file_id) FROM chunks").fetchone()[0]
        threshold = max(DISTINCTIVE_MIN_FILES, int(count * DISTINCTIVE_FRACTION))
        distinctive = []
        for term in terms:
            folded = term.casefold()
            if folded not in self._frequency_cache:
                if len(term) >= 3:
                    frequency = self.db.execute(
                        "SELECT DISTINCT file_id FROM chunks_fts "
                        "JOIN chunks ON chunks.id=chunks_fts.rowid WHERE chunks_fts MATCH ? LIMIT ?",
                        (self._match_expression([term]), threshold + 1),
                    ).fetchall()
                    frequency = len(frequency)
                else:
                    # Trigrams cannot index short terms. Stop once enough files
                    # prove a term common, rather than scanning every chunk.
                    if term.isascii() or term.lower() == term.upper():
                        literal = "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                        frequency = len(self.db.execute(
                            "SELECT DISTINCT file_id FROM chunks_fts "
                            "JOIN chunks ON chunks.id=chunks_fts.rowid "
                            "WHERE chunks_fts.text LIKE ? ESCAPE '\\' OR filename LIKE ? ESCAPE '\\' LIMIT ?",
                            (literal, literal, threshold + 1),
                        ).fetchall())
                    else:
                        # Preserve Unicode casefold expansion for cased scripts.
                        files = set()
                        for r in self.db.execute(
                            "SELECT file_id,text,path FROM chunks JOIN files ON file_id=files.id"
                        ):
                            if folded in r["text"].casefold() or folded in Path(r["path"]).name.casefold():
                                files.add(r["file_id"])
                                if len(files) > threshold:
                                    break
                        frequency = len(files)
                # Bound retained query vocabulary independently of corpus size.
                if len(self._frequency_cache) >= FREQUENCY_CACHE_TERMS:
                    self._frequency_cache.clear()
                self._frequency_cache[folded] = frequency
            if self._frequency_cache[folded] <= threshold:
                distinctive.append(term)
        return distinctive

    def keyword(
        self,
        terms: list[str],
        allowed: set[int],
        *,
        query: str | None = None,
        distinctive: list[str] | None = None,
        evidence: dict[int, int] | None = None,
    ) -> list[int]:
        from .search import lexical_tier

        if not allowed:
            return []
        precise = self.distinctive_terms(terms) if distinctive is None else distinctive
        precise = list(dict.fromkeys(precise + [term for term in terms
            if any(mark in term[1:-1] for mark in "-./_")]))
        phrase = (query if query is not None else " ".join(terms)).strip()
        if not precise:
            if not phrase:
                return []
            precise = [phrase]
        long_terms = [term for term in precise if len(term) >= 3]
        if long_terms:
            candidates = list(self.db.execute(
                "SELECT rowid,text,filename FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts),rowid",
                (self._match_expression(long_terms),),
            ))
        elif all(term.isascii() or term.lower() == term.upper() for term in precise):
            # LIKE has no short trigram postings, but does the unavoidable scan
            # inside SQLite and materializes only matching bodies in Python.
            patterns = ["%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                        for term in precise]
            where = " AND ".join("(text LIKE ? ESCAPE '\\' OR filename LIKE ? ESCAPE '\\')" for _ in patterns)
            candidates = list(self.db.execute(
                "SELECT rowid,text,filename FROM chunks_fts WHERE " + where + " ORDER BY rowid",
                [pattern for pattern in patterns for _ in range(2)],
            ))
        else:
            candidates = list(self.db.execute("SELECT rowid,text,filename FROM chunks_fts ORDER BY rowid"))
        # Full basename evidence is independent of body-frequency admission.
        # This preserves filenames whose terms are common throughout the corpus.
        filename_terms = [term for term in terms if len(term) >= 3]
        if filename_terms:
            existing = {r[0] for r in candidates}
            candidates.extend(r for r in self.db.execute(
                "SELECT rowid,text,filename FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts),rowid",
                ("filename : (" + self._match_expression(filename_terms) + ")",),
            ) if r[0] not in existing)
        ranked, strengths = [], {}
        for chunk_id, text, filename in candidates:
            if chunk_id not in allowed:
                continue
            body, name = text.casefold(), filename.casefold()
            if not (all(t.casefold() in body or t.casefold() in name for t in precise)
                    or (terms and all(t.casefold() in name for t in terms))):
                continue
            strengths[chunk_id] = lexical_tier(phrase, terms, text, filename)
            ranked.append(chunk_id)
        # Stable BM25 ties remain stable; choose a full phrase before a weak
        # matching chunk from the same file, before file-level collapse.
        ranked.sort(key=lambda i: -strengths[i])
        if evidence is not None:
            evidence.update(strengths)
        return ranked

    def totals(self) -> dict:
        from .extraction import file_kind

        counts = {"text": 0, "image": 0, "video": 0}
        chunks, embedded = self.db.execute(
            "SELECT count(*), count(vector) FROM chunks"
        ).fetchone()
        for row in self.db.execute(
            "SELECT path FROM files WHERE status IN ('indexed','keyword','pending')"
        ):
            counts[file_kind(Path(row["path"]))] += 1
        return {
            "files_by_kind": counts,
            "files": self.db.execute(
                "SELECT count(*) FROM files WHERE status IN ('indexed','keyword','pending')"
            ).fetchone()[0],
            "chunks": chunks,
            "embedded_chunks": embedded,
            "keyword_only_chunks": chunks - embedded,
            "index_bytes": sum(
                p.stat().st_size
                for p in (
                    self.path,
                    Path(str(self.path) + "-wal"),
                    Path(str(self.path) + "-shm"),
                )
                if p.exists()
            ),
        }
