# SPDX-License-Identifier: Apache-2.0
"""CPU-only local-search pilot contracts; no MLX/model/network required."""

import asyncio
import io
import os
import sys
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from rapid_mlx.local_search.chunking import CHUNKING_VERSION, Chunk, chunk_units
from rapid_mlx.local_search.extraction import (
    MAX_EMBEDDED_CHUNKS_PER_FILE,
    SkippedError,
    Unit,
    extract,
    read_safe,
    scan,
    validate_folder,
)
from rapid_mlx.local_search.search import fuse, snippet
from rapid_mlx.local_search.service import LocalSearch, local_model_path
from rapid_mlx.local_search.store import Store
from rapid_mlx.middleware.exception_handlers import install_exception_handlers


@pytest.fixture(autouse=True)
def allow_test_scratch(monkeypatch):
    # macOS temp trees resolve under /private; never traverse the actual root.
    from rapid_mlx.local_search import extraction
    monkeypatch.setattr(extraction, "SKIP_PREFIXES", tuple(
        p for p in extraction.SKIP_PREFIXES if p != Path("/private")
    ))
    walk = os.walk

    def temp_tree_only(root, *args, **kwargs):
        assert Path(root).resolve() != Path("/"), "Tests must never walk the real root"
        yield from walk(root, *args, **kwargs)

    monkeypatch.setattr(os, "walk", temp_tree_only)


class FakeEmbedder:
    def __init__(self):
        self.vision = True
        self.image_calls = []
        self.calls = []
        self.threads = []
        self.is_loaded = False

    def embed(self, texts):
        self.calls.append(texts)
        self.threads.append(threading.get_ident())
        self.is_loaded = True
        return [
            [1.0, 1.0 if "apple" in t else 0.0, 1.0 if "banana" in t else 0.0]
            for t in texts
        ]

    def embed_images(self, images):
        self.image_calls.append(len(images))
        self.threads.append(threading.get_ident())
        self.is_loaded = True
        return [[0.2, 1.0, 0.0] for _ in images]


def engine(tmp_path, fake=None, **kwargs):
    fake = fake or FakeEmbedder()
    return LocalSearch(
        tmp_path / "index",
        resolve_model=lambda _: "local-checkpoint",
        embedder_factory=lambda _: fake,
        yield_seconds=0,
        **kwargs,
    )


def folder(tmp_path, name="source"):
    root = tmp_path / name
    root.mkdir()
    return root


@pytest.mark.parametrize("suffix", [".txt", ".md", ".py", ".json", ".yaml", ".csv"])
def test_text_extract(suffix):
    assert (
        extract(Path("file" + suffix), "中文 text\nsecond".encode())[0].text
        == "中文 text\nsecond"
    )


def test_html_docx_pdf():
    html = extract(
        Path("a.html"), b"<h1>Title</h1><p>A &amp; B</p><script>secret()</script>"
    )[0].text
    assert "A & B" in html and "secret" not in html and "<" not in html
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p><w:p><w:r><w:t>World</w:t></w:r></w:p></w:body></w:document>',
        )
    assert extract(Path("a.docx"), data.getvalue())[0].text == "Hello\nWorld"
    units = extract(Path("a.pdf"), small_pdf())
    assert [(u.text, u.page) for u in units] == [("First", 1), ("Second", 2)]
    assert [c.page for c in chunk_units(units)] == [1, 2]


def small_pdf():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for label in ("First", "Second"):
        page = writer.add_blank_page(300, 300)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 30 30 Td ({label}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    data = io.BytesIO()
    writer.write(data)
    return data.getvalue()


def test_pdf_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(SkippedError, match="unsupported"):
        extract(Path("a.pdf"), b"pdf")


@pytest.mark.parametrize(
    "path,data", [("a.txt", b"\x00abc"), ("a.txt", b"\xff"), ("a.mp4", b"movie")]
)
def test_binary_and_unsupported(path, data):
    with pytest.raises(SkippedError):
        extract(Path(path), data)


def test_scan_and_safe_read(tmp_path):
    root = folder(tmp_path)
    (root / "good.md").write_text("hello")
    for name in (
        ".env.production",
        "token.pem",
        "private.key",
        "id_rsa_backup",
        "login.keychain-db",
        ".hidden",
    ):
        (root / name).write_text("secret")
    for name in (
        ".git",
        ".ssh",
        ".aws",
        ".gnupg",
        "node_modules",
        "build",
        "tool.app",
        "venv",
    ):
        (root / name).mkdir()
        (root / name / "bad.txt").write_text("secret")
    (root / "outside.txt").symlink_to(tmp_path / "external.txt")
    files, skipped = scan(root)
    assert files == [root / "good.md"] and skipped == 15
    with pytest.raises(SkippedError, match="size cap"):
        read_safe(root / "good.md", root, 3)
    with pytest.raises(OSError):
        read_safe(root / "outside.txt", root)
    outside = folder(tmp_path, "outside")
    (outside / "bad.txt").write_text("secret")
    (root / "swapped").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        read_safe(root / "swapped" / "bad.txt", root)


def test_chunk_boundaries_and_lines():
    text = "a" * 960 + "\n# Heading\n" + "b" * 1300 + "\n\n" + "c" * 200
    chunks = chunk_units([Unit(text)])
    assert chunks[0].text == "a" * 960
    assert all(len(c.text) <= 1200 for c in chunks)
    assert chunks[1].text.startswith("# Heading")
    assert chunks[1].line == 2
    assert chunks[0].line == 1 and chunks[-1].line >= 3
    assert chunks[-1].text.endswith("c" * 200)
    assert chunk_units([Unit(" \n\n hello\nworld")])[0].line == 3
    assert chunk_units([Unit("  ")]) == []
    with pytest.raises(ValueError):
        chunk_units([Unit(text)], 100, 60)


def test_latin_chunks_never_split_words_and_preserve_source_lines():
    words = [f"word{i:03d}" for i in range(80)]
    text = "  \n\n" + "\n".join(" ".join(words[i : i + 13]) for i in range(0, 80, 13))
    chunks = chunk_units([Unit(text)], target=80, overlap=17)
    assert len(chunks) > 5
    assert {word for chunk in chunks for word in chunk.text.split()} == set(words)
    for chunk in chunks:
        offset = text.index(chunk.text)
        assert chunk.line == text.count("\n", 0, offset) + 1
        assert len(chunk.text) <= 80
    # A separator before the preferred final quarter still beats cutting a word.
    fallback = "short " + "x" * 79 + " tail words"
    assert chunk_units([Unit(fallback)], 80, 10)[0].text == "short"
    exact = chunk_units([Unit("a" * 80 + " next")], 80, 10)
    assert [c.text for c in exact] == ["a" * 80, "next"]


@pytest.mark.parametrize("punctuation", list("。！？；，"))
def test_cjk_chunks_snap_end_and_overlap_to_punctuation(punctuation):
    sentences = [chr(0x4E00 + i) * 13 + punctuation for i in range(20)]
    chunks = chunk_units([Unit("".join(sentences))], target=50, overlap=15)
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 and c.line == 1 for c in chunks)
    for chunk in chunks:
        assert chunk.text in "".join(sentences)
        assert len(chunk.text) % 14 == 0
        assert chunk.text.endswith(punctuation)
    assert all(any(sentence in c.text for c in chunks) for sentence in sentences)


def test_long_unbroken_tokens_and_unspaced_cjk_allow_hard_cuts():
    for token in ("x" * 301, "汉" * 301):
        chunks = chunk_units([Unit(token)], target=80, overlap=15)
        assert [len(c.text) for c in chunks] == [80, 80, 80, 80, 41]
        assert all(c.line == 1 for c in chunks)


@pytest.mark.parametrize("separator", ["\n", ". ", "。", "！", "？", "；", "，"])
def test_snippets_start_at_line_or_sentence_and_prefer_highlight_window(separator):
    text = ("Earlier readable text" + separator) * 30
    text += "Target apple and banana" + separator + "Following text " * 50
    excerpt, highlights = snippet(text, ["apple", "banana"], size=100)
    offset = text.index(excerpt)
    assert offset == 0 or text[:offset].endswith(separator)
    assert len(excerpt) <= 100
    assert [excerpt[a:b] for a, b in highlights] == ["apple", "banana"]


def test_snippets_do_not_cut_into_a_long_single_sentence():
    text = "Beginning " + "readable words " * 60 + "apple"
    excerpt, _ = snippet(text, ["apple"])
    assert excerpt.startswith("Beginning ")


def test_snippets_prefer_more_highlighted_terms():
    text = "Apple alone. " * 50 + "Both apple and banana appear here."
    excerpt, highlights = snippet(text, ["apple", "banana"], size=80)
    assert {excerpt[a:b].lower() for a, b in highlights} == {"apple", "banana"}


def test_semantic_only_snippet_skips_fences_and_punctuation_lines():
    text = "```python\n~~~\n---\n... ! _ \\\n\n  Readable source text\n```"
    excerpt, highlights = snippet(text, ["absent"])
    assert excerpt == "Readable source text\n```"
    assert highlights == []
    # A partial query term in a fence language label must not win the window.
    assert snippet(text, ["python", "absent"])[0] == excerpt
    assert snippet("```\n---", [])[0] == "```\n---"


@pytest.mark.parametrize("old_version", [None, "1"])
@pytest.mark.parametrize("available", [False, True])
def test_chunking_version_invalidates_and_rebuilds_unchanged_files(
    tmp_path, old_version, available
):
    async def scenario():
        root = folder(tmp_path)
        path = root / "a.txt"
        path.write_text("Readable apple text " * 140)
        search = engine(tmp_path)
        if not available:
            search.resolve_model = lambda _: None
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        original_file = search.store.files(fid)[str(path)]
        # Stand in for chunks created by the old raw-offset chunker.
        search.store.replace(
            fid,
            path,
            path.stat(),
            original_file["content_hash"],
            [Chunk("yte count, rate,", 1, None)],
            [[1, 0, 0]] if available else None,
            "indexed" if available else "keyword",
        )
        old_ids = set(search.store.rows())
        with search.store.db:
            if old_version is None:
                search.store.db.execute("DELETE FROM meta WHERE key='chunking_version'")
            else:
                search.store.db.execute(
                    "UPDATE meta SET value=? WHERE key='chunking_version'",
                    (old_version,),
                )
        await search.close()
        resumed = engine(tmp_path)
        if not available:
            resumed.resolve_model = lambda _: None
        assert not resumed.store.vectors
        assert resumed.store.files(fid)[str(path)]["status"] == "pending"
        assert (
            resumed.store.db.execute(
                "SELECT value FROM meta WHERE key='chunking_version'"
            ).fetchone()[0]
            == CHUNKING_VERSION
        )
        resumed.start()
        await resumed.wait_idle()
        assert old_ids.isdisjoint(resumed.store.rows())
        assert [r["text"] for r in resumed.store.rows().values()] == [
            c.text for c in chunk_units([Unit(path.read_text())])
        ]
        rebuilt = resumed.store.files(fid)[str(path)]
        assert rebuilt["content_hash"] == original_file["content_hash"]
        assert rebuilt["status"] == ("indexed" if available else "keyword")
        await resumed.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "value",
    ["relative", "/tmp/../tmp", "/tmp/.ssh", "/does-not-exist"],
)
def test_invalid_folder(value):
    with pytest.raises((ValueError, OSError)):
        validate_folder(value)


@pytest.mark.parametrize("name", [".hidden", ".ssh", ".aws", ".gnupg", "id_rsa", "keys.pem"])
@pytest.mark.parametrize("nested", [False, True])
def test_sensitive_folder_roots_rejected_at_add_time(tmp_path, name, nested):
    sensitive = folder(tmp_path, name)
    root = folder(sensitive, "documents") if nested else sensitive
    search = engine(tmp_path)
    try:
        with pytest.raises(ValueError, match="hidden, sensitive, or an excluded"):
            search.add_folder(str(root))
        assert search.store.folders() == []
        assert scan(root) == ([], 1)
    finally:
        asyncio.run(search.close())


def test_resolved_folder(tmp_path):
    root = folder(tmp_path)
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    assert validate_folder(str(link)) == root.resolve()
    assert validate_folder(str(root)) == root.resolve()


@pytest.mark.parametrize("root_kind", ["home", "ancestor", "outside"])
def test_home_and_package_exclusions_for_any_root(tmp_path, monkeypatch, root_kind):
    from rapid_mlx.local_search.extraction import PACKAGE_SUFFIXES

    home = folder(tmp_path, "home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    root = {"home": home, "ancestor": tmp_path, "outside": folder(tmp_path, "outside")}[root_kind]
    library = home / "Library"
    (library / "Application Support").mkdir(parents=True)
    (library / "Application Support" / "private.txt").write_text("secret")
    (root / "good.txt").write_text("apple")
    # Other folders called Library are ordinary user-selected content.
    ordinary_library = root / "Documents" / "Library"
    ordinary_library.mkdir(parents=True)
    (ordinary_library / "good.txt").write_text("banana")
    for suffix in PACKAGE_SUFFIXES:
        package = root / ("Example" + suffix.upper())
        package.mkdir()
        (package / "private.txt").write_text("secret")
        with pytest.raises(ValueError, match="excluded"):
            validate_folder(str(package))
        assert scan(package) == ([], 1)
    for selected_root in (library, library / "Application Support"):
        assert validate_folder(str(selected_root)) == selected_root.resolve()
        assert scan(selected_root) == ([library / "Application Support" / "private.txt"], 0)
    assert validate_folder(str(root)) == root.resolve()
    files, skipped = scan(root)
    assert set(files) == {root / "good.txt", ordinary_library / "good.txt"}
    assert skipped == len(PACKAGE_SUFFIXES) + (root_kind != "outside")

    async def scenario():
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        search.add_folder(str(root))
        await search.wait_idle()
        assert search.status()["totals"]["files"] == 2
        assert len(fake.calls) == 2
        assert {row["path"] for row in search.store.rows().values()} == {str(p) for p in files}
        await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("parent_first", [False, True])
def test_overlapping_folders_reuse_index_without_reembedding(tmp_path, parent_first):
    async def scenario():
        parent = folder(tmp_path)
        child = folder(parent, "docs")
        sibling = folder(parent, "downloads")
        (child / "a.txt").write_text("apple")
        (sibling / "b.txt").write_text("banana")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        try:
            first, second = (parent, child) if parent_first else (child, parent)
            registered, created = search.add_folder(str(first))
            assert created
            if not parent_first:
                search.add_folder(str(sibling))
            await search.wait_idle()
            before = search.store.rows()
            vectors = dict(search.store.vectors)
            assert len(fake.calls) == 2
            covering, created = search.add_folder(str(second))
            assert created == (not parent_first)
            assert (covering["id"] == registered["id"]) == parent_first
            assert len(search.store.folders()) == 1
            assert set(search.store.rows()) == set(before)
            assert all(r["folder_id"] == covering["id"] for r in search.store.rows().values())
            await search.wait_idle()
            assert len(fake.calls) == 2  # Absorption preserves both fingerprints.
            assert set(search.store.vectors) == set(vectors)
            alias = tmp_path / "alias"
            alias.symlink_to(child, target_is_directory=True)
            repeated, created = search.add_folder(str(alias))
            assert repeated["id"] == covering["id"] and not created
            await search.wait_idle()
            assert len(fake.calls) == 2
            for query in ("apple", "banana"):
                hits = (await search.search(query, 20, [covering["id"]]))["results"]
                assert hits and all(hit["folder_id"] == covering["id"] for hit in hits)
            await search.delete_folder(covering["id"])
            assert not search.store.rows() and not search.store.vectors
            assert search.store.totals()["files"] == 0
            assert not (await search.search("apple", 20, None))["results"]
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("active", [False, True])
def test_absorbing_queued_or_active_folder_resumes_under_new_root(tmp_path, monkeypatch, active):
    from rapid_mlx.local_search import service

    async def scenario():
        parent = folder(tmp_path)
        child = folder(parent, "docs")
        (child / "a.txt").write_text("apple")
        later = folder(child, "nested")
        (later / "b.txt").write_text("banana")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        paused = asyncio.Event()
        release = asyncio.Event()
        sleep = service.asyncio.sleep

        async def pause(seconds):
            if search.progress["files_done"] == 1 and not paused.is_set():
                paused.set()
                await release.wait()
            await sleep(seconds)

        monkeypatch.setattr(service.asyncio, "sleep", pause)
        try:
            old, _ = search.add_folder(str(child))
            if active:
                await asyncio.wait_for(paused.wait(), 5)
                assert len(fake.calls) == 1
            new, created = search.add_folder(str(parent))
            assert created and new["id"] != old["id"]
            assert not search.store.folder(old["id"])
            assert old["id"] not in search.pending
            release.set()
            await search.wait_idle()
            assert len(fake.calls) == 2
            assert search.store.folder(new["id"])["state"] == "idle"
            assert search.status()["totals"]["files"] == 2
            assert all(row["folder_id"] == new["id"] for row in search.store.rows().values())
        finally:
            release.set()
            await search.close()

    asyncio.run(scenario())


def test_pdf_index_and_search_preserve_page_locator(tmp_path, monkeypatch):
    async def scenario():
        root = folder(tmp_path)
        path = root / "pages.pdf"
        path.write_bytes(small_pdf())
        async with client_app(tmp_path, monkeypatch) as (client, app, fake):
            response = await client.post("/v1/local-search/folders", json={"path": str(root)})
            assert response.status_code == 201
            await app.state.local_search.wait_idle()
            assert fake.calls == [["title: pages | text: First", "title: pages | text: Second"]]
            for query, page in (("First", 1), ("Second", 2)):
                response = await client.post("/v1/local-search/search", json={"query": query})
                assert response.status_code == 200
                hit = response.json()["results"][0]
                assert "keyword" in hit["matched"]
                assert hit["path"] == str(path) and hit["page"] == page and hit["line"] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("parent_first", [False, True])
def test_folder_routes_accept_home_and_absorb_overlap(tmp_path, monkeypatch, parent_first):
    home = folder(tmp_path, "home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    docs = folder(home, "Documents")
    (docs / "a.txt").write_text("apple")

    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, app, fake):
            first, second = (home, docs) if parent_first else (docs, home)
            response = await client.post("/v1/local-search/folders", json={"path": str(first)})
            assert response.status_code == 201
            first_id = response.json()["id"]
            await app.state.local_search.wait_idle()
            response = await client.post("/v1/local-search/folders", json={"path": str(second)})
            assert response.status_code == (200 if parent_first else 201)
            assert (response.json()["id"] == first_id) == parent_first
            assert response.json()["path"] == str(home)
            await app.state.local_search.wait_idle()
            # Accept Whole Mac without walking the machine in a CPU test.
            monkeypatch.setattr(app.state.local_search, "queue", lambda _: None)
            response = await client.post("/v1/local-search/folders", json={"path": "/"})
            assert response.status_code == 201
            assert response.json()["path"] == "/"
            assert len(app.state.local_search.store.folders()) == 1
            assert len(fake.calls) == 1
            assert app.state.local_search.status()["totals"]["files"] == 1

    asyncio.run(scenario())


def test_store_transactions_matrix_model_change(tmp_path):
    root = folder(tmp_path)
    path = root / "a.txt"
    path.write_text("apple 中文检索")
    db = tmp_path / "index.sqlite3"
    store = Store(db, "model-a")
    fid, created = store.add_folder(root)
    assert created and store.add_folder(root) == (fid, False)
    store.replace(
        fid,
        path,
        path.stat(),
        "hash",
        [Chunk(path.read_text(), 1, None)],
        [[1, 0, 0]],
        "indexed",
    )
    row_id = next(iter(store.rows()))
    assert store.semantic([1, 0, 0], {row_id}) == [row_id]
    assert store._matrix.dtype == np.float32
    assert store.keyword(["中文检索"], {row_id}) == [row_id]
    assert store.keyword(["中文"], {row_id}) == [row_id]
    assert store.rows([]) == {}
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert len(store.db.execute("SELECT vector FROM chunks").fetchone()[0]) == 6
    store.close()
    store = Store(db, "model-b")
    assert not store.vectors
    assert store.db.execute("SELECT vector FROM chunks").fetchone()[0] is None
    assert next(iter(store.files(fid).values()))["status"] == "pending"
    assert store.keyword(["apple"], set(store.rows()))
    store.delete_folder(fid)
    assert store.totals()["chunks"] == 0
    assert store.db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 0
    store.close()


def test_incremental_add_modify_delete_hash_and_restart(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        path = root / "first.txt"
        path.write_text("apple")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        assert search.status()["totals"]["files"] == 1
        assert fake.calls == [["title: first | text: apple"]]
        search.queue(fid)
        await search.wait_idle()
        assert len(fake.calls) == 1
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        search.queue(fid)
        await search.wait_idle()
        assert len(fake.calls) == 1  # identical hash despite new mtime
        path.write_text("banana")
        (root / "second.md").write_text("apple banana")
        search.queue(fid)
        await search.wait_idle()
        assert len(fake.calls) == 3 and search.status()["totals"]["files"] == 2
        path.unlink()
        search.queue(fid)
        await search.wait_idle()
        assert (
            search.status()["totals"]["files"] == 1 and len(search.store.vectors) == 1
        )
        assert all(t == threading.get_ident() for t in fake.threads)
        search.store.set_folder(fid, "indexing")
        await search.close()
        restarted = engine(tmp_path, fake, model_id="changed-model")
        assert not restarted.store.vectors
        restarted.start()
        await restarted.wait_idle()
        assert len(fake.calls) == 4 and len(restarted.store.vectors) == 1
        await restarted.close()

    asyncio.run(scenario())


def test_text_embedding_budget_keyword_tail_and_small_semantic(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        path = root / "run.txt"
        lines = [
            ("tailneedle" if i == 900 else "apple banana").ljust(1190, " ")
            for i in range(1000)
        ]
        path.write_text("\n".join(lines))
        small = root / "small.txt"
        small.write_text("apple")
        class OrchardEmbedder(FakeEmbedder):
            def embed(self, texts):
                vectors = super().embed(texts)
                # Make orchard semantically closer to apple than to a generic
                # shared bias, rather than relying on a fitted cosine floor.
                return [[0.5, 1, 0] if "query: orchard" in text else vector
                        for text, vector in zip(texts, vectors)]

        fake = OrchardEmbedder()
        search = engine(tmp_path, fake)
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        large_batches = [batch for batch in fake.calls if batch[0].startswith("title: run |")]
        assert len(large_batches) == (MAX_EMBEDDED_CHUNKS_PER_FILE + 3) // 4
        assert sum(map(len, large_batches)) == MAX_EMBEDDED_CHUNKS_PER_FILE
        assert not any("tailneedle" in text for batch in large_batches for text in batch)
        totals = search.status()["totals"]
        assert totals["chunks"] == 1001
        assert totals["embedded_chunks"] == MAX_EMBEDDED_CHUNKS_PER_FILE + 1
        assert totals["keyword_only_chunks"] == 1000 - MAX_EMBEDDED_CHUNKS_PER_FILE
        results = (await search.search("tailneedle", 20, None))["results"]
        hit = next(r for r in results if r["path"] == str(path))
        assert hit["matched"] == ["keyword"] and hit["line"] == 901
        assert "tailneedle" in hit["snippet"] and hit["page"] is None
        assert sum(r["path"] == str(path) for r in results) == 1
        results = (await search.search("orchard", 20, None))["results"]
        assert [r["path"] for r in results] == [str(small)]
        assert results[0]["matched"] == ["semantic"]
        assert search.store._matrix.shape[0] == MAX_EMBEDDED_CHUNKS_PER_FILE + 1
        rows = set(search.store.rows())
        calls = len(fake.calls)
        search.queue(fid)
        await search.wait_idle()
        # Absorption also reuses mixed vector/keyword files without embedding.
        parent_id = search.add_folder(str(tmp_path))[0]["id"]
        await search.wait_idle()
        assert set(search.store.rows()) == rows and len(fake.calls) == calls
        await search.close()
        resumed = engine(tmp_path, fake, model_id="new-model")
        assert not resumed.store.vectors
        assert resumed.store.keyword(["tailneedle"], set(resumed.store.rows()))
        resumed.start()
        await resumed.wait_idle()
        assert sum(len(b) for b in fake.calls[calls:]) == MAX_EMBEDDED_CHUNKS_PER_FILE + 1
        assert resumed.store.folder(parent_id)["chunks"] == 1001
        await resumed.close()

    asyncio.run(scenario())


def test_existing_index_trims_text_vectors_without_reembedding(tmp_path):
    root = folder(tmp_path)
    path = root / "old.pdf"
    path.touch()
    db = tmp_path / "index.sqlite3"
    store = Store(db, "model")
    fid = store.add_folder(root)[0]
    count = MAX_EMBEDDED_CHUNKS_PER_FILE + 5
    store.replace(
        fid, path, path.stat(), "hash",
        [Chunk(f"page {i} tailneedle" if i == count else f"page {i}", None, i)
         for i in range(1, count + 1)],
        [[1.0, 0.0]] * count, "indexed",
    )
    media = root / "video.mp4"
    media.touch()
    store.replace(
        fid, media, media.stat(), None,
        [Chunk("", None, None, "video", float(i)) for i in range(count)],
        [[0.0, 1.0]] * count, "indexed",
    )
    ids = set(store.rows())
    store.close()
    store = Store(db, "model")
    assert set(store.rows()) == ids
    assert len(store.vectors) == MAX_EMBEDDED_CHUNKS_PER_FILE + count
    assert store.files(fid)[str(path)]["status"] == "indexed"
    tail = store.keyword(["tailneedle"], ids)[0]
    assert tail not in store.vectors
    assert f'"page": {count}' in store.rows()[tail]["locator"]
    keyword = root / "keyword.txt"
    keyword.touch()
    store.replace(
        fid, keyword, keyword.stat(), "hash",
        [Chunk("keywordonly", 1, None)], None, "keyword",
    )
    keyword_ids = set(store.keyword(["keywordonly"], set(store.rows())))
    store.close()
    store = Store(db, "new-model")
    assert not store.vectors
    assert store.files(fid)[str(keyword)]["status"] == "keyword"
    assert set(store.keyword(["keywordonly"], set(store.rows()))) == keyword_ids
    assert tail in store.rows() and tail not in store.vectors
    store.close()


def test_single_file_yields_and_cancels_between_embedding_batches(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "large.txt").write_text("apple\n" * 20000)
        fake = FakeEmbedder()
        first_batch = asyncio.Event()
        embed = fake.embed

        def observed(texts):
            vectors = embed(texts)
            first_batch.set()
            return vectors

        fake.embed = observed
        search = engine(tmp_path, fake)

        async def chat_and_cancel():
            await first_batch.wait()
            # A competing task runs before batch two of the same file.
            assert len(fake.calls) == 1
            search.task.cancel()

        chat = asyncio.create_task(chat_and_cancel())
        fid = search.add_folder(str(root))[0]["id"]
        with pytest.raises(asyncio.CancelledError):
            await search.wait_idle()
        await chat
        assert len(fake.calls) == 1 and search.store.folder(fid)["chunks"] == 0
        search.queue(fid)
        await search.wait_idle()
        assert len(fake.calls) == 1 + MAX_EMBEDDED_CHUNKS_PER_FILE // search.batch_size
        await search.close()

    asyncio.run(scenario())


def test_missing_model_keyword_and_resume(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("中文检索 apple")
        available = [False]
        fake = FakeEmbedder()
        search = LocalSearch(
            tmp_path / "index",
            resolve_model=lambda _: "local" if available[0] else None,
            embedder_factory=lambda _: fake,
            yield_seconds=0,
        )
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        status = search.status()
        assert status["folders"][0]["state"] == "waiting_for_model"
        assert status["model"]["downloaded"] is False and not fake.calls
        result = await search.search("中文", 20, None)
        assert result["mode"] == "keyword" and result["results"][0]["matched"] == [
            "keyword"
        ]
        available[0] = True
        assert search.status()["model"]["downloaded"] is True
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "idle" and fake.calls
        await search.close()

    asyncio.run(scenario())


def test_batches_yield_cancel_and_resume(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "long.txt").write_text("apple\n" * 4000)
        fake = FakeEmbedder()
        search = engine(tmp_path, fake, batch_size=2)
        fid = search.add_folder(str(root))[0]["id"]
        while not fake.calls:
            await asyncio.sleep(0.001)
        await search.close()
        assert search.store.path.exists()
        resumed = engine(tmp_path, fake, batch_size=2)
        resumed.start()
        await resumed.wait_idle()
        assert resumed.store.folder(fid)["chunks"] > 10
        assert all(len(batch) <= 2 for batch in fake.calls)
        assert all(
            len(batch[i]) <= len(batch[i + 1])
            for batch in fake.calls
            for i in range(len(batch) - 1)
        )
        await resumed.close()

    asyncio.run(scenario())


def test_fusion_exact_bonus_and_highlights():
    ranking = fuse([1, 2, 3], [3, 4])
    assert ranking[0][0] == 3 and ranking[0][2] == ["semantic", "keyword"]
    assert next(s for i, s, _ in ranking if i == 1) > next(
        s for i, s, _ in ranking if i == 4
    )
    assert {i: score for i, score, _ in fuse([1, 2, 3], [4, 5, 6])} == {
        i: score for i, score, _ in fuse([4, 5, 6], [1, 2, 3])
    }
    assert all(0 <= s <= 1 for _, s, _ in ranking)
    text, offsets = snippet(
        "Earlier sentence. " * 45 + "Apple 中文检索 " + "Later sentence. " * 45,
        ["apple", "中文"],
    )
    assert len(text) <= 500 and [text[a:b] for a, b in offsets] == ["Apple", "中文"]
    text, offsets = snippet("Straße apple", ["apple"])
    assert [text[a:b] for a, b in offsets] == ["apple"]


def test_search_collapses_filters_fallback(tmp_path):
    async def scenario():
        roots = [folder(tmp_path, "one"), folder(tmp_path, "two")]
        for root in roots:
            (root / "a.txt").write_text("apple\n" * 600)
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        ids = [search.add_folder(str(root))[0]["id"] for root in roots]
        await search.wait_idle()
        result = await search.search("apple", 20, None)
        assert result["mode"] == "hybrid" and len(result["results"]) == 2
        assert fake.calls[-1] == ["task: search result | query: apple"]
        result = await search.search("apple", 1, [ids[1]])
        assert (
            len(result["results"]) == 1 and result["results"][0]["folder_id"] == ids[1]
        )
        assert (await search.search("apple", 20, []))["results"] == []
        fake.embed = lambda _: (_ for _ in ()).throw(RuntimeError("unavailable"))
        assert (await search.search("apple", 20, None))["mode"] == "keyword"
        await search.close()

    asyncio.run(scenario())


def test_skipped_missing_and_bad_documents(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "bad.docx").write_bytes(b"not-a-zip")
        (root / "bad.txt").write_bytes(b"\0binary")
        (root / "unknown.bin").write_bytes(b"unknown")
        (root / "good.txt").write_text("apple")
        search = engine(tmp_path)
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        assert search.store.folder(fid)["files_skipped"] == 3
        assert search.store.folder(fid)["files_indexed"] == 1
        search.queue(fid)
        await search.wait_idle()
        assert search.store.folder(fid)["files_skipped"] == 3
        for path in root.iterdir():
            path.unlink()
        root.rmdir()
        search.queue(fid)
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "missing"
        assert not (await search.search("apple", 20, None))["results"]
        await search.close()

    asyncio.run(scenario())


def test_local_availability_never_downloads(monkeypatch):
    import huggingface_hub

    from rapid_mlx import cli

    monkeypatch.setattr(cli, "_cache_runnability", lambda _: False)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **kw: pytest.fail("missing model must not resolve/download"),
    )
    assert local_model_path("missing") is None
    monkeypatch.setattr(cli, "_cache_runnability", lambda _: True)
    calls = []

    def snapshot(model, **kwargs):
        calls.append(kwargs)
        return "/local/model"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert local_model_path("cached") == "/local/model"
    assert calls == [{"local_files_only": True}]


@asynccontextmanager
async def client_app(tmp_path, monkeypatch, enabled=True):
    import rapid_mlx.local_search.service as module
    from rapid_mlx.local_search import runtime

    fake = FakeEmbedder()
    monkeypatch.setattr(module, "LocalSearch", lambda directory: engine(tmp_path, fake))
    runtime.configure(enabled, str(tmp_path / "index"))
    app = FastAPI()
    install_exception_handlers(app)
    await runtime.startup(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            yield client, app, fake
    finally:
        await runtime.shutdown(app)
        runtime.configure(False)


def test_routes_and_frozen_shapes(tmp_path, monkeypatch):
    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("apple banana")
        async with client_app(tmp_path, monkeypatch) as (client, app, fake):
            status = (await client.get("/v1/local-search/status")).json()
            assert set(status) == {"enabled", "model", "folders", "indexing", "totals"}
            assert status["model"]["loaded"] is False and not fake.calls
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(root)}
            )
            assert response.status_code == 201
            folder_data = response.json()
            assert set(folder_data) == {
                "id",
                "path",
                "state",
                "files_indexed",
                "files_skipped",
                "skip_reasons",
                "chunks",
                "last_indexed_at",
                "error",
            }
            fid = folder_data["id"]
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(root)}
            )
            assert response.status_code == 200 and response.json()["id"] == fid
            await app.state.local_search.wait_idle()
            response = await client.post(
                "/v1/local-search/search", json={"query": "apple"}
            )
            assert response.status_code == 200
            body = response.json()
            assert set(body) == {"results", "took_ms", "mode"}
            result = body["results"][0]
            assert set(result) == {
                "kind",
                "timestamp",
                "path",
                "folder_id",
                "title",
                "snippet",
                "highlights",
                "score",
                "matched",
                "page",
                "line",
                "modified_at",
                "size",
            }
            assert result["page"] is None and result["line"] == 1
            assert result["snippet"][slice(*result["highlights"][0])] == "apple"
            for payload in ({"folder_id": fid}, {"folder_id": None}, {}):
                response = await client.post("/v1/local-search/reindex", json=payload)
                assert (
                    response.status_code == 202 and response.json()["enabled"] is True
                )
                await app.state.local_search.wait_idle()
            response = await client.delete(f"/v1/local-search/folders/{fid}")
            assert response.status_code == 204 and not response.content
            assert (await client.get("/v1/local-search/status")).json()["totals"][
                "chunks"
            ] == 0
            response = await client.delete(f"/v1/local-search/folders/{fid}")
            assert (
                response.status_code == 404
                and response.json()["error"]["code"] == "folder_not_found"
            )

    asyncio.run(scenario())


def test_flag_gating_and_cli(tmp_path, monkeypatch):
    from rapid_mlx.cli import build_parser

    args = build_parser().parse_args(["serve", "chat-model"])
    assert args.enable_local_search is False and args.local_search_dir is None
    args = build_parser().parse_args(
        ["serve", "chat-model", "--enable-local-search", "--local-search-dir", "/index"]
    )
    assert args.enable_local_search and args.local_search_dir == "/index"

    async def scenario():
        async with client_app(tmp_path, monkeypatch, enabled=False) as (client, app, _):
            for method, url, body in (
                ("GET", "status", None),
                ("POST", "folders", {"path": "/"}),
                ("DELETE", "folders/no-id", None),
                ("POST", "reindex", {}),
                ("POST", "search", {"query": "apple"}),
            ):
                response = await client.request(
                    method, "/v1/local-search/" + url, json=body
                )
                assert response.status_code == 404
            assert not any(
                "/v1/local-search" in getattr(r, "path", "") for r in app.routes
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "endpoint,body",
    [
        ("folders", {"path": ".ssh"}),
        ("folders", {"path": "relative"}),
        ("search", {"query": ""}),
        ("search", {"query": "   "}),
        ("search", {"query": "a", "limit": 0}),
        ("search", {"query": "a", "limit": 101}),
        ("search", {"query": "apple", "folder_ids": ["unknown"]}),
        ("reindex", {"folder_id": "unknown"}),
    ],
)
def test_route_errors(tmp_path, monkeypatch, endpoint, body):
    if body.get("path") == ".ssh":
        body = {"path": str(folder(tmp_path, ".ssh"))}

    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, _, _):
            response = await client.post("/v1/local-search/" + endpoint, json=body)
            assert response.status_code in (400, 404)
            assert set(response.json()["error"]) == {"message", "type", "code", "param"}

    asyncio.run(scenario())


def test_routes_share_auth_rate_dependencies(tmp_path, monkeypatch):
    from rapid_mlx.config import get_config
    from rapid_mlx.middleware import auth

    cfg = get_config()
    monkeypatch.setattr(cfg, "api_key", "test-key")
    monkeypatch.setattr(auth, "rate_limiter", auth.RateLimiter(1, enabled=True))

    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, _, _):
            response = await client.get("/v1/local-search/status")
            assert response.status_code == 401
            response = await client.get(
                "/v1/local-search/status", headers={"Authorization": "Bearer test-key"}
            )
            assert response.status_code == 200
            response = await client.post(
                "/v1/local-search/search",
                json={"query": "apple"},
                headers={"Authorization": "Bearer test-key"},
            )
            assert response.status_code == 429

    asyncio.run(scenario())


def test_default_state_directory(tmp_path, monkeypatch):
    from rapid_mlx.local_search.runtime import default_directory

    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path / "state"))
    assert default_directory() == tmp_path / "state" / "local-search"


def test_failed_replacement_preserves_index(tmp_path):
    root = folder(tmp_path)
    path = root / "a.txt"
    path.write_text("apple")
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    chunks = [Chunk("apple", 1, None)]
    store.replace(fid, path, path.stat(), "before", chunks, [[1.0, 0.0]], "indexed")
    before = set(store.rows())
    with pytest.raises(IndexError):
        store.replace(
            fid, path, path.stat(), "after", chunks * 2, [[0.0, 1.0]], "indexed"
        )
    assert set(store.rows()) == before and set(store.vectors) == before
    assert store.keyword(["apple"], before)
    store.close()


def test_active_delete_keeps_next_folder_job(tmp_path):
    async def scenario():
        one = folder(tmp_path, "one")
        two = folder(tmp_path, "two")
        (one / "long.txt").write_text("apple\n" * 4000)
        (two / "short.txt").write_text("banana")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake, batch_size=1)
        first = search.add_folder(str(one))[0]["id"]
        second = search.add_folder(str(two))[0]["id"]
        while not fake.calls:
            await asyncio.sleep(0.001)
        await search.delete_folder(first)
        await search.wait_idle()
        assert search.store.folder(first) is None
        assert search.store.folder(second)["files_indexed"] == 1
        assert len(search.store.vectors) == 1
        await search.close()

    asyncio.run(scenario())


def test_replaced_folder_symlink_is_rejected(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("apple")
        search = engine(tmp_path)
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        saved = tmp_path / "saved"
        root.rename(saved)
        root.symlink_to(saved, target_is_directory=True)
        assert (await search.search("apple", 20, None))["results"] == []
        search.queue(fid)
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "error"
        await search.close()

    asyncio.run(scenario())


def test_invalid_model_keeps_keyword_text(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("apple")
        fake = FakeEmbedder()
        fake.embed = lambda texts: [[float("nan")]] * len(texts)
        search = engine(tmp_path, fake)
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "waiting_for_model"
        assert search.store.folder(fid)["error"] == "invalid embedding output"
        assert not search.store.vectors
        result = await search.search("apple", 20, None)
        assert result["mode"] == "keyword" and len(result["results"]) == 1
        await search.close()

    asyncio.run(scenario())


def test_routes_keyword_without_model(tmp_path, monkeypatch):
    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("苹果 搜索")
        async with client_app(tmp_path, monkeypatch) as (client, app, fake):
            app.state.local_search.resolve_model = lambda _: None
            await client.post("/v1/local-search/folders", json={"path": str(root)})
            await app.state.local_search.wait_idle()
            status = (await client.get("/v1/local-search/status")).json()
            assert status["model"] == {
                "id": app.state.local_search.model_id,
                "downloaded": False,
                "loaded": False,
                "vision": False,
            }
            assert status["folders"][0]["state"] == "waiting_for_model"
            result = (
                await client.post("/v1/local-search/search", json={"query": "苹果"})
            ).json()
            assert result["mode"] == "keyword" and len(result["results"]) == 1
            assert not fake.calls

    asyncio.run(scenario())


def test_keyword_limits_do_not_starve_other_files(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "long.txt").write_text("apple\n" * 22000)
        (root / "short.txt").write_text("apple")
        search = LocalSearch(
            tmp_path / "index", resolve_model=lambda _: None, yield_seconds=0
        )
        search.add_folder(str(root))
        await search.wait_idle()
        result = await search.search("apple", 20, None)
        assert {Path(r["path"]).name for r in result["results"]} == {
            "long.txt",
            "short.txt",
        }
        await search.close()

    asyncio.run(scenario())


def test_unreferenced_available_snapshot_uses_existing_resolver(monkeypatch):
    import huggingface_hub

    from rapid_mlx import cli, model_metadata

    monkeypatch.setattr(cli, "_cache_runnability", lambda _: True)
    monkeypatch.setattr(
        model_metadata,
        "resolve_unreferenced_cached_snapshot",
        lambda _: Path("/local/unreferenced"),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **kw: pytest.fail("already resolved locally"),
    )
    assert local_model_path("test/model") == "/local/unreferenced"


@pytest.mark.parametrize("identifier", [False, True])
def test_precise_hybrid_ranking_and_highlights(tmp_path, identifier):
    class RankedEmbedder(FakeEmbedder):
        def embed(self, texts):
            self.is_loaded = True
            vectors = []
            for text in texts:
                if text.startswith("task:"):
                    vectors.append([1.0, 0.0])
                else:
                    name = text.split(" | ")[0].removeprefix("title: ")
                    rank = int(name[-1])
                    vectors.append([1.0, rank / 5.0])
            return vectors

    async def scenario():
        root = folder(tmp_path)
        common = "how do I roll back a bad "
        target = 2 if identifier else 0
        for i in range(6):
            content = common + "configuration"
            if i == target:
                content += " RAPID_MLX_HOME" if identifier else " release"
            (root / f"guide{i}.md").write_text(content)
        search = engine(tmp_path, RankedEmbedder())
        search.add_folder(str(root))
        await search.wait_idle()
        query = "how do I RAPID_MLX_HOME" if identifier else common + "release"
        rows = search.store.rows()
        vector = search._embed(["task: query"])[0]
        pure = search.store.semantic(vector, set(rows))
        assert Path(rows[pure[target]]["path"]).name == f"guide{target}.md"
        result = await search.search(query, 4, None)
        assert Path(result["results"][0]["path"]).name == f"guide{target}.md"
        first = result["results"][0]
        assert first["matched"] == ["semantic", "keyword"]
        assert [first["snippet"][a:b] for a, b in first["highlights"]] == [
            "RAPID_MLX_HOME" if identifier else "release"
        ]
        await search.close()

    asyncio.run(scenario())


def test_no_distinctive_terms_require_literal_phrase(tmp_path):
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    for i in range(4):
        path = root / f"guide{i}.txt"
        path.write_text("how to do a rollback")
        store.replace(
            fid,
            path,
            path.stat(),
            None,
            [Chunk(path.read_text(), 1, None)],
            None,
            "keyword",
        )
    allowed = set(store.rows())
    assert store.distinctive_terms(["how", "do", "a", "rollback"]) == []
    assert (
        store.keyword(
            ["how", "do", "a", "rollback"], allowed, query="how do a rollback"
        )
        == []
    )
    assert store.keyword(["how", "to"], allowed, query="how to")
    assert store.keyword(["how", "to"], allowed, query="how-to") == []
    store.close()


def test_distinctive_all_terms_cjk_and_filename(tmp_path):
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    for name, text in [
        ("RAPID_MLX_HOME.txt", "苹果 错误 code42"),
        ("other.txt", "香蕉 code42"),
        ("third.txt", "普通内容"),
    ]:
        path = root / name
        path.write_text(text)
        store.replace(
            fid, path, path.stat(), None, [Chunk(text, 1, None)], None, "keyword"
        )
    rows = store.rows()
    allowed = set(rows)
    expected = [i for i, row in rows.items() if "HOME" in row["path"]]
    assert store.keyword(["苹果", "错误"], allowed) == expected
    assert store.keyword(["苹果", "香蕉"], allowed) == []
    assert store.keyword(["错误", "RAPID_MLX_HOME"], allowed) == expected
    assert store.keyword(["RAPID_MLX_HOME"], allowed) == expected
    assert store.keyword(["missing", "RAPID_MLX_HOME"], allowed) == []
    store.close()


def test_fusion_combines_different_chunks_of_same_file():
    rows = {1: {"path": "right"}, 2: {"path": "other"}, 3: {"path": "right"}}
    ranking = fuse([1, 2], [3], rows=rows)
    assert ranking[0][0] == 3
    assert ranking[0][1] == pytest.approx(1.0)
    assert ranking[0][2] == ["semantic", "keyword"]
    assert len(ranking) == 2
    assert fuse([1], [2])[0][1] == fuse([2], [1])[0][1]
    assert fuse([1], [1])[0][1] > fuse([2], [1])[0][1]


def write_video(path, colors=(0, 0, 255), fps=2, seconds=12):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    assert writer.isOpened(), "test codec must be available"
    try:
        for frame in range(fps * seconds):
            color = colors[min(len(colors) - 1, int(frame / fps / 5))]
            writer.write(np.full((32, 32, 3), color, dtype=np.uint8))
    finally:
        writer.release()


@pytest.mark.parametrize(
    "suffix", ["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"]
)
def test_image_extraction_first_frame(tmp_path, suffix):
    from PIL import Image

    from rapid_mlx.local_search.extraction import media_frames

    root = folder(tmp_path)
    path = root / ("picture." + suffix)
    red = Image.new("RGB", (32, 24), "red")
    if suffix == "gif":
        red.save(
            path, save_all=True, append_images=[Image.new("RGB", red.size, "blue")]
        )
    else:
        red.save(path)
    frames = list(media_frames(path, root))
    assert len(frames) == 1 and frames[0].timestamp is None
    assert frames[0].image.mode == "RGB"
    pixel = frames[0].image.getpixel((0, 0))
    assert pixel[0] >= 250 and pixel[2] <= 5


def test_video_sampling_timestamp_dedupe_and_cap(tmp_path):
    from rapid_mlx.local_search.extraction import media_frames, sample_timestamps

    root = folder(tmp_path)
    path = root / "clip.mp4"
    write_video(path)
    frames = list(media_frames(path, root))
    # Consecutive black frames at 0 and 5 seconds collapse; white at 10 survives.
    assert [f.timestamp for f in frames] == [0.0, 10.0]
    assert frames[0].image.getpixel((0, 0))[0] < 5
    assert frames[1].image.getpixel((0, 0))[0] > 245
    assert sample_timestamps(1200, 2) == [float(i) for i in range(0, 600, 5)]
    long = sample_timestamps(7200, 2)
    assert len(long) == 120 and long[0] == 0 and long[-1] == 3599.5
    assert np.diff(long) == pytest.approx([3599.5 / 119] * 119)
    for count, fps in [(0, 1), (100, 0), (100, float("nan"))]:
        with pytest.raises(SkippedError):
            sample_timestamps(count, fps)


def test_video_six_second_scenes_survive_five_second_sampling(tmp_path):
    import cv2
    from rapid_mlx.local_search.extraction import media_frames

    root = folder(tmp_path)
    path = root / "scenes.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    assert writer.isOpened()
    try:
        for color in (0, 80, 160, 240):
            for _ in range(60):
                writer.write(np.full((32, 32, 3), color, np.uint8))
    finally:
        writer.release()
    frames = list(media_frames(path, root))
    assert [f.timestamp for f in frames] == [0.0, 10.0, 15.0, 20.0]
    for index, frame in enumerate(frames):
        assert index * 6 <= frame.timestamp < (index + 1) * 6
        assert abs(frame.image.getpixel((0, 0))[0] - index * 80) < 6


def test_video_long_decode_is_bounded(tmp_path, monkeypatch):
    import cv2

    from rapid_mlx.local_search.extraction import media_frames

    root = folder(tmp_path)
    path = root / "long.mkv"
    path.write_bytes(b"fake container")
    seeks = []

    class Capture:
        def isOpened(self):  # noqa: N802 - OpenCV interface
            return True

        def get(self, key):
            return 2 if key == cv2.CAP_PROP_FPS else 7200

        def set(self, key, value):
            seeks.append(value)
            return True

        def read(self):
            return True, np.full((8, 8, 3), len(seeks) % 2 * 255, np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    frames = list(media_frames(path, root))
    assert len(frames) == len(seeks) == 120
    assert frames[-1].timestamp == 3599.5 and capture.released


@pytest.mark.parametrize("suffix", ["png", "mp4", "heic"])
def test_bad_media_skipped_and_size_caps(tmp_path, suffix):
    from rapid_mlx.local_search.extraction import (
        IMAGE_MAX_BYTES,
        VIDEO_MAX_BYTES,
        media_frames,
    )

    root = folder(tmp_path)
    path = root / ("bad." + suffix)
    path.write_bytes(b"not media")
    with pytest.raises(SkippedError):
        list(media_frames(path, root))
    with path.open("wb") as stream:
        stream.truncate((VIDEO_MAX_BYTES if suffix == "mp4" else IMAGE_MAX_BYTES) + 1)
    with pytest.raises(SkippedError, match="size cap"):
        list(media_frames(path, root))


def test_media_index_routes_kinds_and_keyword_fallback(tmp_path, monkeypatch):
    from PIL import Image

    async def scenario():
        root = folder(tmp_path)
        (root / "z-document.txt").write_text("apple")
        Image.new("RGB", (24, 24), "red").save(root / "a-apple.png")
        write_video(root / "b-apple.mp4")
        (root / "bad.png").write_bytes(b"corrupt")
        async with client_app(tmp_path, monkeypatch) as (client, app, fake):
            # Media cannot start until the folder's documents are searchable.
            original = fake.embed_images

            def check_order(images):
                assert any(
                    row["kind"] == "text"
                    for row in app.state.local_search.store.rows().values()
                )
                return original(images)

            fake.embed_images = check_order
            await client.post("/v1/local-search/folders", json={"path": str(root)})
            await app.state.local_search.wait_idle()
            status = (await client.get("/v1/local-search/status")).json()
            assert status["totals"]["files_by_kind"] == {
                "text": 1,
                "image": 1,
                "video": 1,
            }
            assert status["model"]["vision"] is True
            assert status["folders"][0]["files_skipped"] == 1
            assert fake.image_calls == [1, 2]
            assert set(fake.threads) == {threading.get_ident()}
            for kinds in (None, ["text"], ["image"], ["video"], [], ["image", "video"]):
                response = await client.post(
                    "/v1/local-search/search", json={"query": "apple", "kinds": kinds}
                )
                assert response.status_code == 200
                hits = response.json()["results"]
                assert {hit["kind"] for hit in hits} == set(
                    kinds if kinds is not None else ["text", "image", "video"]
                )
                for hit in hits:
                    if hit["kind"] != "text":
                        assert hit["snippet"] == "" and hit["highlights"] == []
                        assert hit["page"] is hit["line"] is None
                        # High-cosine vectors pass the available background gate.
                        assert hit["matched"] == ["semantic", "keyword"]
                    assert (hit["timestamp"] is not None) == (hit["kind"] == "video")
                grouped = (await client.post("/v1/local-search/search", json={
                    "query": "apple", "kinds": kinds, "group_limit": 1,
                })).json()
                assert grouped["results"] == hits
                if kinds is None or len(set(kinds)) > 1:
                    assert {g["kind"] for g in grouped["groups"]} == {hit["kind"] for hit in hits}
                    assert all(g["total"] == len(g["results"]) == 1 for g in grouped["groups"])
                    for group in grouped["groups"]:
                        if group["kind"] == "video":
                            assert group["results"][0]["timestamp"] is not None
                else:
                    assert "groups" not in grouped
            for invalid in (0, 51, -1, "invalid"):
                assert (await client.post("/v1/local-search/search", json={
                    "query": "apple", "group_limit": invalid,
                })).status_code == 400
            response = await client.post(
                "/v1/local-search/search", json={"query": "apple", "kinds": ["audio"]}
            )
            assert response.status_code == 400
            # Unchanged media reuses its vectors; failed query model still allows filename hits.
            before = list(fake.image_calls)
            app.state.local_search.queue(status["folders"][0]["id"])
            await app.state.local_search.wait_idle()
            assert fake.image_calls == before
            fake.embed = lambda _: (_ for _ in ()).throw(RuntimeError("offline"))
            result = await app.state.local_search.search(
                "apple", 20, None, ["image", "video"]
            )
            assert result["mode"] == "keyword"
            assert all(r["matched"] == ["keyword"] for r in result["results"])
            grouped = await app.state.local_search.search("apple", 20, None, ["image", "video"], 1)
            assert grouped["results"] == result["results"]
            assert {g["kind"] for g in grouped["groups"]} == {"image", "video"}
            assert all(g["total"] == 1 for g in grouped["groups"])
            assert (await app.state.local_search.search("not present", 20, None, group_limit=5))["groups"] == []
            rows = app.state.local_search.store.db.execute(
                "SELECT text,filename FROM chunks_fts WHERE text=''"
            ).fetchall()
            assert len(rows) == 3 and all(
                r["filename"].startswith(("a-apple", "b-apple")) for r in rows
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("count", [1, 19, 20, 29, 30, 120])
def test_semantic_gate_uses_index_background_and_rank_cap(count):
    from rapid_mlx.local_search.search import semantic_ranking, SEMANTIC_FILE_CAP

    rows = {i: {"path": str(i)} for i in range(count)}
    scores = [(i, 0.7) for i in range(count)]
    assert semantic_ranking(scores, rows, 0.6) == list(range(min(count, SEMANTIC_FILE_CAP)))
    assert semantic_ranking(scores, rows, 0.8) == []
    assert semantic_ranking(scores, rows, 0.7) == list(range(min(count, SEMANTIC_FILE_CAP)))
    assert semantic_ranking([(0, 0.0)], rows, 0.0) == []
    assert semantic_ranking([], {}, 0.0) == []


def test_query_outlier_rescue_and_repeated_chunks_do_not_change_null():
    from rapid_mlx.local_search.search import semantic_ranking

    scores = [(0, 0.6)] + [(i, 0.2) for i in range(1, 101)]
    rows = {i: {"path": str(i)} for i in range(101)}
    assert semantic_ranking(scores, rows, 0.9) == [0]
    # Additional frames/chunks of the same file cannot create extra observations.
    repeated = [(101 + i, 0.6) for i in range(1000)]
    rows.update({i: {"path": "0"} for i, _ in repeated})
    assert semantic_ranking([scores[0]] + repeated + scores[1:], rows, 0.9) == [0]
    # A smooth, modest high score is not a significant rare outlier.
    assert semantic_ranking([(i, 0.2 + i / 1000) for i in range(101)], rows, 0.9) == []
    # Summation roundoff must not manufacture variance in a uniform pool.
    assert semantic_ranking([(i, 0.2) for i in range(101)], rows, 0.9) == []


def test_background_matches_cross_file_pairs_and_invalidates(tmp_path):
    search = engine(tmp_path)
    root = folder(tmp_path)
    fid = search.store.add_folder(root)[0]
    for name, kind, vectors in [
        ("a", "text", [[1, 0, 0], [0, 1, 0]]),
        ("b", "text", [[1, 0, 0]]),
        ("c", "image", [[0, 0, 1]]),
    ]:
        path = root / name
        path.write_text("fixture")
        search.store.replace(fid, path, path.stat(), None,
                             [Chunk("", None, None, kind)] * len(vectors), vectors, "indexed")
    rows = search.store.search_rows()
    # No query scoring is needed to initialise the background. The two
    # cross-file text pairs are 1 and 0; all within-file pairs are excluded.
    bars = search.store.semantic_background(rows)
    assert bars == {"text": 1 / 6, "image": 0.0, "video": 0.0}
    assert search.store.semantic_background(rows) is bars
    # Equal-file pooled pairs are 0.5, 0, 0: mean 1/6, capping the
    # narrower text-only estimate. Kind filtering retains the same pool.
    assert search.store.semantic_background(search.store.search_rows(kinds=["text"]))["text"] == 1 / 6
    # Repeating the same frames/chunks leaves the file centroid unchanged.
    path = root / "a"
    search.store.replace(fid, path, path.stat(), None,
                         [Chunk("", None, None)] * 20,
                         [[1, 0, 0], [0, 1, 0]] * 10, "indexed")
    assert search.store.semantic_background(search.store.search_rows())["text"] == 1 / 6
    ids = [r["id"] for r in search.store.db.execute("SELECT id FROM files WHERE path=?", (str(root / "b"),))]
    search.store.remove_file(ids[0])
    assert search.store.semantic_background(search.store.search_rows())["text"] == 0.0
    search.store.close()


def test_semantic_window_reserves_heads_without_reordering_stronger_hits():
    rows = {i: {"path": str(i), "kind": "text" if i < 15 else "image"} for i in range(17)}
    scores = {i: 0.95 - i / 100 for i in range(15)}
    scores.update({15: 0.7, 16: 0.69})
    kwargs = dict(rows=rows, semantic_legs=[list(range(15)), [15, 16], []], semantic_scores=scores)
    original = fuse([], [], **kwargs)
    assert [i for i, _, _ in original[:10]] == list(range(10))
    reserved = fuse([], [], limit=10, **kwargs)
    assert [i for i, _, _ in reserved[:10]] == list(range(8)) + [15, 16]
    assert [score for _, score, _ in reserved[:10]] == sorted([score for _, score, _ in reserved[:10]], reverse=True)
    literal = fuse([], [14], keyword_evidence={14: 2}, limit=10, **kwargs)
    assert literal[0][0] == 14 and {15, 16} <= {i for i, _, _ in literal[:10]}


def test_semantic_gate_float32_boundary_and_one_file():
    from rapid_mlx.local_search.search import semantic_ranking

    rows = {0: {"path": "one"}}
    assert semantic_ranking([(0, float(np.float32(0.9)))], rows, 0.9) == [0]
    assert semantic_ranking([(0, 0.899)], rows, 0.9) == []
    assert semantic_ranking([(0, 0.1)], rows, 0.0) == [0]


@pytest.mark.parametrize("strong_kind", ["image", "text"])
def test_standardized_semantic_order_with_fake_embedder(tmp_path, monkeypatch, strong_kind):
    class QueryEmbedder(FakeEmbedder):
        def embed(self, texts):
            super().embed(texts)
            return [[1.0, 0.0, 0.0] for _ in texts]

    async def scenario():
        fake = QueryEmbedder()
        search = engine(tmp_path, fake)
        root = folder(tmp_path)
        fid = search.store.add_folder(root)[0]
        cosines = {}
        for kind in ("text", "image"):
            # Higher raw text cosine is ordinary when image is the outlier;
            # reverse the evidence without granting media an unconditional boost.
            values = ([.65] + [.2 + n / 1000 for n in range(20)] if kind == "image"
                      else [.8, .79] + [.65 + n / 128 for n in range(20)])
            if strong_kind == "text":
                values = ([.65] + [.55 + n / 250 for n in range(20)] if kind == "image"
                          else [.85] + [.2 + n / 1000 for n in range(20)])
            for n, value in enumerate(values):
                path = root / f"{kind}_{n}.txt"
                path.write_text("fixture")
                cosines[str(path)] = value
                search.store.replace(fid, path, path.stat(), None,
                                     [Chunk("fixture", None, None, kind)],
                                     [[value, np.sqrt(1 - value ** 2), 0]], "indexed")
        # Isolate ordering from admission; all candidates have positive cosine.
        monkeypatch.setattr(search.store, "semantic_background", lambda _: dict.fromkeys(("text", "image", "video"), 0.0))
        hits = (await search.search("meaning", 10, None))["results"]
        assert hits[0]["path"] == str(root / f"{strong_kind}_0.txt")
        assert hits[0]["matched"] == ["semantic"]
        for kind in ("text", "image"):
            values = [cosines[hit["path"]] for hit in hits if hit["kind"] == kind]
            assert values == sorted(values, reverse=True)
        grouped = await search.search("meaning", 10, None, group_limit=5)
        assert grouped["results"] == hits
        assert grouped["groups"][0]["kind"] == strong_kind
        assert {g["kind"]: g["total"] for g in grouped["groups"]} == {"text": 21 if strong_kind == "text" else 22, "image": 21}
        for group in grouped["groups"]:
            assert len(group["results"]) == 5
            values = [cosines[hit["path"]] for hit in group["results"]]
            assert values == sorted(values, reverse=True)
        # Full filename and exact phrase evidence keep the existing positions.
        before = (await search.search("text_0", 10, None))["results"]
        assert before[0]["path"] == str(root / "text_0.txt")
        lexical = await search.search("text_0", 1, None, group_limit=1)
        assert lexical["results"] == before[:1]
        assert lexical["groups"][0]["kind"] == "text"
        assert lexical["groups"][0]["results"][0]["path"] == str(root / "text_0.txt")
        await search.close()
        assert fake.calls

    asyncio.run(scenario())


def test_standardized_near_ties_interleave_without_moving_lexical_hits():
    rows = {i: {"path": str(i), "kind": "text" if i < 6 else "image"} for i in range(9)}
    kwargs = dict(rows=rows, semantic_legs=[[0, 1, 2, 3, 4, 5], [6, 7, 8]],
                  semantic_scores={i: .9 - i / 100 for i in rows},
                  keyword_evidence={0: 3, 1: 2, 2: 1})
    baseline = fuse([], [0, 1, 2], **kwargs)
    ranked = fuse([], [0, 1, 2], semantic_standardized=dict.fromkeys(rows, 4.0), **kwargs)
    assert ranked[:3] == baseline[:3]
    assert [i for i, _, _ in ranked[3:]] == [3, 6, 4, 7, 5, 8]


def test_grouped_rankings_lexical_tiers_and_best_file_head_statistics():
    from rapid_mlx.local_search.search import grouped_rankings

    rows = {i: {"path": str(i), "kind": "text" if i < 4 else "image"} for i in range(6)}
    # The document's lexical chunk differs from its best semantic chunk.
    rows[6] = {"path": "0", "kind": "text"}
    groups = grouped_rankings(
        [0, 1, 2], rows=rows, semantic_legs=[[6, 3, 2, 1], [4, 5]],
        semantic_scores={6: .9, 3: .8, 2: .7, 1: .6, 4: .95, 5: .9},
        semantic_standardized={6: 1, 3: .5, 2: .2, 1: 0, 4: 8, 5: 6},
        keyword_evidence={0: 3, 1: 2, 2: 1},
    )
    assert [kind for kind, _ in groups] == ["text", "image"]
    assert [[i for i, _, _ in hits] for _, hits in groups] == [[0, 1, 2, 3], [4, 5]]
    # With no lexical evidence the stronger standardized head wins; ties have
    # deterministic kind order, without near-tie interleaving between sections.
    for strength, order in [(8, ["image", "text"]), (1, ["text", "image"])]:
        groups = grouped_rankings(
            [], rows=rows, semantic_legs=[[6], [4]],
            semantic_scores={6: .9, 4: .7}, semantic_standardized={6: 1, 4: strength},
            keyword_evidence={},
        )
        assert [kind for kind, _ in groups] == order


def test_group_totals_exclude_missing_files_and_respect_scope(tmp_path):
    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        fid = search.store.add_folder(root)[0]
        for kind in ("text", "image", "video"):
            for n in range(3):
                path = root / f"{kind}-{n}.txt"
                path.write_text("fixture")
                search.store.replace(fid, path, path.stat(), None,
                    [Chunk("fixture" if kind == "text" else "", None, None, kind)], [[1, 0, 0]], "indexed")
        (root / "text-1.txt").unlink()
        search._embed = lambda _: [[1, 0, 0]]
        response = await search.search("meaning", 1, None, group_limit=1)
        assert {g["kind"]: g["total"] for g in response["groups"]} == {"text": 2, "image": 3, "video": 3}
        assert len(response["results"]) == 1
        assert all(len(g["results"]) == 1 for g in response["groups"])
        assert (await search.search("meaning", 1, [], group_limit=50))["groups"] == []
        assert "groups" not in await search.search("meaning", 1, None, ["text", "text"], 50)
        await search.close()

    asyncio.run(scenario())


def test_adaptive_gate_and_keyword_bypass_with_fake_embedder(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "fruit.md").write_text("banana")
        (root / "other.md").write_text("banana")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        search.add_folder(str(root))
        await search.wait_idle()
        # Identical indexed vectors set their background to one. The query's
        # shared bias dimension alone cannot admit unrelated uniform matches.
        assert not (await search.search("unrelated", 20, None))["results"]
        hits = (await search.search("banana", 20, None))["results"]
        assert len(hits) == 2 and all("keyword" in r["matched"] for r in hits)
        # Keyword evidence bypasses rejection even with an orthogonal query.
        search._embed = lambda _: [[0, 1, 0]]
        hits = (await search.search("banana", 20, None))["results"]
        assert len(hits) == 2 and all(r["matched"] == ["keyword"] for r in hits)
        assert fake.calls
        await search.close()

    asyncio.run(scenario())


def test_query_outlier_rescue_with_fake_embedder(tmp_path):
    class OutlierEmbedder(FakeEmbedder):
        def embed(self, texts):
            super().embed(texts)
            vectors = []
            for text in texts:
                if text.startswith("task: search result"):
                    vectors.append([0, 0, 1] if "absent" in text else [1, 0, 0])
                else:
                    cosine = 0.6 if "needle" in text else 0.2
                    vectors.append([cosine, np.sqrt(1 - cosine ** 2), 0])
            return vectors

    async def scenario():
        root = folder(tmp_path)
        for n in range(101):
            (root / f"file_{n}.txt").write_text("needle" if n == 0 else "bulk")
        search = engine(tmp_path, OutlierEmbedder())
        search.add_folder(str(root))
        await search.wait_idle()
        hits = (await search.search("meaning", 10, None))["results"]
        assert len(hits) == 1 and hits[0]["path"] == str(root / "file_0.txt")
        assert hits[0]["matched"] == ["semantic"]
        assert search.store.semantic_background(search.store.search_rows())["text"] > 0.6
        assert not (await search.search("absent", 10, None))["results"]
        await search.close()

    asyncio.run(scenario())


def test_singleton_kinds_interleave_and_keep_best_video_frame(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        search = engine(tmp_path)
        fid = search.store.add_folder(root)[0]
        for kind, value in [("text", 0.99), ("image", 0.76), ("video", 0.50)]:
            path = root / (
                "hit." + {"text": "txt", "image": "png", "video": "mp4"}[kind]
            )
            path.write_bytes(b"fixture")
            chunks = [
                Chunk(
                    "document" if kind == "text" else "",
                    1 if kind == "text" else None,
                    None,
                    kind,
                    0.0 if kind == "video" else None,
                )
            ]
            vectors = [[value, np.sqrt(1 - value * value), 0]]
            if kind == "video":
                chunks.append(Chunk("", None, None, kind, 10.0))
                vectors.append([0.80, np.sqrt(1 - 0.80 * 0.80), 0])
            search.store.replace(
                fid, path, path.stat(), None, chunks, vectors, "indexed"
            )
        search._embed = lambda _: np.array([[1.0, 0.0, 0.0]])
        result = await search.search("unrelated", 20, None)
        assert len(result["results"]) == 3
        assert [r["kind"] for r in result["results"]] == ["text", "image", "video"]
        scores = [r["score"] for r in result["results"]]
        assert scores[0] > scores[1] > scores[2]
        video = next(r for r in result["results"] if r["kind"] == "video")
        assert video["timestamp"] == 10.0
        result = await search.search("hit", 20, None, ["video"])
        assert result["results"][0]["timestamp"] == 10.0
        assert result["results"][0]["matched"] == ["semantic", "keyword"]
        await search.close()

    asyncio.run(scenario())


def test_adaptive_background_independent_of_other_kinds_and_scoped_folders(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        search = engine(tmp_path)
        fid = search.store.add_folder(root)[0]
        # Each kind has its own background; dense stills cannot suppress a video.
        for n, (kind, values) in enumerate(
            [("image", [0.63])] + [("image", [0.7])] * 28
            + [("video", [0.76, 0.69]), ("text", [0.0])]
        ):
            path = root / f"entry_{n}.{kind}"
            path.write_bytes(b"fixture")
            chunks = [Chunk("", None, None, kind, float(j * 5) if kind == "video" else None)
                      for j in range(len(values))]
            vectors = [[v, np.sqrt(1 - v * v), 0] for v in values]
            search.store.replace(fid, path, path.stat(), None, chunks, vectors, "indexed")
        search._embed = lambda _: np.array([[1.0, 0.0, 0.0]])
        for kinds in (None, ["video"], ["image", "video"]):
            results = (await search.search("unrelated", 100, None, kinds))["results"]
            media = [r for r in results if r["kind"] != "text"]
            assert len(media) == 1
            video = next(r for r in media if r["kind"] == "video")
            assert video["timestamp"] == 0.0
        # A file rejected by its index background still qualifies by filename.
        results = (await search.search("entry_0", 100, None, ["image"]))["results"]
        hit = next(r for r in results if r["path"] == str(root / "entry_0.image"))
        assert hit["matched"] == ["keyword"]
        # Folder selection still bounds the pool: an unrelated folder's high
        # cosines must not raise the bar for these selected media.
        other = folder(tmp_path, "other")
        other_fid = search.store.add_folder(other)[0]
        path = other / "other.image"
        path.write_bytes(b"fixture")
        search.store.replace(other_fid, path, path.stat(), None,
                             [Chunk("", None, None, "image")] * 40,
                             [[1, 0, 0]] * 40, "indexed")
        results = (await search.search("unrelated", 100, [fid], ["video"]))["results"]
        assert len(results) == 1 and results[0]["timestamp"] == 0.0
        await search.close()

    asyncio.run(scenario())


def test_media_schema_and_chunking_rebuild_previously_skipped_files(tmp_path):
    from PIL import Image

    async def scenario():
        root = folder(tmp_path)
        path = root / "photo.png"
        Image.new("RGB", (24, 24), "red").save(path)
        search = engine(tmp_path)
        fid = search.store.add_folder(root)[0]
        search.store.replace(
            fid, path, path.stat(), None, [], None, "skipped: unsupported"
        )
        search.store.db.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
        search.store.db.execute(
            "UPDATE meta SET value='2' WHERE key='chunking_version'"
        )
        search.store.db.commit()
        await search.close()
        search = engine(tmp_path)
        assert search.store.files(fid)[str(path)]["status"] == "pending"
        search.start()
        await search.wait_idle()
        assert len(search.store.vectors) == 1
        assert search.store.rows()[next(iter(search.store.vectors))]["kind"] == "image"
        assert search.status()["totals"]["files_by_kind"]["image"] == 1
        await search.close()

    asyncio.run(scenario())


def test_media_cancellation_preserves_per_file_commit_and_resumes(tmp_path):
    from PIL import Image

    async def scenario():
        root = folder(tmp_path)
        Image.new("RGB", (24, 24), "red").save(root / "a.png")
        write_video(root / "b.mp4", colors=(0, 255, 0, 255, 0, 255), fps=2, seconds=30)
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        fid = search.add_folder(str(root))[0]["id"]
        while len(fake.image_calls) < 2:
            await asyncio.sleep(0)
        await search._cancel_worker()
        assert len(search.store.files(fid)) == 1
        assert all(row["kind"] == "image" for row in search.store.rows().values())
        search.queue(fid)
        await search.wait_idle()
        assert search.store.folder(fid)["files_indexed"] == 2
        assert max(fake.image_calls) <= 4
        await search.close()

    asyncio.run(scenario())


def cpu_embedding_module(monkeypatch):
    """Execute engine policy code with a stub array module; never import Metal."""
    import importlib.util
    from types import ModuleType

    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    spec = importlib.util.spec_from_file_location(
        "rapid_mlx.embedding", Path(__file__).parents[1] / "rapid_mlx" / "embedding.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "rapid_mlx.embedding", module)
    spec.loader.exec_module(module)
    return module


def test_vision_flag_default_and_text_media_rejection_on_http(monkeypatch):
    from types import SimpleNamespace

    from rapid_mlx.config import get_config
    from rapid_mlx.routes import embeddings

    module = cpu_embedding_module(monkeypatch)
    engine = module.EmbeddingEngine("mlx-community/embeddinggemma-2-4bit")
    assert engine.vision is False
    with pytest.raises(module.EmbeddingUnsupportedMediaError, match="vision=True"):
        engine.embed_images([])
    ids = dict(
        image_token_id=258880,
        audio_token_id=258881,
        video_token_id=258884,
        boi_token_id=255999,
        eoi_token_id=258882,
        boa_token_id=256000,
        eoa_token_index=258883,
    )
    engine._model = SimpleNamespace(
        model_type="embedding_gemma2", config=SimpleNamespace(**ids)
    )
    engine.effective_max_length = 8192
    engine.count_tokens = lambda _: 1
    engine._tokenizer = lambda texts, **kwargs: {"input_ids": [[2, 258880]]}
    # Even a vision-enabled local engine cannot bypass the text API guard.
    cfg = get_config()
    monkeypatch.setattr(cfg, "embedding_model_locked", engine.model_name)
    monkeypatch.setattr(cfg, "embedding_engine", engine)
    monkeypatch.setattr(cfg, "api_key", None)
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(load_embedding_model=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.service.helpers",
        SimpleNamespace(_resolve_request_alias_or_default=lambda name, locked: locked),
    )
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(embeddings.router)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            for vision in (False, True):
                engine.vision = vision
                for payload in ("<|image|>", [258880], [[258881]]):
                    result = await client.post(
                        "/v1/embeddings", json={"model": "default", "input": payload}
                    )
                    assert result.status_code == 400
                    assert (
                        result.json()["error"]["code"]
                        == "unsupported_embedding_modality"
                    )
                result = await client.post(
                    "/v1/embeddings",
                    json={"model": "default", "input": {"image": "file.png"}},
                )
                assert result.status_code == 400

    asyncio.run(scenario())


def test_vision_image_engine_uses_pilot_sequence_and_mixed_sizes(monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    module = cpu_embedding_module(monkeypatch)
    module.mx.array = np.asarray
    module.mx.float32 = np.float32
    cleared = []
    module.mx.clear_cache = lambda: cleared.append(True)
    engine = module.EmbeddingEngine("mlx-community/embeddinggemma-2-4bit", vision=True)
    config = SimpleNamespace(boi_token_id=10, image_token_id=11, eoi_token_id=12)
    captured = []

    class Model:
        vision_enabled = True

        def __call__(self, ids, **kwargs):
            captured.append((ids, kwargs))
            return SimpleNamespace(text_embeds=np.array([[1.0, 0.0], [0.0, 1.0]]))

    engine._model = Model()
    engine._model.config = config
    engine.effective_max_length = 8192

    class Tokenizer:
        def decode(self, ids):
            return str(ids[0]) + " "

        def __call__(self, pieces, **kwargs):
            assert pieces == ["10 11 11 12 ", "10 11 11 11 12 "]
            return {
                "input_ids": np.array(
                    [[2, 10, 11, 11, 12, 3, 0], [2, 10, 11, 11, 11, 12, 3]]
                ),
                "attention_mask": np.array(
                    [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]
                ),
            }

    engine._tokenizer = Tokenizer()

    def prepare(images):
        assert [im.size for im in images] == [(8, 8), (16, 8)]
        return {"pixel_values": [np.zeros((3, 8, 8)), np.zeros((3, 8, 16))]}, [2, 3]

    engine._image_processor = prepare
    result = engine.embed_images([Image.new("RGB", (8, 8)), Image.new("RGB", (16, 8))])
    assert result == [[1.0, 0.0], [0.0, 1.0]] and len(cleared) == 1
    assert [p.shape for p in captured[0][1]["pixel_values"]] == [(3, 8, 8), (3, 8, 16)]
    assert captured[0][1]["attention_mask"][0, -1] == 0
    assert engine.embed_images([]) == []
    engine.effective_max_length = 2
    with pytest.raises(module.EmbeddingInputTooLongError):
        engine.embed_images([Image.new("RGB", (8, 8)), Image.new("RGB", (16, 8))])
    assert len(captured) == 1


def test_status_vision_available_checkpoint_without_loading(tmp_path):
    import json

    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text(
        json.dumps(
            {"model_type": "embedding_gemma2", "vision_config": {"hidden_size": 8}}
        )
    )
    search = LocalSearch(tmp_path / "index", resolve_model=lambda _: str(model))
    assert search.status()["model"]["vision"] is True
    assert search.engine is None
    config.write_text(json.dumps({"model_type": "embedding_gemma2"}))
    assert search.status()["model"]["vision"] is False
    asyncio.run(search.close())


def test_engine_native_loader_vision_opt_in_only(monkeypatch):
    from types import SimpleNamespace

    from rapid_mlx import model_downloads

    module = cpu_embedding_module(monkeypatch)
    monkeypatch.setattr(model_downloads, "require_local", lambda _: None)
    calls = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(model_type="embedding_gemma2"), object()

    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.models.embedding_gemma2.loader",
        SimpleNamespace(load=load),
    )
    name = "mlx-community/embeddinggemma-2-4bit"
    module.EmbeddingEngine(name).load()
    module.EmbeddingEngine(name, vision=True).load()
    assert calls == [(name, {}), (name, {"vision": True})]


def test_media_without_model_filename_index_resumes(tmp_path):
    from PIL import Image

    async def scenario():
        root = folder(tmp_path)
        Image.new("RGB", (24, 24), "red").save(root / "sunset.png")
        write_video(root / "sunset.mp4")
        search = engine(tmp_path)
        search.resolve_model = lambda _: None
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "waiting_for_model"
        assert search.status()["totals"]["files_by_kind"] == {
            "text": 0,
            "image": 1,
            "video": 1,
        }
        result = await search.search("sunset", 20, None)
        assert len(result["results"]) == 2 and result["mode"] == "keyword"
        assert not search.store.vectors
        search.resolve_model = lambda _: "local-checkpoint"
        search.status()
        await search.wait_idle()
        assert len(search.store.vectors) == 3
        assert search.store.folder(fid)["state"] == "idle"
        await search.close()

    asyncio.run(scenario())


def test_unexpected_decoder_failure_does_not_abort_folder(tmp_path, monkeypatch):
    from PIL import Image

    async def scenario():
        root = folder(tmp_path)
        (root / "good.txt").write_text("apple")
        (root / "bad.tiff").write_bytes(b"corrupt")
        monkeypatch.setattr(
            Image,
            "open",
            lambda _: (_ for _ in ()).throw(SyntaxError("invalid TIFF header")),
        )
        search = engine(tmp_path)
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "idle"
        assert search.store.folder(fid)["files_skipped"] == 1
        assert search.store.folder(fid)["files_indexed"] == 1
        await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("identifier", ["DEC-0054", "src/lib/i18n.ts", "RAPID_MLX_HOME", "foo.bar"])
@pytest.mark.parametrize("in_filename", [False, True])
def test_keyword_compounds_are_literal(tmp_path, identifier, in_filename):
    from rapid_mlx.local_search.search import query_terms

    assert query_terms(f"Find ({identifier}). 中文") == ["Find", identifier, "中文"]
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    split = " unrelated ".join(__import__("re").split(r"[-/._]", identifier))
    # Paths containing slashes must be tested in content, since only basenames
    # participate in filename keyword lookup.
    name = identifier if in_filename and "/" not in identifier else "exact"
    entries = [(name + ".txt", "needle " + ("" if name == identifier else identifier)),
               ("split.txt", "needle " + split),
               ("separator.txt", "needle " + identifier.replace(".", " ").replace("-", " ").replace("_", " ").replace("/", " "))]
    for name, text in entries:
        path = root / name
        path.write_text(text)
        store.replace(fid, path, path.stat(), None, [Chunk(text, 1, None)], None, "keyword")
    rows = store.rows()
    expected = [i for i, row in rows.items() if Path(row["path"]).name == entries[0][0]]
    assert store.keyword(query_terms(identifier), set(rows), query=identifier) == expected
    # A common identifier must not be discarded for another distinctive term.
    assert store.keyword(query_terms("needle " + identifier), set(rows),
                         distinctive=["needle"]) == expected
    store.close()


def test_video_dedupe_accumulates_slow_drift(tmp_path, monkeypatch):
    import cv2
    from rapid_mlx.local_search.extraction import media_frames

    root = folder(tmp_path)
    path = root / "drift.mp4"
    path.write_bytes(b"fixture")

    class Capture:
        value = -1
        released = False

        def isOpened(self):
            return True

        def get(self, key):
            return 1 if key == cv2.CAP_PROP_FPS else 26

        def set(self, key, value):
            self.value += 1
            return True

        def read(self):
            return True, np.full((32, 32, 3), self.value, np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    frames = list(media_frames(path, root))
    assert [f.timestamp for f in frames] == [0.0, 10.0, 20.0]
    assert capture.released


def test_whole_mac_validation_and_overlap_without_walking(tmp_path):
    assert validate_folder("/") == Path("/")
    for first, second in ((Path("/"), tmp_path), (tmp_path, Path("/"))):
        store = Store(tmp_path / ("root-first.sqlite" if first == Path("/") else "root-last.sqlite"), "fake")
        try:
            fid, _ = store.add_folder(first)
            assert store.add_folder(first) == (fid, False)
            covering_id, created = store.add_folder(second)
            assert created == (second == Path("/"))
            assert (covering_id == fid) == (first == Path("/"))
            assert len(store.folders()) == 1
            assert store.folders()[0]["path"] == "/"
        finally:
            store.close()


@pytest.mark.parametrize("chosen", [
    "whole", "home", "documents", "volume", "library", "icloud", "other-library",
    "system", "scratch",
])
def test_absolute_os_and_all_user_library_skips(tmp_path, monkeypatch, chosen):
    from rapid_mlx.local_search import extraction
    os_names = ("System", "Library", "usr", "bin", "sbin", "private", "dev", "cores", "opt", "Network", "Volumes")
    prefixes = tuple(tmp_path / n for n in os_names)
    monkeypatch.setattr(extraction, "SKIP_PREFIXES", prefixes)
    monkeypatch.setattr(extraction, "USERS_DIRECTORY", tmp_path / "Users")
    home = tmp_path / "Users" / "alice"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for prefix in prefixes:
        prefix.mkdir()
        (prefix / "excluded.txt").write_text("secret")
    for user in ("alice", "bob"):
        library = tmp_path / "Users" / user / "Library" / "nested"
        library.mkdir(parents=True)
        (library / "excluded.txt").write_text("secret")
    documents = home / "Documents"
    documents.mkdir()
    (documents / "good.txt").write_text("apple")
    volume = tmp_path / "Volumes" / "External"
    volume.mkdir()
    (volume / "good.txt").write_text("banana")
    icloud = home / "Library" / "Mobile Documents"
    icloud.mkdir()
    (icloud / "good.txt").write_text("apple")
    scratch = tmp_path / "private" / "tmp"
    scratch.mkdir()
    (scratch / "good.txt").write_text("banana")
    # The temp tree stands in for "/"; never walk the real filesystem root.
    root = {
        "whole": tmp_path, "home": home, "documents": documents, "volume": volume,
        "library": home / "Library", "icloud": icloud,
        "other-library": tmp_path / "Users" / "bob" / "Library",
        "system": tmp_path / "System", "scratch": scratch,
    }[chosen]
    expected = {
        "whole": [documents / "good.txt"], "home": [documents / "good.txt"],
        "documents": [documents / "good.txt"], "volume": [volume / "good.txt"],
        "library": [home / "Library" / "nested" / "excluded.txt", icloud / "good.txt"],
        "icloud": [icloud / "good.txt"],
        "other-library": [root / "nested" / "excluded.txt"],
        "system": [root / "excluded.txt"], "scratch": [scratch / "good.txt"],
    }[chosen]
    assert validate_folder(str(root)) == root
    files, skipped = scan(root)
    assert set(files) == set(expected)
    assert skipped == {"whole": len(prefixes) + 2, "home": 1}.get(chosen, 0)
    for selected in (*prefixes, home / "Library", tmp_path / "Users" / "bob" / "Library"):
        assert validate_folder(str(selected)) == selected
        selected_files, selected_skipped = scan(selected)
        assert selected_files
        assert selected_skipped == 0
    assert validate_folder(str(volume)) == volume
    # Boundary matching: similarly named user directories are allowed.
    ordinary = tmp_path / "System Notes"
    ordinary.mkdir()
    assert validate_folder(str(ordinary)) == ordinary

    async def scenario():
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        try:
            search.add_folder(str(root))
            await search.wait_idle()
            assert {row["path"] for row in search.store.rows().values()} == {str(path) for path in files}
            assert len(fake.calls) == len(expected)
            assert search.status()["folders"][0]["files_skipped"] == skipped
        finally:
            await search.close()

    asyncio.run(scenario())


def test_unreadable_directory_is_skipped_and_indexing_continues(tmp_path, monkeypatch):
    root = folder(tmp_path)
    denied = root / "denied"
    denied.mkdir()
    (root / "good.txt").write_text("apple")
    original = os.scandir

    def scandir(path):
        if Path(path) == denied:
            raise PermissionError("TCC denied")
        return original(path)

    monkeypatch.setattr(os, "scandir", scandir)

    async def scenario():
        search = engine(tmp_path)
        try:
            search.add_folder(str(root))
            await search.wait_idle()
            status = search.status()
            assert status["folders"][0]["state"] == "idle"
            assert status["folders"][0]["files_skipped"] == 1
            assert status["totals"]["files"] == 1
        finally:
            await search.close()

    asyncio.run(scenario())


def test_files_searchable_before_discovery_finishes(tmp_path, monkeypatch):
    from rapid_mlx.local_search import service
    root = folder(tmp_path)
    first = root / "first.txt"
    first.write_text("apple")
    nested = root / "nested"
    nested.mkdir()
    (nested / "later.txt").write_text("banana")

    finished = False
    original_scan = service.scan_batches

    def scan_in_progress(root):
        nonlocal finished
        yield from original_scan(root)
        finished = True

    monkeypatch.setattr(service, "scan_batches", scan_in_progress)

    async def scenario():
        search = engine(tmp_path)
        continued = asyncio.Event()
        release = asyncio.Event()

        async def pause():
            assert search.progress["active"]
            assert search.progress["files_total"] == 0
            assert not finished
            assert search.progress["files_done"] == 1
            assert any(r["path"] == str(first) for r in (await search.search("apple", 20, None))["results"])
            continued.set()
            await release.wait()

        # Pause indexing at the first file's cooperative yield, before requesting
        # the next discovery batch, so this test never depends on timing.
        paused = False
        async_sleep = service.asyncio.sleep

        async def sleep(seconds):
            nonlocal paused
            if not paused and search.progress["files_done"] == 1:
                paused = True
                await pause()
            await async_sleep(seconds)

        monkeypatch.setattr(service.asyncio, "sleep", sleep)
        try:
            search.add_folder(str(root))
            await asyncio.wait_for(continued.wait(), 5)
            release.set()
            await search.wait_idle()
            assert search.status()["totals"]["files"] == 2
        finally:
            release.set()
            await search.close()

    asyncio.run(scenario())


def test_whole_mac_api_acceptance_does_not_walk_root(tmp_path, monkeypatch):
    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, app, _):
            monkeypatch.setattr(app.state.local_search, "queue", lambda _: None)
            response = await client.post("/v1/local-search/folders", json={"path": "/"})
            assert response.status_code == 201
            assert response.json()["path"] == "/"
            again = await client.post("/v1/local-search/folders", json={"path": "/"})
            assert again.status_code == 200
            assert again.json()["id"] == response.json()["id"]

    asyncio.run(scenario())


def test_full_phrase_beats_two_weak_votes_and_keeps_best_chunk():
    from rapid_mlx.local_search.search import lexical_tier

    rows = {
        1: {"path": "near.md", "kind": "text"},
        2: {"path": "target.md", "kind": "text"},
        3: {"path": "target.md", "kind": "text"},
        4: {"path": "frame.mp4", "kind": "video"},
    }
    query = "orbital resonance certificate"
    assert lexical_tier(query, query.split(), "orbital\nresonance certificate", "target.md") == 2
    assert lexical_tier(query, query.split(), "orbital certificate resonance", "near.md") == 0
    # A keyword-only literal chunk defeats a semantic+partial keyword neighbour,
    # and repeated weak chunks from that same file cannot replace its snippet.
    ranked = fuse([], [2, 1, 3], rows=rows, semantic_legs=[[1], [], [4]],
                  semantic_scores={1: .9, 4: .8}, keyword_evidence={2: 2, 1: 0, 3: 0})
    assert ranked[0][0] == 2
    assert ranked[0][2] == ["keyword"]
    assert len(ranked) == 3
    assert ranked[0][1] > ranked[1][1]


def test_literal_lookup_preserves_keyword_order_among_full_phrases():
    rows = {1: {"path": "first.md"}, 2: {"path": "second.md"}}
    # Extra semantic support cannot move a weaker lexical match above the
    # strongest phrase hit; no corpus-specific multiplier is involved.
    ranked = fuse([2], [1, 2], rows=rows, keyword_evidence={1: 2, 2: 2})
    assert [i for i, _, _ in ranked] == [1, 2]
    assert ranked[0][1] > ranked[1][1]


def test_content_copies_are_adjacent_without_hiding_paths():
    rows = {
        1: {"path": "original.md", "kind": "text", "content_hash": "same"},
        2: {"path": "unrelated.md", "kind": "text", "content_hash": "other"},
        3: {"path": "copy.md", "kind": "text", "content_hash": "same"},
    }
    ranked = fuse([1, 2, 3], [], rows=rows)
    assert [i for i, _, _ in ranked] == [1, 3, 2]
    assert len({rows[i]["path"] for i, _, _ in ranked}) == 3
    assert ranked[0][1] == ranked[1][1] > ranked[2][1]
    # A missing hash must never group unrelated files.
    rows[1]["content_hash"] = rows[3]["content_hash"] = None
    assert [i for i, _, _ in fuse([1, 2, 3], [], rows=rows)] == [1, 2, 3]


def test_distinctive_frequency_is_invariant_to_keyword_only_tails(tmp_path):
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    for name, text in [("a.txt", "needle common"), ("b.txt", "common"), ("data.txt", "data")]:
        path = root / name
        path.write_text(text)
        store.replace(fid, path, path.stat(), None, [Chunk(text, 1, None)], None, "keyword")
    before = store.distinctive_terms(["needle", "common"])
    data = root / "data.txt"
    # A generated file may dwarf the rest of the corpus, without making words
    # occurring in several other files 'distinctive'.
    store.replace(fid, data, data.stat(), None, [Chunk("data", n, None) for n in range(200)], None, "keyword")
    assert store.distinctive_terms(["needle", "common"]) == before == ["needle"]
    # Replacing a file must invalidate frequency results, including removals.
    target = root / "b.txt"
    store.replace(fid, target, target.stat(), None, [Chunk("needle common", 1, None)], None, "keyword")
    assert store.distinctive_terms(["needle"]) == []
    store.close()


def test_common_filename_terms_still_match_and_exact_basename_wins(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        search = LocalSearch(tmp_path / "index", resolve_model=lambda _: None)
        fid = search.store.add_folder(root)[0]
        for name, text in [("orbit-map.txt", "unrelated"),
                           ("b.txt", "orbit and map"), ("c.txt", "orbit map")]:
            path = root / name
            path.write_text(text)
            search.store.replace(fid, path, path.stat(), None, [Chunk(text, 1, None)], None, "keyword")
        assert search.store.distinctive_terms(["orbit", "map"]) == []
        hits = (await search.search("orbit map", 10, None))["results"]
        assert Path(hits[0]["path"]).name == "orbit-map.txt"
        assert hits[0]["matched"] == ["keyword"]
        assert all(0 <= hit["score"] <= 1 for hit in hits)
        await search.close()
    asyncio.run(scenario())


def test_metadata_search_cache_tracks_replacements_deletion_scope_and_touch(tmp_path):
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    path = root / "a.txt"
    path.write_text("first")
    store.replace(fid, path, path.stat(), "old", [Chunk("first", 1, None)], None, "keyword")
    rows = store.search_rows()
    first_id = next(iter(rows))
    assert "text" not in rows[first_id]
    assert store.chunk_text(first_id) == "first"
    assert store.search_rows([], None) == {}
    assert store.search_rows(None, ["image"]) == {}
    assert store.search_rows(None, ["text"])[first_id]["content_hash"] == "old"
    path.write_text("replacement")
    store.replace(fid, path, path.stat(), "new", [Chunk("replacement", 1, None)], None, "keyword")
    rows = store.search_rows()
    replacement_id = next(iter(rows))
    assert first_id not in rows
    assert rows[replacement_id]["content_hash"] == "new"
    assert store.chunk_text(replacement_id) == "replacement"
    path.write_text("longer replacement")
    store.touch(rows[replacement_id]["file_id"], path.stat())
    assert store.search_rows()[replacement_id]["size"] == path.stat().st_size
    store.remove_file(rows[replacement_id]["file_id"])
    assert store.search_rows() == {}
    store.close()


def test_video_keyword_evidence_survives_best_frame_selection():
    rows = {1: {"path": "unique.mp4", "kind": "video"},
            2: {"path": "unique.mp4", "kind": "video"},
            3: {"path": "other.md", "kind": "text"}}
    ranked = fuse([], [1], rows=rows, semantic_legs=[[3], [], [2]],
                  semantic_scores={2: .7, 3: .9}, keyword_evidence={1: 3})
    assert ranked[0][0] == 2
    assert ranked[0][2] == ["semantic", "keyword"]


def test_semantic_collapse_keeps_first_chunk_and_counts_one_vote():
    rows = {1: {"path": "frames.mp4", "kind": "video"},
            2: {"path": "other.png", "kind": "image"},
            3: {"path": "frames.mp4", "kind": "video"}}
    ranked = fuse([1, 2, 3], [], rows=rows)
    assert [i for i, _, _ in ranked] == [1, 2]
    assert ranked[0][1] == fuse([1], [], rows=rows)[0][1]


def office_zip(parts):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in parts.items():
            archive.writestr(name, text)
    return data.getvalue()


def office_fixture(suffix):
    word = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    drawing = "http://schemas.openxmlformats.org/drawingml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package = "http://schemas.openxmlformats.org/package/2006/relationships"
    sheet = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

    def relationships(entries):
        return f'<Relationships xmlns="{package}">' + "".join(
            f'<Relationship Id="{rid}" Type="{rel}/{kind}" Target="{target}"/>'
            for rid, kind, target in entries
        ) + '</Relationships>'

    def paragraph(ns, label):
        return f'<a:p xmlns:a="{ns}"><a:r><a:t>{label}</a:t></a:r></a:p>'

    if suffix == ".docx":
        parts = {"word/document.xml": f'<w:document xmlns:w="{word}"><w:body>' +
                 paragraph(word, "apple body") + '<w:tbl><w:tr><w:tc>' +
                 paragraph(word, "banana table") + '</w:tc></w:tr></w:tbl></w:body></w:document>'}
        for name in ("header1", "footer1", "footnotes", "endnotes", "comments"):
            parts[f"word/{name}.xml"] = paragraph(word, name + " legal clause")
        return office_zip(parts), [None]
    if suffix == ".pptx":
        presentation = "http://schemas.openxmlformats.org/presentationml/2006/main"
        # Relationship order, not filenames, determines the user-visible slide.
        parts = {
            "ppt/presentation.xml": f'<p:presentation xmlns:p="{presentation}" xmlns:r="{rel}"><p:sldIdLst><p:sldId r:id="s2"/><p:sldId r:id="s1"/></p:sldIdLst></p:presentation>',
            "ppt/_rels/presentation.xml.rels": relationships([
                ("s1", "slide", "slides/slide1.xml"), ("s2", "slide", "slides/slide2.xml")]),
            "ppt/slides/slide1.xml": paragraph(drawing, "banana second slide"),
            "ppt/slides/slide2.xml": paragraph(drawing, "apple first slide"),
            "ppt/slides/_rels/slide2.xml.rels": relationships([
                ("note", "notesSlide", "../notesSlides/notesSlide7.xml")]),
            "ppt/notesSlides/notesSlide7.xml": paragraph(drawing, "speaker patent notes"),
        }
        return office_zip(parts), [1, 2]
    parts = {
        "xl/workbook.xml": f'<workbook xmlns="{sheet}" xmlns:r="{rel}"><sheets><sheet name="First" r:id="s2"/><sheet name="Second" r:id="s1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": relationships([
            ("s1", "worksheet", "worksheets/sheet1.xml"), ("s2", "worksheet", "worksheets/sheet2.xml")]),
        "xl/sharedStrings.xml": f'<sst xmlns="{sheet}"><si><r><t>apple </t></r><r><t>shared string</t></r></si></sst>',
        "xl/worksheets/sheet2.xml": f'<worksheet xmlns="{sheet}"><sheetData><row><c t="s"><v>0</v></c><c t="inlineStr"><is><t>banana inline string</t></is></c><c><v>42</v></c></row></sheetData></worksheet>',
        "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{sheet}"><sheetData><row><c t="inlineStr"><is><t>second sheet</t></is></c></row></sheetData></worksheet>',
    }
    return office_zip(parts), [1, 2]


@pytest.mark.parametrize("suffix", [".docx", ".pptx", ".xlsx"])
def test_office_text_and_persisted_locators(tmp_path, suffix):
    data, pages = office_fixture(suffix)
    units = extract(Path("file" + suffix), data)
    assert [u.page for u in units] == pages
    if suffix == ".docx":
        assert all(label in units[0].text for label in (
            "apple body", "banana table", "header1", "footer1", "footnotes", "endnotes", "comments"))
    elif suffix == ".pptx":
        assert "apple first slide" in units[0].text and "speaker patent notes" in units[0].text
        assert "banana second slide" in units[1].text
    else:
        assert units[0].text == "apple shared string\nbanana inline string\n42"
        assert units[1].text == "second sheet"

    async def scenario():
        import json
        root = folder(tmp_path)
        (root / ("file" + suffix)).write_bytes(data)
        search = engine(tmp_path)
        try:
            search.add_folder(str(root))
            await search.wait_idle()
            rows = list(search.store.rows().values())
            assert [json.loads(row["locator"])["page"] for row in rows] == pages
            assert all(row["id"] in search.store.vectors for row in rows)
            result = (await search.search("apple", 20, None))["results"][0]
            assert result["page"] == pages[0]
        finally:
            await search.close()
    asyncio.run(scenario())


def pdf_fixture(*, encrypted=False, mixed=False, near_empty=False):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(300, 300)
    if near_empty:
        # Reuse valid resources but retain only a page-number glyph.
        page = writer.add_page(PdfReader(io.BytesIO(small_pdf())).pages[0])
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 30 30 Td (1) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    if mixed:
        writer.add_page(PdfReader(io.BytesIO(small_pdf())).pages[0])
    if encrypted:
        writer.encrypt("password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("kwargs,reason", [
    ({}, "Scanned PDF without text"),
    ({"near_empty": True}, "Scanned PDF without text"),
    ({"encrypted": True}, "Password-protected"),
])
def test_pdf_skip_classification(kwargs, reason):
    with pytest.raises(SkippedError, match=reason):
        extract(Path("file.pdf"), pdf_fixture(**kwargs))


def test_mixed_pdf_indexes_text_page(tmp_path):
    data = pdf_fixture(mixed=True)
    assert extract(Path("mixed.pdf"), data) == [Unit("First", 2)]

    async def scenario():
        root = folder(tmp_path)
        (root / "mixed.pdf").write_bytes(data)
        search = engine(tmp_path)
        try:
            search.add_folder(str(root))
            await search.wait_idle()
            assert (await search.search("First", 20, None))["results"][0]["page"] == 2
            assert search.status()["folders"][0]["skip_reasons"] == {}
        finally:
            await search.close()
    asyncio.run(scenario())


def test_skip_reasons_incremental_restart_and_changes(tmp_path, monkeypatch):
    from rapid_mlx.local_search import service
    expected = {"Unsupported file type": 6, "Too large": 1, "No readable text": 1,
                "Scanned PDF without text": 1, "Password-protected": 1,
                "Could not be read": 2, "Private or system file": 1,
                "Audio is not supported yet": 1}
    root = folder(tmp_path)
    for suffix in (".doc", ".xls", ".ppt", ".rtf", ".pages", ".unknown"):
        (root / ("old" + suffix)).write_bytes(b"legacy")
    (root / "empty.txt").write_text(" \n\t")
    (root / "bad.docx").write_bytes(b"corrupt")
    (root / "scanned.pdf").write_bytes(pdf_fixture())
    (root / "encrypted.pdf").write_bytes(pdf_fixture(encrypted=True))
    (root / "sound.mp3").write_bytes(b"audio")
    (root / ".env").write_text("secret")
    (root / "big.txt").write_bytes(b"a" * 100)
    (root / "denied.txt").write_text("unreadable")
    (root / "good.txt").write_text("apple")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "ignored.bin").write_bytes(b"excluded subtree")
    original_read = service.read_safe

    def limited_read(path, root):
        if path.name == "denied.txt":
            raise PermissionError("denied")
        return original_read(path, root, 80)

    monkeypatch.setattr(service, "read_safe", limited_read)
    # PDF bytes exceed the test-only text cap; let those use the normal cap.
    def read(path, root):
        return original_read(path, root) if path.suffix == ".pdf" else limited_read(path, root)
    monkeypatch.setattr(service, "read_safe", read)

    async def scenario():
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        assert search.status()["folders"][0]["skip_reasons"] == expected
        assert search.store.folder(fid)["files_skipped"] == sum(expected.values()) + 1
        await search.close()
        search = engine(tmp_path, fake)
        try:
            assert search.status()["folders"][0]["skip_reasons"] == expected
            def no_read(*args):
                pytest.fail("unchanged skipped files must not be read or extracted")
            monkeypatch.setattr(service, "read_safe", no_read)
            monkeypatch.setattr(service, "extract", no_read)
            search.queue(fid)
            await search.wait_idle()
            assert search.status()["folders"][0]["skip_reasons"] == expected
            assert len(fake.calls) == 1
            monkeypatch.setattr(service, "read_safe", read)
            monkeypatch.setattr(service, "extract", extract)
            (root / "empty.txt").write_text("banana now readable")
            (root / "scanned.pdf").unlink()
            search.queue(fid)
            await search.wait_idle()
            changed = dict(expected)
            del changed["No readable text"], changed["Scanned PDF without text"]
            assert search.status()["folders"][0]["skip_reasons"] == changed
        finally:
            await search.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("suffix", [".docx", ".pptx", ".xlsx"])
def test_office_zip_bomb_bound(suffix, monkeypatch):
    # Several individually small members exceed the aggregate expansion cap.
    data = office_zip({f"part{i}.xml": "x" * 800 for i in range(4)})
    reads = []
    original = zipfile.ZipFile.open
    def opened(self, *args, **kwargs):
        reads.append(args[0])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "open", opened)
    with pytest.raises(SkippedError, match="size cap"):
        extract(Path("bomb" + suffix), data, max_bytes=2000)
    assert not reads  # Refuse before any inflation/XML parsing.


def test_xlsx_cell_cap_and_shared_string_expansion(monkeypatch):
    from rapid_mlx.local_search import extraction
    sheet = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    data, _ = office_fixture(".xlsx")
    monkeypatch.setattr(extraction, "XLSX_MAX_CELLS_PER_SHEET", 2)
    assert extract(Path("a.xlsx"), data)[0].text == "apple shared string\nbanana inline string"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts["xl/sharedStrings.xml"] = f'<sst xmlns="{sheet}"><si><t>{"a" * 1000}</t></si></sst>'
    parts["xl/worksheets/sheet2.xml"] = f'<worksheet xmlns="{sheet}">' + '<c t="s"><v>0</v></c>' * 20 + '</worksheet>'
    monkeypatch.setattr(extraction, "XLSX_MAX_CELLS_PER_SHEET", 100)
    with pytest.raises(SkippedError, match="size cap"):
        extract(Path("a.xlsx"), office_zip(parts), max_bytes=6000)


def test_existing_index_upgrade_without_full_rebuild(tmp_path):
    from rapid_mlx.local_search import MODEL_ID
    root = folder(tmp_path)
    store = Store(tmp_path / "index" / "index.sqlite3", MODEL_ID)
    fid = store.add_folder(root)[0]
    good = root / "good.txt"
    good.write_text("apple unchanged")
    store.replace(fid, good, good.stat(), None, [Chunk("apple unchanged", 1, None)],
                  [[1., 0., 0.]], "indexed")
    good_id = next(iter(store.vectors))
    for suffix in (".docx", ".pptx", ".xlsx"):
        path = root / ("office" + suffix)
        path.write_bytes(office_fixture(suffix)[0])
        store.replace(fid, path, path.stat(), None, [], None,
                      "indexed" if suffix == ".docx" else "skipped: unsupported")
    blank = root / "scan.pdf"
    blank.write_bytes(pdf_fixture())
    store.replace(fid, blank, blank.stat(), None, [], None, "indexed")
    unsupported = root / "legacy.doc"
    unsupported.write_bytes(b"old")
    store.replace(fid, unsupported, unsupported.stat(), None, [], None, "skipped: unsupported")
    # Simulate a database written before the additive extraction upgrade.
    with store.db:
        store.db.execute("DELETE FROM meta WHERE key='extraction_version'")
    store.close()

    async def scenario():
        search = engine(tmp_path)
        try:
            assert search.store.folder(fid)["skip_reasons"] == {"Unsupported file type": 1}
            assert good_id in search.store.vectors
            search.start()
            await search.wait_idle()
            assert search.store.folder(fid)["skip_reasons"] == {
                "Unsupported file type": 1, "Scanned PDF without text": 1}
            assert search.store.folder(fid)["files_indexed"] == 4
            assert good_id in search.store.vectors
            assert search.store.chunk_text(good_id) == "apple unchanged"
            assert not any("unchanged" in text for batch in search.engine.calls for text in batch)
        finally:
            await search.close()
    asyncio.run(scenario())


def test_xlsx_reuses_per_file_embedding_budget(tmp_path):
    sheet = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    data, _ = office_fixture(".xlsx")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts["xl/worksheets/sheet2.xml"] = f'<worksheet xmlns="{sheet}">' + (
        '<c t="inlineStr"><is><t>apple ' + 'legal clause ' * 20 + '</t></is></c>') * 400 + '</worksheet>'

    async def scenario():
        root = folder(tmp_path)
        (root / "large.xlsx").write_bytes(office_zip(parts))
        search = engine(tmp_path)
        try:
            search.add_folder(str(root))
            await search.wait_idle()
            assert len(search.store.rows()) > MAX_EMBEDDED_CHUNKS_PER_FILE
            assert len(search.store.vectors) == MAX_EMBEDDED_CHUNKS_PER_FILE
        finally:
            await search.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("text", [
    "a4" * 150,
    ("QWxhZGRpbjpvcGVuIHNlc2FtZQ" * 3 + "\n") * 5,
    "QWxhZGRpbjpvcGVuIHNlc2FtZQ" * 15,
    '{"payload":[' + ','.join('"value"' for _ in range(60)) + ']}' ,
    " ".join(chr(97+i//26)+chr(97+i%26) for i in range(50)),
])
def test_low_information_shapes(text):
    from rapid_mlx.local_search.eligibility import low_information
    assert low_information(text)


@pytest.mark.parametrize("text", [
    "A short note",
    "apple\n" * 50,
    "This report explains how to benchmark throughput and compare the results. " * 10,
    "如何运行基准测试并比较每秒处理的令牌数量。" * 30,
    "ベンチマークの結果を比較してグラフを表示します。" * 30,
    "벤치마크 결과와 초당 토큰 처리량을 비교합니다. " * 30,
    "def decode(tokens):\n    return [token.value for token in tokens]\n" * 10,
    "const chart=results.map(row=>({label:row.name,value:row.tokens/row.seconds}));" * 10,
    "The values are " + "0123456789abcdef" * 5 + "; keep these notes for the run. " * 20,
])
def test_language_and_source_remain_semantic(text):
    from rapid_mlx.local_search.eligibility import low_information
    assert not low_information(text)


def test_semantic_format_policy():
    from rapid_mlx.local_search.eligibility import KEYWORD_ONLY_SUFFIXES, semantic_format
    for suffix in KEYWORD_ONLY_SUFFIXES:
        assert not semantic_format(Path("data" + suffix.upper()))
    for suffix in (".md", ".txt", ".rst", ".tex", ".html", ".pdf", ".docx", ".pptx", ".xlsx", ".py", ".swift"):
        assert semantic_format(Path("document" + suffix))


def test_machine_formats_and_noise_index_without_embedding(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        from rapid_mlx.local_search.eligibility import KEYWORD_ONLY_SUFFIXES
        for suffix in KEYWORD_ONLY_SUFFIXES:
            (root / ("data" + suffix)).write_text('first line\nunique_literal_needle')
        (root / "noise.txt").write_text("ab" * 200)
        (root / "notes.md").write_text("apple benchmark throughput")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        fid = search.add_folder(str(root))[0]["id"]
        await search.wait_idle()
        assert len(fake.calls) == 1 and len(fake.calls[0]) == 1
        assert len(search.store.vectors) == 1
        hits = (await search.search("unique_literal_needle", 100, None))["results"]
        data_hits = [h for h in hits if Path(h["path"]).suffix in KEYWORD_ONLY_SUFFIXES]
        assert len(data_hits) == len(KEYWORD_ONLY_SUFFIXES)
        assert all(h["matched"] == ["keyword"] and h["line"] == 1 for h in data_hits)
        calls = len(fake.calls)
        search.queue(fid)
        await search.wait_idle()
        assert len(fake.calls) == calls
        await search.close()
    asyncio.run(scenario())


def test_eligibility_migration_keeps_keyword_locators_and_prose(tmp_path):
    root = folder(tmp_path)
    db = tmp_path / "migration.sqlite3"
    store = Store(db, "model")
    fid = store.add_folder(root)[0]
    for name, text, status, vectors in [
        ("data.jsonl", "needle", "indexed", [[1., 0.]]),
        ("noise.txt", "ab" * 200, "indexed", [[1., 0.]]),
        ("note.md", "readable prose needle", "indexed", [[0., 1.]]),
        ("offline.txt", "readable but no model yet", "keyword", None),
        ("offline.log", "literal data", "keyword", None),
    ]:
        path = root / name
        path.write_text(text)
        store.replace(fid, path, path.stat(), name, [Chunk(text, 42, None)], vectors, status)
    before = store.rows()
    prose_id = next(i for i, r in before.items() if r['path'].endswith('note.md'))
    prose_vector = store.vectors[prose_id].copy()
    with store.db:
        store.db.execute("DELETE FROM meta WHERE key='eligibility_version'")
    store.close()
    store = Store(db, "model")
    assert store.rows() == before
    assert list(store.vectors) == [prose_id]
    np.testing.assert_array_equal(store.vectors[prose_id], prose_vector)
    assert len(store.keyword(["needle"], set(before))) == 2
    assert store.files(fid)[str(root / 'offline.txt')]['status'] == 'keyword'
    assert store.files(fid)[str(root / 'offline.log')]['status'] == 'indexed'
    store.close()
    # Opening twice has no further migration or model-dependent changes.
    store = Store(db, "model")
    assert store.rows() == before and list(store.vectors) == [prose_id]
    store.close()
