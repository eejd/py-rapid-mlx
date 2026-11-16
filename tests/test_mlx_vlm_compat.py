# SPDX-License-Identifier: Apache-2.0
import pytest

from rapid_mlx import mlx_vlm_compat as compat


@pytest.mark.parametrize(
    "version, expected",
    [
        ("0.7.2", True),
        ("0.7.12", True),
        ("0.7.12+local01", True),
        ("0.6.9", False),
        ("0.8.0", False),
        ("garbage", False),
    ],
)
def test_chat_versions(version, expected):
    assert compat.mlx_vlm_version_qualified(version, "chat") is expected


@pytest.mark.parametrize(
    "capability",
    [
        "video",
        "audio",
        "diffusion",
        "image generation",
        "dflash",
        "mtp",
        "media prefix cache",
    ],
)
def test_unqualified_paths_are_strict(capability, monkeypatch):
    monkeypatch.setattr(compat, "version", lambda _: "0.7.12+local01")
    with pytest.raises(RuntimeError, match=capability):
        compat.require_mlx_vlm_capability(capability)
    monkeypatch.setattr(compat, "version", lambda _: "0.7.2")
    compat.require_mlx_vlm_capability(capability)


def test_video_request_is_rejected_at_boundary(monkeypatch):
    from rapid_mlx.api.utils import (
        UnsupportedContentBlockError,
        validate_content_blocks_for_capabilities,
    )

    monkeypatch.setattr(compat, "version", lambda _: "0.7.12+local01")
    messages = [
        {
            "content": [
                {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}
            ]
        }
    ]
    with pytest.raises(UnsupportedContentBlockError, match="video.*0.7.2"):
        validate_content_blocks_for_capabilities(
            messages, model_name="model", allow_image=True, allow_video=True
        )


def test_dflash_gate_rejects_unqualified_version(monkeypatch):
    from rapid_mlx.speculative.dflash.eligibility import have_runtime

    monkeypatch.setattr(compat, "version", lambda _: "0.7.12+local01")
    assert not have_runtime()


def test_runtime_spec_matches_project():
    from pathlib import Path

    import tomllib
    from packaging.requirements import Requirement

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    dep = next(
        Requirement(x)
        for x in project["project"]["dependencies"]
        if Requirement(x).name == "mlx-vlm"
    )
    for v in (
        "0.7.2",
        "0.7.2.1",
        "0.7.2.post1",
        "0.7.12",
        "0.7.12.post1",
        "0.7.12+local01",
        "0.7.4",
        "0.7.13",
        "0.8.0",
        "0.7.12rc1",
    ):
        assert compat.mlx_vlm_version_qualified(v, "chat") == (v in dep.specifier)


@pytest.mark.parametrize(
    "installed,expected", [("0.7.2", True), ("0.7.12+local01", True), ("0.7.4", False)]
)
def test_real_vision_gate(monkeypatch, installed, expected):
    import sys
    from types import ModuleType

    from rapid_mlx.models import mllm

    monkeypatch.setitem(sys.modules, "mlx_vlm", ModuleType("mlx_vlm"))
    monkeypatch.setattr(mllm, "version", lambda _: installed)
    status, _ = mllm.vision_runtime_status()
    assert (status is mllm.VisionRuntimeStatus.OK) == expected


@pytest.mark.parametrize("installed,expected", [("0.7.2", "auto"), ("0.7.12", "off")])
def test_media_prefix_gate(monkeypatch, installed, expected):
    from rapid_mlx.mllm_batch_generator import MLLMBatchGenerator

    class Stop:
        def __setattr__(self, name, value):
            if name == "_media_structural_singleton":
                raise RuntimeError("stop after media gate")
            object.__setattr__(self, name, value)

    monkeypatch.setattr(compat, "version", lambda _: installed)
    fake = Stop()
    with pytest.raises(RuntimeError, match="stop after media gate"):
        MLLMBatchGenerator.__init__(fake, model=None, processor=None)
    assert fake.media_prefix_cache == expected


@pytest.mark.parametrize("installed,expected", [("0.7.2", True), ("0.7.12", False)])
def test_native_mtp_gate(monkeypatch, installed, expected):
    from rapid_mlx.speculative.native_mtp import runtime

    monkeypatch.setattr(runtime, "version", lambda _: installed)
    monkeypatch.setattr(runtime, "find_spec", lambda _: object())
    assert runtime.have_runtime() is expected


def test_diffusion_cli_guard_is_actionable(monkeypatch):
    from rapid_mlx.models import mllm
    from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing

    monkeypatch.setattr(
        mllm, "vision_runtime_status", lambda: (mllm.VisionRuntimeStatus.OK, None)
    )
    monkeypatch.setattr(mllm, "version", lambda _: "0.7.12+local01")
    with pytest.raises(OptionalRuntimeMissing, match="diffusion.*0.7.12"):
        mllm.require_mlx_vlm_or_exit("model", text_diffusion=True)
    monkeypatch.setattr(mllm, "version", lambda _: "0.7.2")
    mllm.require_mlx_vlm_or_exit("model", text_diffusion=True)


def test_hidream_gate(monkeypatch):
    from rapid_mlx.image.hidream_runtime.runtime import HiDreamO1

    monkeypatch.setattr(compat, "version", lambda _: "0.7.12")
    with pytest.raises(RuntimeError, match="image generation.*0.7.2"):
        HiDreamO1("unused")


def test_diffusion_worker_rejects_unqualified(monkeypatch):
    from types import SimpleNamespace

    from rapid_mlx.runtime.diffusion_lane import DiffusionEngine

    ready = SimpleNamespace(set=lambda: None)
    worker = SimpleNamespace(_ready=ready)
    monkeypatch.setattr(compat, "version", lambda _: "0.7.12")
    DiffusionEngine._worker_loop(worker)
    assert "diffusion" in str(worker._load_error)
    assert "0.7.2" in str(worker._load_error)
