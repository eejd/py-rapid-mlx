# SPDX-License-Identifier: Apache-2.0
import pytest

from rapid_mlx import mlx_vlm_compat as compat


@pytest.mark.parametrize("version, expected", [
    ("0.7.2", True), ("0.7.12", True), ("0.7.12+local01", True),
    ("0.6.9", False), ("0.8.0", False), ("garbage", False),
])
def test_chat_versions(version, expected):
    assert compat.mlx_vlm_version_qualified(version, "chat") is expected


@pytest.mark.parametrize("capability", ["video", "audio", "diffusion", "image generation", "dflash", "mtp", "media prefix cache"])
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
    messages = [{"content": [{"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}]}]
    with pytest.raises(UnsupportedContentBlockError, match="video.*0.7.2"):
        validate_content_blocks_for_capabilities(
            messages, model_name="model", allow_image=True, allow_video=True
        )


def test_dflash_gate_rejects_unqualified_version(monkeypatch):
    from rapid_mlx.speculative.dflash.eligibility import have_runtime

    monkeypatch.setattr(compat, "version", lambda _: "0.7.12+local01")
    assert not have_runtime()
