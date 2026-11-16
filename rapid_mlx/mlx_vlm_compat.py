# SPDX-License-Identifier: Apache-2.0
"""Capabilities qualified against mlx-vlm releases.

The 0.7.12 qualification covers text and image chat only. Native video,
diffusion, speculative decoding and media-prefix reuse retain the old pin.
"""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version

QUALIFIED_MLX_VLM = {
    "chat": ("0.7.2", "0.7.12"),
    "video": ("0.7.2",),
    "audio": ("0.7.2",),
    "diffusion": ("0.7.2",),
    "dflash": ("0.7.2",),
    "mtp": ("0.7.2",),
    "media prefix cache": ("0.7.2",),
}


def mlx_vlm_version_qualified(installed: str, capability: str) -> bool:
    try:
        parsed = Version(installed)
    except InvalidVersion:
        return False
    return not parsed.is_prerelease and not parsed.is_postrelease and parsed.base_version in QUALIFIED_MLX_VLM[capability]


def mlx_vlm_capability_qualified(capability: str) -> bool:
    try:
        return mlx_vlm_version_qualified(version("mlx-vlm"), capability)
    except PackageNotFoundError:
        return False


def require_mlx_vlm_capability(capability: str) -> None:
    if not mlx_vlm_capability_qualified(capability):
        qualified = ", ".join(QUALIFIED_MLX_VLM[capability])
        raise RuntimeError(f"mlx-vlm {capability} requires a qualified version: {qualified}")
