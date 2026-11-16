# SPDX-License-Identifier: Apache-2.0
"""Capabilities qualified against mlx-vlm releases.

The 0.7.12 qualification covers text and image chat only. Native video,
diffusion, speculative decoding and media-prefix reuse retain the old pin.
"""

from importlib.metadata import PackageNotFoundError, version

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

CHAT_MLX_VLM_SPEC = (
    ">=0.7.2,<0.7.13,!=0.7.3.*,!=0.7.4.*,!=0.7.5.*,!=0.7.6.*,"
    "!=0.7.7.*,!=0.7.8.*,!=0.7.9.*,!=0.7.10.*,!=0.7.11.*"
)

QUALIFIED_MLX_VLM = {
    "chat": ("0.7.2", "0.7.12"),
    "video": ("0.7.2",),
    "audio": ("0.7.2",),
    "diffusion": ("0.7.2",),
    "image generation": ("0.7.2",),
    "dflash": ("0.7.2",),
    "mtp": ("0.7.2",),
    "media prefix cache": ("0.7.2",),
}


def mlx_vlm_version_qualified(installed: str, capability: str) -> bool:
    try:
        parsed = Version(installed)
    except InvalidVersion:
        return False
    if capability == "chat":
        return parsed in SpecifierSet(CHAT_MLX_VLM_SPEC)
    return not parsed.is_prerelease and Version(parsed.base_version) in {
        Version(v) for v in QUALIFIED_MLX_VLM[capability]
    }


def mlx_vlm_capability_qualified(capability: str) -> bool:
    try:
        return mlx_vlm_version_qualified(version("mlx-vlm"), capability)
    except PackageNotFoundError:
        return False


def qualified_runtime_hint(capability: str) -> str:
    requirements = ", ".join(f"mlx-vlm=={v}" for v in QUALIFIED_MLX_VLM[capability])
    return f"Use {requirements} via the environment's package manager."


def require_mlx_vlm_capability(capability: str) -> None:
    if not mlx_vlm_capability_qualified(capability):
        qualified = ", ".join(QUALIFIED_MLX_VLM[capability])
        raise RuntimeError(
            f"mlx-vlm {capability} requires a qualified version: {qualified}"
        )
