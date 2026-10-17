# SPDX-License-Identifier: Apache-2.0
"""Python binding for the vendored TurboQuant fused Metal kernels.

The Metal source lives in ``turboquant_fused.metal`` (vendored from
arozanov/turboquant-mlx, Apache-2.0). This module loads the source at
first call, dispatches via ``mx.fast.metal_kernel`` and caches the
compiled handle at module scope.

:func:`fused_quantize_k8` provides the rapid-mlx K-side symmetric
uniform path for the K8V4 mix.

It returns ``None`` when Metal compilation fails so the caller
can fall back to the pure-MLX reference path without an unhandled
exception (R15 Phase 4 deliverable: "Falls back to unfused path if
Metal compilation fails").
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metal source loading
# ---------------------------------------------------------------------------

_METAL_SOURCE_PATH = Path(__file__).with_name("turboquant_fused.metal")
_KERNEL_SENTINEL = "// >>> kernel: "

# Module-level cache for compiled kernels. ``False`` is the
# "tried-and-failed" sentinel so we don't keep recompiling on the hot
# path after the first failure.
_KERNEL_CACHE: dict[str, object] = {}


def _load_kernel_sources() -> dict[str, str]:
    """Split the .metal file into one ``{name: source}`` mapping.

    Sentinel format is ``// >>> kernel: <name>`` on a line of its own.
    Everything between two sentinels (or from a sentinel to EOF) is the
    body shipped to ``mx.fast.metal_kernel``.
    """
    text = _METAL_SOURCE_PATH.read_text(encoding="utf-8")
    parts: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(_KERNEL_SENTINEL):
            if current_name is not None:
                parts[current_name] = "\n".join(current_lines)
            current_name = line[len(_KERNEL_SENTINEL) :].strip()
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        parts[current_name] = "\n".join(current_lines)
    return parts


def _compile_kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
) -> object | None:
    """Compile a kernel from the .metal file. Returns ``None`` on failure.

    Failure paths (any of which is benign — the caller falls back to the
    pure-MLX reference path):
      * Metal is unavailable on this host (CPU-only build, or running
        under a remote test runner that does not have a GPU).
      * The Metal source has a syntax error after a future upstream
        port.
      * ``mx.fast.metal_kernel`` raises because of an MLX version
        mismatch (the API moved between 0.20 and 0.22).
    """
    cached = _KERNEL_CACHE.get(name)
    if cached is False:
        return None
    if cached is not None:
        return cached
    try:
        sources = _load_kernel_sources()
        if name not in sources:
            logger.warning(
                "turboquant_fused: kernel %r missing from %s; falling back",
                name,
                _METAL_SOURCE_PATH.name,
            )
            _KERNEL_CACHE[name] = False
            return None
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=sources[name],
        )
    except Exception as exc:  # pragma: no cover - exercised only on Metal-less hosts
        logger.warning(
            "turboquant_fused: failed to compile %r (%s); falling back", name, exc
        )
        _KERNEL_CACHE[name] = False
        return None
    _KERNEL_CACHE[name] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Public bindings
# ---------------------------------------------------------------------------


def is_metal_available() -> bool:
    """Return True when Metal compilation will work on this host.

    Used by the metrics route to surface the
    ``rapid_mlx_turboquant_fused_kernel{status}`` gauge at serve boot.
    """
    try:
        return mx.default_device() == mx.gpu and mx.metal.is_available()
    except Exception:
        return False


def fused_quantize_k8(
    vectors: mx.array,
    signs: mx.array,
    dim: int,
) -> tuple[mx.array, mx.array, mx.array] | None:
    """Encode K vectors with the K-side symmetric uniform fused kernel.

    Returns ``(packed_uint8, norms_fp32, k_scales_fp32)`` on success,
    ``None`` on compile failure (caller falls back to the pure-MLX
    K-side reference path).
    """
    if not is_metal_available():
        return None
    kernel = _compile_kernel(
        "tq_fused_quantize_k8",
        input_names=["inp", "signs", "dims"],
        output_names=["packed_k_out", "norms_out", "k_scales_out"],
    )
    if kernel is None:
        return None

    n_vecs = vectors.shape[0]
    dims_arr = mx.array([dim, 8], dtype=mx.uint32)

    try:
        outputs = kernel(
            inputs=[
                vectors.reshape(n_vecs * dim).astype(mx.float32),
                signs.astype(mx.float32),
                dims_arr,
            ],
            template=[],
            grid=(n_vecs * dim, 1, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(n_vecs * dim,), (n_vecs,), (n_vecs,)],
            output_dtypes=[mx.uint8, mx.float32, mx.float32],
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("turboquant_fused: K8 dispatch failed (%s); falling back", exc)
        return None

    packed = outputs[0].reshape(n_vecs, dim)
    return packed, outputs[1], outputs[2]


def reset_kernel_cache_for_tests() -> None:
    """Test-only hook: drop the module cache so the next call recompiles."""
    _KERNEL_CACHE.clear()
