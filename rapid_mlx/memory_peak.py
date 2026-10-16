"""Process-cumulative Metal peaks across local prefill sampling windows."""

import threading

import mlx.core as mx

_peak_lock = threading.Lock()
_peak_carry = 0


def reset_peak_for_sample() -> None:
    """Start a local window without losing the process high-water mark."""
    global _peak_carry
    with _peak_lock:
        _peak_carry = max(_peak_carry, int(mx.get_peak_memory()))
        mx.reset_peak_memory()


def cumulative_peak_bytes() -> int:
    """Read the process high-water mark, including previous sample windows."""
    with _peak_lock:
        return max(_peak_carry, int(mx.get_peak_memory()))
