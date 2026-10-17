"""Process-cumulative Metal peaks across local prefill sampling windows."""

import threading

import mlx.core as mx

_peak_lock = threading.Lock()
_peak_carry = 0
_sample_epoch = 0


def reset_peak_for_sample() -> int:
    """Start a local window and return its token, preserving process peaks."""
    global _peak_carry, _sample_epoch
    with _peak_lock:
        _peak_carry = max(_peak_carry, int(mx.get_peak_memory()))
        mx.reset_peak_memory()
        _sample_epoch += 1
        return _sample_epoch


def sample_window_intact(token: int) -> bool:
    """Whether no other sampling reset has interrupted this window."""
    with _peak_lock:
        return token == _sample_epoch


def cumulative_peak_bytes() -> int:
    """Read the process high-water mark, including previous sample windows."""
    with _peak_lock:
        return max(_peak_carry, int(mx.get_peak_memory()))
