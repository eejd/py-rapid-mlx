# SPDX-License-Identifier: Apache-2.0
"""Original restrained sound design: low-power warning, relay, servo shutdown, air."""

import argparse
import wave
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sr = 48000
    n = 30 * sr
    rng = np.random.default_rng(8707)
    t = np.arange(n) / sr
    # Filtered air, stereo with decorrelated channels. No external samples.
    f = np.fft.rfftfreq(n, 1 / sr)
    shape = (
        np.sqrt(np.maximum(f, 80) / 500) ** -1
        * np.minimum(f / 100, 1) ** 2
        * np.exp(-((f / 2100) ** 2))
    )
    air = []
    for _ in range(2):
        v = np.fft.irfft(np.fft.rfft(rng.normal(size=n)) * shape, n=n)
        v = v / np.sqrt(np.mean(v * v)) * 0.0022
        v *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.13 * t + 0.4)
        air.append(v)
    out = np.stack(air, axis=1)

    def add(start, signal, pan=0):
        i = round(start * sr)
        count = min(len(signal), n - i)
        out[i : i + count, 0] += signal[:count] * np.sqrt((1 - pan) / 2)
        out[i : i + count, 1] += signal[:count] * np.sqrt((1 + pan) / 2)

    for stamp in (3.45, 4.65):
        u = np.arange(int(0.18 * sr)) / sr
        env = np.sin(np.pi * u / 0.18) ** 2
        add(
            stamp,
            0.038
            * env
            * (np.sin(2 * np.pi * 880 * u) + 0.12 * np.sin(2 * np.pi * 1760 * u)),
            -0.2,
        )
    u = np.arange(int(0.12 * sr)) / sr
    add(
        11.52,
        0.023
        * np.exp(-u * 65)
        * (rng.normal(size=len(u)) * 0.45 + np.sin(2 * np.pi * 340 * u)),
        0.45,
    )
    heater = np.sin(2 * np.pi * 57 * t) + 0.22 * np.sin(2 * np.pi * 114 * t)
    heater *= 0.0015 * np.clip((t - 11.5) / 0.6, 0, 1) * np.clip((21 - t) / 1.5, 0, 1)
    out += heater[:, None]
    u = np.arange(int(0.9 * sr)) / sr
    freq = 650 * np.exp(-u * 2.8) + 70
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.minimum(u / 0.025, 1) * (1 - u / 0.9) ** 2
    add(17.45, 0.027 * env * (np.sin(phase) + 0.15 * np.sin(phase * 2)), -0.3)
    fade = np.minimum(t / 0.5, 1) * np.clip((30 - t) / 1.5, 0, 1)
    out *= fade[:, None]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(a.output), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(out, -1, 1) * 32767).astype("<i2").tobytes())
    print(a.output)


if __name__ == "__main__":
    main()
