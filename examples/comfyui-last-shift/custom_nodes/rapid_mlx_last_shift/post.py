# SPDX-License-Identifier: Apache-2.0
"""Run the established deterministic edit as an explicit ComfyUI node."""

import subprocess
import sys
from pathlib import Path


class RapidMLXLastShiftEdit:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "shots_directory": ("STRING",),
                "filename_prefix": ("STRING", {"default": "last-shift-qwen/final"}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "render"
    CATEGORY = "Rapid-MLX"

    def render(self, shots_directory, filename_prefix):
        import folder_paths
        from comfy_api.latest import InputImpl

        prefix = Path(filename_prefix)
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(
                "Output prefix must stay inside the ComfyUI output directory"
            )
        output = Path(folder_paths.get_output_directory()) / prefix
        output.parent.mkdir(parents=True, exist_ok=True)
        audio = output.parent / "audio"
        audio.mkdir(exist_ok=True)
        root = Path(__file__).resolve().parents[2]
        subprocess.run(
            ["swift", str(root / "score.swift"), str(audio / "score.wav")], check=True
        )
        subprocess.run(
            [
                sys.executable,
                str(root / "sound.py"),
                "--output",
                str(audio / "sound.wav"),
            ],
            check=True,
        )
        film = output.with_suffix(".mp4")
        subprocess.run(
            [
                sys.executable,
                str(root / "render.py"),
                "--input",
                shots_directory,
                "--score",
                str(audio / "score.wav"),
                "--sound",
                str(audio / "sound.wav"),
                "--output",
                str(film),
            ],
            check=True,
        )
        return (InputImpl.VideoFromFile(str(film)),)
