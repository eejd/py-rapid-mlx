# SPDX-License-Identifier: Apache-2.0
"""Produce chronological start/middle/end review frames and an art storyboard."""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--media", type=Path, default=ROOT / "media")
    p.add_argument("--storyboard", type=Path, default=ROOT / "storyboard.json")
    a = p.parse_args()
    out = a.media / "qa"
    out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(a.storyboard.read_text())
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 23)
    labels = [
        "The choice",
        "One percent",
        "The last seedling",
        "The gift",
        "Warmth",
        "The last light",
        "Stillness",
        "Years later",
        "The flower",
    ]
    board = Image.new("RGB", (2280, 1440), "#101719")
    d = ImageDraw.Draw(board)
    d.text((40, 25), "THE LAST SHIFT", font=font, fill="#ecdfc4")
    for n, job in enumerate(jobs):
        x = 40 + (n % 3) * 750
        y = 90 + (n // 3) * 445
        with Image.open(
            a.media / "art" / job.get("keyframe", job["id"] + ".png")
        ) as im:
            board.paste(ImageOps.fit(im.convert("RGB"), (720, 360)), (x, y))
        d.text((x, y + 375), f"{n + 1:02d}  {labels[n]}", font=font, fill="#ecdfc4")
        src = a.media / "shots" / (job["id"] + ".mp4")
        if not src.exists():
            continue
        frames = []
        for k, stamp in enumerate([0.08, 1.33, 2.58]):
            target = out / f"{job['id']}-{k}.png"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(stamp),
                    "-i",
                    str(src),
                    "-frames:v",
                    "1",
                    str(target),
                ],
                check=True,
            )
            frames.append(target)
        sheet = Image.new("RGB", (1920, 360), "#101719")
        sd = ImageDraw.Draw(sheet)
        for k, f in enumerate(frames):
            with Image.open(f) as im:
                sheet.paste(im.resize((640, 320)), (k * 640, 0))
            sd.text((k * 640 + 10, 328), f"{job['id']} | {k}", font=small, fill="white")
        sheet.save(out / (job["id"] + "-motion.jpg"), quality=95)
    board.save(out / "storyboard.jpg", quality=96)


if __name__ == "__main__":
    main()
