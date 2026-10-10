# SPDX-License-Identifier: Apache-2.0
"""Edit generated shots into a film; retain their native audio and add title cards."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def run(args):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True
    )


def font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads((Path(__file__).parent / "storyboard.json").read_text())
    sources = [args.input / (job["id"] + ".mp4") for job in jobs]
    for source in sources:
        if not source.is_file():
            parser.error(f"Missing generated shot: {source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path("/private/tmp/demo-cinema-render")
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        temporary = Path(temporary)
        clips = []
        encode = [
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
        ]
        for index, (headline, subtitle) in enumerate(
            [
                ("THE LAST LANTERN", "An AI-generated microfilm"),
                (
                    "RAPID-MLX",
                    "ComfyUI + Qwen-Image 2.1 + LTX-2.5 | Generated locally on a Mac",
                ),
            ]
        ):
            image = Image.new("RGB", (1920, 1080), "#071019")
            draw = ImageDraw.Draw(image)
            draw.text((960, 465), headline, font=font(98), anchor="mm", fill="#ffd08a")
            draw.text((960, 585), subtitle, font=font(32), anchor="mm", fill="#e6e6dd")
            draw.text(
                (960, 990),
                "AI GENERATED  |  Editorial upscale from 704 x 480",
                font=font(24),
                anchor="mm",
                fill="#91a7b5",
            )
            png, clip = temporary / f"card{index}.png", temporary / f"card{index}.mp4"
            image.save(png)
            run(
                [
                    "-loop",
                    "1",
                    "-framerate",
                    "24",
                    "-i",
                    str(png),
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=48000:cl=stereo",
                    "-t",
                    "3",
                    *encode,
                    str(clip),
                ]
            )
            clips.append(clip)
        shots = []
        for index, source in enumerate(sources):
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_streams",
                        "-of",
                        "json",
                        str(source),
                    ]
                )
            )
            has_audio = any(
                stream["codec_type"] == "audio" for stream in probe["streams"]
            )
            if not has_audio:
                raise RuntimeError(
                    f"Expected native LTX audio missing: {source}; review before editing."
                )
            clip = temporary / f"shot{index}.mp4"
            run(
                [
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-vf",
                    "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x071019,setsar=1,fps=24",
                    "-af",
                    "aresample=48000,apad",
                    "-shortest",
                    *encode,
                    str(clip),
                ]
            )
            shots.append(clip)
        listing = temporary / "concat.txt"
        # Temporary filenames are controlled and contain no shell/concat quoting characters.
        listing.write_text(
            "".join(f"file '{path}'\n" for path in [clips[0], *shots, clips[1]])
        )
        run(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(args.output),
            ]
        )
    print(args.output)


if __name__ == "__main__":
    main()
