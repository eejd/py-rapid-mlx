# SPDX-License-Identifier: Apache-2.0
"""Thirty-second narrative edit; frame-exact timing and original continuous score."""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FRAMES = [72, 60, 72, 72, 84, 72, 60, 96, 132]


def run(args):
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *map(str, args),
        ],
        check=True,
    )


def probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graphics(path, text, title=False):
    # Typeset graphics only; photographic assets are never painted or retouched here.
    im = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(
        "/System/Library/Fonts/Supplemental/Arial.ttf", 60 if title else 36
    )
    x, y = (1790, 850) if title else (115, 925)
    anchor = "ra" if title else "la"
    d.text(
        (x, y),
        text,
        font=font,
        anchor=anchor,
        fill=(248, 239, 211, 245),
        stroke_width=1,
        stroke_fill=(15, 20, 17, 170),
    )
    im.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--sound", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    jobs = json.loads((ROOT / "storyboard.json").read_text())
    if len(jobs) != len(FRAMES):
        raise ValueError("Edit requires all nine approved shots")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path("/private/tmp/demo-last-shift-render")
    scratch.mkdir(parents=True, exist_ok=True)
    evidence = {
        "native_resolution": "1024x512",
        "delivery": "1920x1080 (1920x960 editorial upscale plus letterbox)",
        "fps": 24,
        "frames": sum(FRAMES),
        "shots": [],
        "score_sha256": sha(a.score),
        "sound_sha256": sha(a.sound),
    }
    encode = [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "24",
        "-fps_mode",
        "cfr",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
    ]
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        t = Path(tmp)
        clips = []
        for n, (job, frames) in enumerate(zip(jobs, FRAMES)):
            src = a.input / (job["id"] + ".mp4")
            info = probe(src)
            v = next(s for s in info["streams"] if s["codec_type"] == "video")
            if (v["width"], v["height"], v["avg_frame_rate"], int(v["nb_frames"])) != (
                1024,
                512,
                "24/1",
                65,
            ):
                raise ValueError(f"Unexpected native stream: {src}")
            if not any(s["codec_type"] == "audio" for s in info["streams"]):
                raise ValueError(f"Native audio absent: {src}")
            duration = frames / 24
            native = 65 / 24
            speed = min(duration / native, 1.8)
            # Interpolate before upscale. End padding covers filter boundary frames;
            # the final shot intentionally settles into a brief closing hold.
            vf = f"setpts={speed:.9f}*(PTS-STARTPTS),minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,tpad=stop_mode=clone:stop_duration=3,trim=duration={duration},scale=1920:960:flags=lanczos,pad=1920:1080:0:60:black,setsar=1"
            if n == 0:
                vf += ",fade=t=in:st=0:d=0.35"
            if n == 8:
                vf += f",fade=t=out:st={duration - 0.8}:d=0.8"
            audio_info = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-i",
                    str(src),
                    "-vn",
                    "-af",
                    "volumedetect",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stderr
            match = re.search(r"max_volume: (-?[0-9.]+|-inf) dB", audio_info)
            if not match:
                raise RuntimeError("Audio peak measurement absent")
            # Silence must bypass loudnorm: some FFmpeg builds emit NaNs for it.
            normal = (
                "loudnorm=I=-31:TP=-7:LRA=7," if float(match.group(1)) > -80 else ""
            )
            af = f"atempo={1 / speed:.9f},{normal}aresample=48000,apad,atrim=duration={duration},afade=t=in:d=0.08,afade=t=out:st={duration - 0.15}:d=0.15"
            out = t / f"{n}.mp4"
            run(["-i", src, "-vf", vf, "-af", af, "-t", duration, *encode, out])
            clips.append(out)
            evidence["shots"].append(
                {
                    "id": job["id"],
                    "input_sha256": sha(src),
                    "native_frames": 65,
                    "edit_frames": frames,
                    "time_scale": speed,
                    "native_probe": info,
                }
            )
        listing = t / "concat.txt"
        listing.write_text("".join(f"file '{s}'\n" for s in clips))
        joined = t / "joined.mp4"
        run(["-f", "concat", "-safe", "0", "-i", listing, *encode, "-t", 30, joined])
        opening = t / "opening.png"
        years = t / "years.png"
        title = t / "title.png"
        battery = t / "battery.png"
        graphics(battery, "POWER REMAINING: 1%")
        graphics(opening, "MARS · ABANDONED OUTPOST")
        graphics(years, "YEARS LATER")
        graphics(title, "THE LAST SHIFT", title=True)
        # Shot eight begins at 20.5s. Graphics remain part of the picture, never cards.
        graph = "[2:v]format=rgba,fade=t=in:st=0.5:d=0.3:alpha=1,fade=t=out:st=2.2:d=0.3:alpha=1[o];[3:v]format=rgba,fade=t=in:st=20.7:d=0.3:alpha=1,fade=t=out:st=22:d=0.3:alpha=1[y];[4:v]format=rgba,fade=t=in:st=26.8:d=0.5:alpha=1,fade=t=out:st=29.2:d=0.8:alpha=1[t];[5:v]format=rgba,fade=t=in:st=3.15:d=0.3:alpha=1,fade=t=out:st=5:d=0.3:alpha=1[b];[0:v][o]overlay[v1];[v1][y]overlay[v2];[v2][t]overlay[v3];[v3][b]overlay[v];[0:a]volume=0.6[amb];[1:a]loudnorm=I=-25:TP=-4:LRA=8,aresample=48000,afade=t=in:d=1,afade=t=out:st=28:d=2[mus];[amb][mus][6:a]amix=inputs=3:duration=first:normalize=0,loudnorm=I=-20:TP=-1.5:LRA=9,aresample=48000[a]"
        run(
            [
                "-i",
                joined,
                "-i",
                a.score,
                "-loop",
                1,
                "-i",
                opening,
                "-loop",
                1,
                "-i",
                years,
                "-loop",
                1,
                "-i",
                title,
                "-loop",
                1,
                "-i",
                battery,
                "-i",
                a.sound,
                "-filter_complex",
                graph,
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-t",
                30,
                *encode,
                "-movflags",
                "+faststart",
                a.output,
            ]
        )
    run(["-i", a.output, "-f", "null", "-"])
    evidence.update(output_sha256=sha(a.output), output_probe=probe(a.output))
    (a.output.parent / "render-evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print(a.output)


if __name__ == "__main__":
    main()
