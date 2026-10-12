# SPDX-License-Identifier: Apache-2.0
"""Queue the Qwen remake's image, motion and editorial stages through ComfyUI."""

import argparse
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

from batch import api

ROOT = Path(__file__).resolve().parent


def execute(base, workflow, target, timeout=14400, source_sha256=None):
    receipt_path = target.with_suffix(target.suffix + ".comfy.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = (
        workflow
        if source_sha256 is None
        else {"workflow": workflow, "source_sha256": source_sha256}
    )
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["workflow_sha256"] != digest:
            raise RuntimeError("Workflow changed; select a fresh output directory")
        if target.exists() and receipt.get("output_sha256"):
            if (
                hashlib.sha256(target.read_bytes()).hexdigest()
                != receipt["output_sha256"]
            ):
                raise RuntimeError("Output hash mismatch")
            print("Reused", target, flush=True)
            return
    else:
        queued = api(base, "/prompt", {"prompt": workflow})
        receipt = {
            "prompt_id": queued["prompt_id"],
            "workflow": workflow,
            "workflow_sha256": digest,
        }
        if source_sha256 is not None:
            receipt["source_sha256"] = source_sha256
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Waiting", target.name, receipt["prompt_id"], flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = api(base, "/history/" + receipt["prompt_id"]).get(
            receipt["prompt_id"]
        )
        if history and history.get("status", {}).get("status_str") == "error":
            raise RuntimeError(json.dumps(history["status"]))
        if history and history.get("status", {}).get("completed"):
            break
        time.sleep(3)
    else:
        raise TimeoutError("Prompt is recorded; inspect the queue before rerunning")
    artifacts = history.get("outputs", {}).get("2", {}).get("images", [])
    if len(artifacts) != 1:
        raise RuntimeError("Expected exactly one ComfyUI saved artifact")
    artifact = artifacts[0]
    query = {k: artifact[k] for k in ("filename", "subfolder", "type") if k in artifact}
    with urllib.request.urlopen(
        base.rstrip("/") + "/view?" + urllib.parse.urlencode(query), timeout=120
    ) as response:
        data = response.read()
    target.write_bytes(data)
    receipt.update(
        artifact=artifact,
        history=history,
        output_sha256=hashlib.sha256(data).hexdigest(),
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Saved", target, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["images", "videos", "edit"], required=True)
    p.add_argument("--comfy-url", default="http://127.0.0.1:8199")
    p.add_argument("--image-url", default="http://127.0.0.1:19427")
    p.add_argument("--video-url", default="http://127.0.0.1:19428")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--comfy-input", type=Path, required=True)
    p.add_argument("--limit", type=int, default=99)
    p.add_argument("--only", help="Comma-separated shot IDs")
    p.add_argument(
        "--image-storyboard", type=Path, default=ROOT / "qwen-storyboard.json"
    )
    p.add_argument(
        "--video-storyboard", type=Path, default=ROOT / "qwen-motion-storyboard.json"
    )
    p.add_argument(
        "--review-directory",
        type=Path,
        help="Wait for a matching keyframe review before queuing each motion shot",
    )
    a = p.parse_args()
    image_jobs = json.loads(a.image_storyboard.read_text())
    video_jobs = json.loads(a.video_storyboard.read_text())
    selected = None if not a.only else set(a.only.split(","))
    if a.stage == "edit":
        workflow = {
            "1": {
                "class_type": "RapidMLXLastShiftEdit",
                "inputs": {
                    "shots_directory": str((a.output / "shots").resolve()),
                    "filename_prefix": "last-shift-qwen/final/the-last-shift-qwen-en",
                },
            },
            "2": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["1", 0],
                    "filename_prefix": "last-shift-qwen/delivery",
                    "format": "mp4",
                    "format.codec": "auto",
                    "codec": "auto",
                },
            },
        }
        execute(a.comfy_url, workflow, a.output / "final/the-last-shift-qwen-en.mp4")
        return
    jobs = image_jobs if a.stage == "images" else video_jobs
    jobs = [j for j in jobs if selected is None or j["id"] in selected][: a.limit]
    for job in jobs:
        shot = job["id"]
        if a.stage == "images":
            workflow = {
                "1": {
                    "class_type": "RapidMLXQwenImage",
                    "inputs": {
                        "base_url": a.image_url,
                        **{
                            k: job[k]
                            for k in ("prompt", "width", "height", "steps", "seed")
                        },
                    },
                },
                "2": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["1", 0],
                        "filename_prefix": "last-shift-qwen/art/" + shot,
                    },
                },
            }
            target = a.output / "art" / (shot + ".png")
            execute(a.comfy_url, workflow, target)
            a.comfy_input.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, a.comfy_input / ("qwen-" + shot + ".png"))
        else:
            source = a.output / "art" / (shot + ".png")
            if a.review_directory:
                review_path = a.review_directory / (shot + ".json")
                deadline = time.monotonic() + 14400
                while not (source.exists() and review_path.exists()):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Keyframe review is still pending: " + shot)
                    time.sleep(3)
                review = json.loads(review_path.read_text())
                if (
                    review.get("status") != "accepted"
                    or review.get("keyframe_sha256")
                    != hashlib.sha256(source.read_bytes()).hexdigest()
                ):
                    raise RuntimeError(
                        "Keyframe review does not match this source image: " + shot
                    )
            workflow = {
                "0": {
                    "class_type": "LoadImage",
                    "inputs": {"image": "qwen-" + shot + ".png"},
                },
                "1": {
                    "class_type": "RapidMLXLastShift",
                    "inputs": {
                        "image": ["0", 0],
                        "base_url": a.video_url,
                        "prompt": job["motion_prompt"],
                        "width": 1024,
                        "height": 512,
                        "frames": 65,
                        "seed": job["seed"],
                    },
                },
                "2": {
                    "class_type": "SaveVideo",
                    "inputs": {
                        "video": ["1", 0],
                        "filename_prefix": "last-shift-qwen/shots/" + shot,
                        "format": "mp4",
                        "format.codec": "auto",
                        "codec": "auto",
                    },
                },
            }
            loaded_source = a.comfy_input / ("qwen-" + shot + ".png")
            if (
                hashlib.sha256(loaded_source.read_bytes()).digest()
                != hashlib.sha256(source.read_bytes()).digest()
            ):
                raise RuntimeError(
                    "ComfyUI input differs from reviewed keyframe: " + shot
                )
            execute(
                a.comfy_url,
                workflow,
                a.output / "shots" / (shot + ".mp4"),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    main()
