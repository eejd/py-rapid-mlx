# SPDX-License-Identifier: Apache-2.0
"""Two-pass ComfyUI batch: finish all keyframes before running video shots."""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def api(base, route, payload=None):
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["keyframes", "shots"])
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8189")
    parser.add_argument("--image-url", default="http://127.0.0.1:18427")
    parser.add_argument("--video-url", default="http://127.0.0.1:18428")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=9000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    template = json.loads((ROOT / "workflow-api.json").read_text())
    jobs = json.loads((ROOT / "storyboard.json").read_text())[: args.limit]
    for shot in jobs:
        workflow = json.loads(json.dumps(template))
        workflow["1"]["inputs"].update(
            base_url=args.image_url, prompt=shot["keyframe_prompt"], seed=shot["seed"]
        )
        workflow["2"]["inputs"]["filename_prefix"] = "rapid-cinema/" + shot["id"]
        if args.stage == "keyframes":
            del workflow["3"], workflow["4"]
            node, kind, suffix = "2", "images", ".png"
        else:
            frame_receipt = json.loads(
                (args.output / (shot["id"] + "-keyframes.json")).read_text()
            )
            # SaveImage's artifact is already in Comfy's output folder; no upload or inference.
            frame = frame_receipt["artifact"]
            workflow["1"] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": frame["subfolder"] + "/" + frame["filename"] + " [output]"
                },
            }
            del workflow["2"]
            workflow["3"]["inputs"].update(
                base_url=args.video_url, prompt=shot["motion_prompt"], seed=shot["seed"]
            )
            workflow["4"]["inputs"]["filename_prefix"] = "rapid-cinema/" + shot["id"]
            node, kind, suffix = "4", "images", ".mp4"
        fingerprint = hashlib.sha256(
            json.dumps(workflow, sort_keys=True).encode()
        ).hexdigest()
        path = args.output / (shot["id"] + "-" + args.stage + ".json")
        target = args.output / (shot["id"] + suffix)
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt["request_sha256"] != fingerprint:
                raise RuntimeError("Workflow changed; use a fresh output directory.")
            if receipt.get("file_sha256") and target.exists():
                if (
                    hashlib.sha256(target.read_bytes()).hexdigest()
                    != receipt["file_sha256"]
                ):
                    raise RuntimeError("Output file hash mismatch.")
                print("Already saved", target, flush=True)
                continue
        else:
            queued = api(args.comfy_url, "/prompt", {"prompt": workflow})
            receipt = {
                "prompt_id": queued["prompt_id"],
                "workflow": workflow,
                "request_sha256": fingerprint,
            }
            path.write_text(json.dumps(receipt, indent=2) + "\n")
        print("Waiting", shot["id"], receipt["prompt_id"], flush=True)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            history = api(args.comfy_url, "/history/" + receipt["prompt_id"]).get(
                receipt["prompt_id"]
            )
            if history and history.get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(history["status"]))
            if history and history.get("status", {}).get("completed"):
                break
            time.sleep(2)
        else:
            raise TimeoutError(
                "Prompt ID recorded; rerun to continue polling, inspect Comfy history if restarted."
            )
        artifacts = history.get("outputs", {}).get(node, {}).get(kind, [])
        if len(artifacts) != 1:
            raise RuntimeError(f"Expected one saved artifact; inspect {history}")
        artifact = artifacts[0]
        query = {
            k: artifact[k] for k in ("filename", "subfolder", "type") if k in artifact
        }
        with urllib.request.urlopen(
            args.comfy_url.rstrip("/") + "/view?" + urllib.parse.urlencode(query),
            timeout=120,
        ) as response:
            data = response.read()
        target.write_bytes(data)
        receipt.update(
            artifact=artifact,
            history=history,
            file_sha256=hashlib.sha256(data).hexdigest(),
        )
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        print("Saved", target, flush=True)


if __name__ == "__main__":
    main()
