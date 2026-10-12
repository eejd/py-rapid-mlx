# SPDX-License-Identifier: Apache-2.0
"""Qwen Image 2.1 through Rapid; ComfyUI carries pixels, never loads weights."""

import base64
import hashlib
import io
import json
import time
from pathlib import Path

from .client import multipart, request, save_json


class RapidMLXQwenImage:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "base_url": ("STRING", {"default": "http://127.0.0.1:19427"}),
                "prompt": ("STRING", {"multiline": True}),
                "width": (
                    "INT",
                    {"default": 1536, "min": 256, "max": 2048, "step": 64},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 256, "max": 2048, "step": 64},
                ),
                "steps": ("INT", {"default": 40, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 87101, "min": 0, "max": 2147483647}),
            },
            "optional": {"reference": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Rapid-MLX"

    def generate(self, base_url, prompt, width, height, steps, seed, reference=None):
        import folder_paths
        import numpy as np
        import torch
        from PIL import Image

        fields = {
            "model": "qwen-image-2.1",
            "prompt": prompt,
            "steps": steps,
            "seed": seed,
            "size": f"{width}x{height}",
            "guidance": 1.0,
            "response_format": "b64_json",
        }
        png = b""
        if reference is not None:
            if len(reference) != 1:
                raise ValueError(
                    "Exactly one img2img source is supported by this runtime"
                )
            pixels = np.clip(reference[0].detach().cpu().numpy() * 255, 0, 255).astype(
                np.uint8
            )
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            png = buffer.getvalue()
        route = "/v1/images/edits" if png else "/v1/images/generations"
        digest = hashlib.sha256(
            json.dumps(
                {"server": base_url, "route": route, "fields": fields}, sort_keys=True
            ).encode()
            + png
        ).hexdigest()
        out = Path(folder_paths.get_output_directory()) / "last-shift-qwen-image-jobs"
        out.mkdir(parents=True, exist_ok=True)
        receipt_path, image_path = out / (digest + ".json"), out / (digest + ".png")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "completed":
                raise RuntimeError(
                    "Previous synchronous request outcome is unresolved; inspect server log before rerunning"
                )
            if (
                not image_path.exists()
                or hashlib.sha256(image_path.read_bytes()).hexdigest()
                != receipt["image_sha256"]
            ):
                raise RuntimeError(
                    "Cached image missing or changed; preserve receipt and investigate"
                )
        else:
            health = json.loads(request(base_url, "/health"))
            receipt = {
                "status": "pending",
                "request_sha256": digest,
                "fields": fields,
                "route": route,
                "health": health,
                "started": time.time(),
            }
            save_json(receipt_path, receipt)
            if png:
                body, kind = multipart(fields, png, image_field="image")
            else:
                body, kind = json.dumps(fields).encode(), "application/json"
            result = json.loads(request(base_url, route, body, kind, timeout=7200))
            if (
                result.get("cancelled")
                or len(result.get("data", [])) != 1
                or "b64_json" not in result["data"][0]
            ):
                raise RuntimeError("Rapid did not return one completed image")
            raw = base64.b64decode(result["data"][0]["b64_json"], validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                actual_size = im.size
            if actual_size != (width, height):
                raise RuntimeError("Rapid returned an unexpected image size")
            image_path.write_bytes(raw)
            receipt.update(
                status="completed",
                completed=time.time(),
                image_sha256=hashlib.sha256(raw).hexdigest(),
                actual_size=list(actual_size),
                response_metadata={k: v for k, v in result.items() if k != "data"},
            )
            save_json(receipt_path, receipt)
        with Image.open(image_path) as im:
            pixels = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(pixels).unsqueeze(0),)
