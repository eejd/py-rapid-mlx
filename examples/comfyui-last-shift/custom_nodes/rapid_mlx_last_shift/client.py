# SPDX-License-Identifier: Apache-2.0
"""Standard-library video client; importing this module never loads a model."""

import hashlib
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path


def request(base, route, data=None, content_type=None, timeout=120):
    headers = {"X-Rapid-Client": "comfyui-cinema"}
    if content_type:
        headers["Content-Type"] = content_type
    if key := os.environ.get("RAPID_MLX_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + route, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def multipart(fields, png, image_field="input_reference"):
    boundary = "rapid" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.extend(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="{image_field}"; filename="keyframe.png"\r\nContent-Type: image/png\r\n\r\n'.encode(),
            png,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(parts), "multipart/form-data; boundary=" + boundary


def save_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def generate(base, fields, png, output, timeout=7200, check_cancel=lambda: None):
    """Persist job IDs before polling; identical reruns resume instead of submitting."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        json.dumps(
            {"server": base.rstrip("/"), "fields": fields}, sort_keys=True
        ).encode()
        + png
    ).hexdigest()
    receipt_path = output / (fingerprint + ".json")
    target = output / (fingerprint + ".mp4")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
    else:
        body, kind = multipart(fields, png)
        # If this POST loses its response, inspect server jobs before retrying.
        job = json.loads(request(base, "/v1/videos", body, kind))
        receipt = {"job_id": job["id"], "request_sha256": fingerprint, "fields": fields}
        save_json(receipt_path, receipt)
    if target.exists() and receipt.get("video_sha256"):
        if hashlib.sha256(target.read_bytes()).hexdigest() != receipt["video_sha256"]:
            raise RuntimeError(
                "Saved video hash mismatch; preserve receipt and investigate."
            )
        return target
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_cancel()
        job = json.loads(request(base, "/v1/videos/" + receipt["job_id"]))
        receipt["job"] = job
        save_json(receipt_path, receipt)
        status = job["status"]
        if status == "completed":
            data = request(base, "/v1/videos/" + receipt["job_id"] + "/content")
            if not data:
                raise RuntimeError("Server returned an empty video.")
            temporary = target.with_suffix(".mp4.tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
            receipt["video_sha256"] = hashlib.sha256(data).hexdigest()
            save_json(receipt_path, receipt)
            return target
        if status in ("failed", "cancelled"):
            raise RuntimeError(
                f"Video job {receipt['job_id']} {status}: {job.get('error')}"
            )
        time.sleep(2)
    raise TimeoutError(
        f"Job {receipt['job_id']} remains recorded in {receipt_path}; rerun to resume polling."
    )
