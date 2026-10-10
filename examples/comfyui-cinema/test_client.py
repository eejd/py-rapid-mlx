# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "client", Path(__file__).parent / "custom_nodes/rapid_mlx_video/client.py"
)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class VideoJobs(unittest.TestCase):
    def test_completed_rerun_does_not_submit(self):
        with tempfile.TemporaryDirectory() as output:
            responses = [b'{"id":"video_1"}', b'{"status":"completed"}', b"mp4-bytes"]
            with patch.object(client, "request", side_effect=responses) as request:
                target = client.generate(
                    "http://localhost:18428", {"seed": 1}, b"png", output
                )
                self.assertEqual(target.read_bytes(), b"mp4-bytes")
                self.assertEqual(request.call_count, 3)
            with patch.object(client, "request") as request:
                self.assertEqual(
                    client.generate(
                        "http://localhost:18428", {"seed": 1}, b"png", output
                    ),
                    target,
                )
                request.assert_not_called()
            target.write_bytes(b"corrupted")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                client.generate("http://localhost:18428", {"seed": 1}, b"png", output)

    def test_disconnect_resumes_existing_job(self):
        with tempfile.TemporaryDirectory() as output:
            with (
                patch.object(
                    client,
                    "request",
                    side_effect=[b'{"id":"video_2"}', ConnectionError()],
                ),
                self.assertRaises(ConnectionError),
            ):
                client.generate("http://localhost", {}, b"png", output)
            receipt = json.loads(next(Path(output).glob("*.json")).read_text())
            self.assertEqual(receipt["job_id"], "video_2")
            with patch.object(
                client, "request", side_effect=[b'{"status":"completed"}', b"video"]
            ) as request:
                client.generate("http://localhost", {}, b"png", output)
                self.assertTrue(
                    all(call.args[2:] == () for call in request.call_args_list)
                )
                self.assertEqual(
                    request.call_args_list[0].args[1], "/v1/videos/video_2"
                )

    def test_failed_job_never_downloads(self):
        with (
            tempfile.TemporaryDirectory() as output,
            patch.object(
                client,
                "request",
                side_effect=[b'{"id":"video_3"}', b'{"status":"failed","error":"oom"}'],
            ) as request,
        ):
            with self.assertRaisesRegex(RuntimeError, "failed.*oom"):
                client.generate("http://localhost", {}, b"png", output)
            self.assertEqual(request.call_count, 2)

    def test_multipart_preserves_reference(self):
        data, kind = client.multipart({"frames": 97, "fps": 24}, b"PNG\x00binary")
        boundary = kind.split("boundary=")[1].encode()
        self.assertIn(b'name="input_reference"; filename="keyframe.png"', data)
        self.assertIn(b"PNG\x00binary", data)
        self.assertTrue(data.endswith(b"--" + boundary + b"--\r\n"))


if __name__ == "__main__":
    unittest.main()
