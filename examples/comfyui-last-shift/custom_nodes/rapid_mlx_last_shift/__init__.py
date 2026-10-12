# SPDX-License-Identifier: Apache-2.0
"""Local image, motion and editorial nodes for The Last Shift."""

from .client import generate
from .post import RapidMLXLastShiftEdit
from .qwen import RapidMLXQwenImage


class RapidMLXLastShift:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE",),
                "base_url": ("STRING", {"default": "http://127.0.0.1:18428"}),
                "prompt": ("STRING", {"multiline": True}),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 1024, "step": 64},
                ),
                "height": ("INT", {"default": 512, "min": 256, "max": 512, "step": 64}),
                "frames": ("INT", {"default": 65, "min": 9, "max": 65, "step": 8}),
                "seed": ("INT", {"default": 7300, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate"
    CATEGORY = "Rapid-MLX"

    def generate(self, image, base_url, prompt, width, height, frames, seed):
        import io
        from pathlib import Path

        import folder_paths
        import numpy as np
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy_api.latest import InputImpl
        from PIL import Image

        if len(image) != 1:
            raise ValueError(
                "One keyframe per shot is required; queue shots sequentially."
            )
        pixels = np.clip(image[0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        fields = {
            "model": "ltx-2.5-mlx-q8",
            "prompt": prompt,
            "size": f"{width}x{height}",
            "frames": frames,
            "fps": 24,
            "seed": seed,
        }
        path = generate(
            base_url,
            fields,
            buffer.getvalue(),
            Path(folder_paths.get_output_directory()) / "last-shift-jobs",
            check_cancel=throw_exception_if_processing_interrupted,
        )
        return (InputImpl.VideoFromFile(str(path)),)


NODE_CLASS_MAPPINGS = {
    "RapidMLXLastShift": RapidMLXLastShift,
    "RapidMLXQwenImage": RapidMLXQwenImage,
    "RapidMLXLastShiftEdit": RapidMLXLastShiftEdit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RapidMLXLastShift": "Rapid-MLX · Last Shift Shot",
    "RapidMLXQwenImage": "Rapid-MLX · Qwen Image 2.1",
    "RapidMLXLastShiftEdit": "Rapid-MLX · Last Shift Edit",
}
