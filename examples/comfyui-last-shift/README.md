# The Last Shift

A silent, approximately 30-second science-fiction story. At a deserted Mars base, an obsolete maintenance robot has 1% power left. It notices the final seedling dying in an unheated greenhouse. It gives the greenhouse its own orange power cartridge and shuts down beside the plant. Years later, the greenhouse is flourishing. A white flower grows in its open palm.

The English delivery is `media/final-en/the-last-shift-en.mp4`. All editorial text is English: “MARS · ABANDONED OUTPOST”, “POWER REMAINING: 1%”, “YEARS LATER”, and “THE LAST SHIFT”. The original export remains in `media/final/`.

## Production contract

Keep one recognizable protagonist, one orange cartridge and one greenhouse. Show the cartridge in the chest, then an empty chest and the cartridge in the robot's hand, then the cartridge in the heater. Warm light belongs to the seedling as the robot's eye goes dark. The matching seated compositions make the time jump readable. No dialogue or promotional title cards interrupt the story.

R-07: old ivory enamel, dark mechanical joints, compact rectangular head, one circular amber eye, navy left shoulder triangle, three-finger hands, short sturdy legs. The empty chest and dark eye must persist after shutdown. Preserve the original seated pose in the years-later shot. Final flower: five white petals, golden center, growing from soil in the open right palm.

## Pipeline and provenance

Character reference and nine consistent cinematic keyframes: built-in image generation, reference-conditioned. Motion: leased Apple M4 Pro, 48 GiB, ComfyUI → Rapid-MLX asynchronous video API → LTX-2.5 MLX q8 distilled runtime. The still-image stage is separate from the local motion stage. Native motion target: 1024×512, 65 frames, 24 fps per shot. Delivery scales to a 1920×960 image within a 1920×1080 frame; this is an editorial upscale, not native 1080p inference.

Pinned baseline: Rapid-MLX `146a9a5c6`; LTX runtime `57952288076766abe27dda3a774b2c24f7346977`; ComfyUI `0df64eb242b7c5759c3e86afd5d1846d923b1033`. Model: `MrMofer/ltx-2.5-mlx-q8`, revision `f1b56e7dc89f71a9af2cddac787b89ed22a8b7fc`.

The original sparse piano cue in `score.swift` uses macOS's bundled instrument through AVFoundation offline rendering. Original warning, relay and shutdown cues are synthesized by `sound.py`. Native generated shot audio is retained quietly under the continuous score. No third-party music recording is used.

## Reproduce

Install `custom_nodes/rapid_mlx_last_shift` into ComfyUI's custom nodes directory. Start the existing Rapid-MLX video server and ComfyUI. Put the approved PNGs from `media/art` into ComfyUI/input. Names must match each `storyboard.json` keyframe field, or its shot ID when no override is present. Shots 02 and 05 use revised keyframes that remove unnecessary hand movement. Inspect `workflow-api.json`: LoadImage → RapidMLXLastShift → SaveVideo. This adapter reuses the established persisted-job client; accepted job IDs and hashes survive polling restarts.

```sh
python batch.py --output media/shots
swift score.swift media/score.wav
python sound.py --output media/sound.wav
python render.py --input media/shots --score media/score.wav --sound media/sound.wav --output media/final-en/the-last-shift-en.mp4
```

All model weights use the standard shared HF cache. Media, QA images and inference receipts stay under ignored `media/`; never commit private machine addresses, lease files or tokens. Keep actual selected-shot evidence in `VERIFIED.md` after review.
