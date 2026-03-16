---
name: image-enhance-genai
description: Movie-style "enhance" for images: center-crop + classical upscale + GenAI super-resolution/denoise/sharpen to produce a crisp, high-resolution result. Use when a user asks to "enhance", "super-res", "zoom and sharpen", "make it HD", or wants the funny/bullshit hacker-movie ENHANCE vibe (not for forensics/evidence).
metadata:
  openclaw:
    requires:
      bins: ["ffmpeg", "ffprobe", "python3", "uv"]
      env: ["GEMINI_API_KEY"]
    primaryEnv: GEMINI_API_KEY
---

# Image Enhance (GenAI)

Run crop + classical upscale (Lanczos) + a GenAI "crisp/detailed" pass.

Important: the GenAI step may invent plausible detail. This is aesthetic, not forensic.

## Quick start

```bash
# From repo/workspace root
# (Use --project so dependencies like OpenCV are available for --guide mode)
uv run --project ./skills/image-enhance-genai ./skills/image-enhance-genai/scripts/enhance.py \
  -i /path/to/input.jpg \
  --crop-frac 0.30 \
  --strength med \
  --resolution 2K
```

The script prints a `MEDIA:` line pointing at the output PNG.

## Parameters

- `--crop-frac` (0.1–1.0): fraction of the shorter side to keep in a square crop.
  - smaller = more zoom (e.g. 0.25 is tighter than 0.35)
- `--crop`: crop location preset:
  - `center` (default)
  - `top`, `bottom`, `left`, `right`
  - `top-left`, `top-right`, `bottom-left`, `bottom-right`
- `--crop-x` / `--crop-y` (0..1): set the crop **center** as fractions of width/height.
  - overrides `--crop`
  - example: `--crop-x 0.85 --crop-y 0.15` (top-right-ish)
- `--guide /path/to/annotated.png`: optional *annotated* version of the image with a **red circle** around the target.
  - overrides `--crop`, `--crop-x`, `--crop-y`
- `--guide-pad` (float): padding multiplier around the detected red circle box (default 1.25)
- `--progressive`: 2-stage enhance (recommended with `--guide`) to keep more context:
  1) crop a larger context square around the red circle → enhance
  2) crop tighter around the same target inside the enhanced image → enhance again
- `--context-pad` / `--final-pad`: control stage-1 vs stage-2 crop sizes (multipliers of the red-circle bbox)
- `--prompt-mode` (`balanced|strong|text`): choose the prompt preset
  - `balanced`: restoration/upscale with anti-hallucination language (default)
  - `strong`: more aggressive “movie enhance” vibe
  - `text`: tries hard not to guess/alter text
- `--strength` (`low|med|high`): how aggressively GenAI may reinterpret/invent micro-detail within the chosen prompt mode
- `--resolution` (`1K|2K|4K`): output resolution for the GenAI step
- `--preupscale` (int): classical upscale multiplier before GenAI (default 2)
- `-o/--out`: output path (default `./out/<timestamp>-enhanced.png`)

## Implementation notes (for maintainers)

- Crop + pre-upscale is done with ffmpeg (`crop`, `scale=...:flags=bilinear`).
- GenAI pass uses the existing OpenClaw `nano-banana-pro` script:
  `/opt/homebrew/lib/node_modules/openclaw/skills/nano-banana-pro/scripts/generate_image.py`
