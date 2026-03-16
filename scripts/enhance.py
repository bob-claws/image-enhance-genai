#!/usr/bin/env python3
"""Crop + classical upscale + GenAI enhancement ("movie enhance" style).

Pipeline:
  1) Center-crop square region
  2) Classical upscale (ffmpeg Lanczos)
  3) GenAI super-resolution / denoise / deblock / sharpen via nano-banana-pro

This is *not* forensic enhancement; the GenAI step may invent plausible details.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ffprobe_dims(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


def ensure_even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


def find_nano_banana_script() -> Path:
    # Preferred: system-installed OpenClaw skill path.
    p = Path("/opt/homebrew/lib/node_modules/openclaw/skills/nano-banana-pro/scripts/generate_image.py")
    if p.exists():
        return p
    raise FileNotFoundError(
        "Could not find nano-banana-pro generate_image.py at expected path: "
        + str(p)
    )


def build_prompt(mode: str, strength: str) -> str:
    """Return a Nano Banana editing prompt.

    mode:
      - balanced: default restoration/upscale
      - strong: more aggressive "movie enhance" vibe
      - text: avoid hallucinating/rewriting text

    strength still controls aggressiveness within a mode.
    """

    mode = (mode or "balanced").lower().strip()

    if mode == "text":
        base = (
            "Upscale and clarify this image. Unblur it and reduce compression artifacts/noise. "
            "Keep the exact same layout, geometry, and perspective. "
            "Preserve all existing text exactly; do not guess, rewrite, or invent any text. "
            "If text is illegible, keep it illegible rather than hallucinating. "
            "Do not add or remove objects."
        )
    elif mode == "strong":
        base = (
            "Upscale this image to a very high-resolution, crisp, detailed result. "
            "Unblur it strongly and remove compression artifacts. "
            "Keep the exact same composition, perspective, and geometry (no re-posing, no shape changes). "
            "Do not add or remove objects."
        )
    else:  # balanced
        base = (
            "Restore and upscale this image to a high-resolution, crisp result. "
            "Unblur it and reduce noise/JPEG artifacts. "
            "Keep the exact same composition, perspective, layout, and geometry. "
            "Do not add/remove objects or invent text/patterns. "
            "This is restoration, not creative reinterpretation; if uncertain, keep it soft rather than guessing."
        )

    strength = (strength or "med").lower().strip()
    if strength == "low":
        return base + " Minimal reinterpretation; preserve the input as much as possible."
    if strength == "high":
        return base + " Aggressive enhancement of plausible fine texture/detail is OK."
    return base


def detect_annotation_bbox(
    base_path: Path,
    guide_path: Path,
    *,
    diff_thresh: int = 35,
    blur_ksize: int = 3,
) -> tuple[int, int, int, int]:
    """Return bounding box (x, y, w, h) around the largest *annotation* region.

    Robust approach: detect the annotation purely by differencing the clean base image
    and the annotated guide image.

    This makes the annotation color irrelevant and avoids accidental matches with
    scene content.

    Notes:
    - WhatsApp/etc can recompress images; small pixel-level differences may appear.
      We mitigate by optional blur + a threshold.
    - If multiple marks exist, we pick the largest connected component.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    guide = cv2.imread(str(guide_path), cv2.IMREAD_COLOR)
    if base is None:
        raise ValueError(f"Could not read base image: {base_path}")
    if guide is None:
        raise ValueError(f"Could not read guide image: {guide_path}")

    if base.shape[:2] != guide.shape[:2]:
        guide = cv2.resize(guide, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)

    diff = cv2.absdiff(base, guide)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    if blur_ksize and blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        diff_gray = cv2.GaussianBlur(diff_gray, (blur_ksize, blur_ksize), 0)

    _, mask = cv2.threshold(diff_gray, int(diff_thresh), 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No annotation detected (try thicker mark, or lower --diff-thresh)")

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 10 or h < 10:
        raise ValueError("Annotation detected but bounding box is too small")
    return int(x), int(y), int(w), int(h)


def main() -> int:
    ap = argparse.ArgumentParser(description="Crop + upscale + GenAI enhance an image.")
    ap.add_argument("-i", "--input", required=True, help="Input image path")
    ap.add_argument(
        "--crop-frac",
        type=float,
        default=0.30,
        help=(
            "Crop size as fraction of the shorter image dimension (0.1-1.0). "
            "Smaller = more zoom. Default: 0.30"
        ),
    )
    ap.add_argument(
        "--crop",
        default="center",
        help=(
            "Crop location preset. One of: center, top, bottom, left, right, "
            "top-left, top-right, bottom-left, bottom-right. Default: center"
        ),
    )
    ap.add_argument(
        "--crop-x",
        type=float,
        default=None,
        help=(
            "Optional crop center X as fraction of image width (0..1). Overrides --crop preset. "
            "Example: 0.85 for near-right."
        ),
    )
    ap.add_argument(
        "--crop-y",
        type=float,
        default=None,
        help=(
            "Optional crop center Y as fraction of image height (0..1). Overrides --crop preset. "
            "Example: 0.15 for near-top."
        ),
    )
    ap.add_argument(
        "--preupscale",
        type=int,
        default=2,
        help="Classical upscale multiplier before GenAI step. Default: 2",
    )
    ap.add_argument(
        "--guide",
        default=None,
        help=(
            "Optional annotated guide image (same base image) with a red circle around the target. "
            "If provided, the crop is derived from the circle bounding square (overrides --crop/--crop-x/--crop-y)."
        ),
    )
    ap.add_argument(
        "--guide-pad",
        type=float,
        default=1.25,
        help="Padding multiplier applied to the detected annotation bounding box before cropping (default 1.25).",
    )
    ap.add_argument(
        "--diff-thresh",
        type=int,
        default=35,
        help="When using --guide: pixel-diff threshold (0-255) to detect the annotation. Default: 35",
    )
    ap.add_argument(
        "--diff-blur",
        type=int,
        default=3,
        help="When using --guide: Gaussian blur kernel size to suppress compression noise (odd int, 0 disables). Default: 3",
    )
    ap.add_argument(
        "--resolution",
        choices=["1K", "2K", "4K"],
        default="2K",
        help="GenAI output resolution passed to nano-banana-pro. Default: 2K",
    )
    ap.add_argument(
        "--progressive",
        action="store_true",
        help=(
            "Enable 2-stage progressive enhance: do a larger context crop + enhance first, "
            "then crop tighter around the target inside the enhanced image and enhance again. "
            "Best used with --guide."
        ),
    )
    ap.add_argument(
        "--context-pad",
        type=float,
        default=2.0,
        help=(
            "When --progressive + --guide: stage-1 crop side = bbox_side * context_pad. Default 2.0"
        ),
    )
    ap.add_argument(
        "--final-pad",
        type=float,
        default=1.25,
        help=(
            "When --progressive + --guide: stage-2 crop side = bbox_side * final_pad (in original px, mapped into stage-1 output). Default 1.25"
        ),
    )
    ap.add_argument(
        "--prompt-mode",
        choices=["balanced", "strong", "text"],
        default="balanced",
        help="Prompt preset: balanced (default), strong (more aggressive), text (avoid hallucinated text)",
    )
    ap.add_argument(
        "--strength",
        choices=["low", "med", "high"],
        default="med",
        help="How aggressively GenAI may invent plausible detail within the prompt mode. Default: med",
    )
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output PNG path. Default: ./out/<timestamp>-enhanced.png",
    )
    ap.add_argument(
        "--gif",
        action="store_true",
        help=(
            "Also generate an 'ENHANCE' animated GIF of the progressive steps (cropped area only). "
            "When used with --progressive, the GIF shows: original crop -> stage1-informed crop -> final enhance."
        ),
    )
    ap.add_argument(
        "--gif-path",
        default=None,
        help="Output GIF path (default: same as --out but with .gif)",
    )
    ap.add_argument(
        "--gif-fps",
        type=int,
        default=12,
        help="GIF frame rate (default 12)",
    )
    ap.add_argument(
        "--gif-seconds",
        type=float,
        default=3.0,
        help="GIF total duration in seconds (default 3.0)",
    )
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    crop_frac = args.crop_frac
    if not (0.1 <= crop_frac <= 1.0):
        raise ValueError("--crop-frac must be between 0.1 and 1.0")

    out_path = Path(args.out) if args.out else Path("out") / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}-enhanced.png"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = ffprobe_dims(inp)
    min_dim = min(w, h)
    crop_px = ensure_even(max(64, int(min_dim * crop_frac)))

    pre_px = crop_px * int(args.preupscale)

    tmp_dir = out_path.parent / f".tmp-enhance-{int(time.time())}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    crop_path = tmp_dir / "crop-preup.png"

    # Determine crop box (x,y,crop_px) either from a red-circle guide or from presets/coords.
    if args.guide:
        guide = Path(args.guide).expanduser().resolve()
        if not guide.exists():
            raise FileNotFoundError(str(guide))

        gx, gy, gw, gh = detect_annotation_bbox(inp, guide, diff_thresh=args.diff_thresh, blur_ksize=args.diff_blur)

        side = ensure_even(max(64, int(round(max(gw, gh) * float(args.guide_pad)))))

        cx_px = gx + gw / 2
        cy_px = gy + gh / 2

        x = int(round(cx_px - side / 2))
        y = int(round(cy_px - side / 2))

        crop_px = min(side, w, h)
        x = max(0, min(x, w - crop_px))
        y = max(0, min(y, h - crop_px))

    else:
        preset = (args.crop or "center").lower().strip()
        preset_map = {
            "center": (0.5, 0.5),
            "top": (0.5, 0.2),
            "bottom": (0.5, 0.8),
            "left": (0.2, 0.5),
            "right": (0.8, 0.5),
            "top-left": (0.2, 0.2),
            "topleft": (0.2, 0.2),
            "top-right": (0.8, 0.2),
            "topright": (0.8, 0.2),
            "bottom-left": (0.2, 0.8),
            "bottomleft": (0.2, 0.8),
            "bottom-right": (0.8, 0.8),
            "bottomright": (0.8, 0.8),
        }

        if args.crop_x is not None or args.crop_y is not None:
            cx = 0.5 if args.crop_x is None else float(args.crop_x)
            cy = 0.5 if args.crop_y is None else float(args.crop_y)
        else:
            if preset not in preset_map:
                raise ValueError(f"Unknown --crop preset: {args.crop}")
            cx, cy = preset_map[preset]

        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise ValueError("--crop-x/--crop-y must be between 0 and 1")

        x = int(round(cx * w - crop_px / 2))
        y = int(round(cy * h - crop_px / 2))
        x = max(0, min(x, w - crop_px))
        y = max(0, min(y, h - crop_px))

    nano = find_nano_banana_script()
    prompt = build_prompt(args.prompt_mode, args.strength)

    def crop_and_preup(in_path: Path, out_path_png: Path, x0: int, y0: int, side0: int) -> None:
        pre_px0 = side0 * int(args.preupscale)
        vf0 = f"crop={side0}:{side0}:{x0}:{y0},scale={pre_px0}:{pre_px0}:flags=bilinear"
        run(["ffmpeg", "-y", "-i", str(in_path), "-vf", vf0, str(out_path_png)])

    def genai(in_path_png: Path, out_path_png: Path) -> None:
        run([
            "uv",
            "run",
            str(nano),
            "-i",
            str(in_path_png),
            "--prompt",
            prompt,
            "--filename",
            str(out_path_png),
            "--resolution",
            args.resolution,
        ])

    if args.progressive and args.guide:
        # Stage 1: larger context crop around the annotation
        gpath = Path(args.guide).expanduser().resolve()
        gx, gy, gw, gh = detect_annotation_bbox(inp, gpath, diff_thresh=args.diff_thresh, blur_ksize=args.diff_blur)
        bbox_side = max(gw, gh)
        side1 = ensure_even(max(64, int(round(bbox_side * float(args.context_pad)))))
        cx_px = gx + gw / 2
        cy_px = gy + gh / 2
        x1 = int(round(cx_px - side1 / 2))
        y1 = int(round(cy_px - side1 / 2))
        side1 = min(side1, w, h)
        x1 = max(0, min(x1, w - side1))
        y1 = max(0, min(y1, h - side1))

        stage1_pre = tmp_dir / "stage1-preup.png"
        stage1_out = tmp_dir / "stage1-gen.png"
        crop_and_preup(inp, stage1_pre, x1, y1, side1)
        genai(stage1_pre, stage1_out)

        # Stage 2: tighter crop, mapped into stage-1 output coordinates
        out_w, out_h = ffprobe_dims(stage1_out)
        out_side = min(out_w, out_h)
        scale = out_side / side1

        cx_local = cx_px - x1
        cy_local = cy_px - y1
        cx2 = cx_local * scale
        cy2 = cy_local * scale

        side2 = ensure_even(max(64, int(round(bbox_side * float(args.final_pad) * scale))))
        side2 = min(side2, out_side)
        x2 = int(round(cx2 - side2 / 2))
        y2 = int(round(cy2 - side2 / 2))
        x2 = max(0, min(x2, out_side - side2))
        y2 = max(0, min(y2, out_side - side2))

        stage2_pre = tmp_dir / "stage2-preup.png"
        crop_and_preup(stage1_out, stage2_pre, x2, y2, side2)
        genai(stage2_pre, out_path)

    else:
        # Single-stage
        crop_and_preup(inp, crop_path, x, y, crop_px)
        genai(crop_path, out_path)

    # Optional GIF generation (cropped area only)
    if args.gif:
        gif_path = Path(args.gif_path) if args.gif_path else out_path.with_suffix(".gif")
        gif_path = gif_path.resolve()
        gif_path.parent.mkdir(parents=True, exist_ok=True)

        # Build 3-frame "progressive enhance" animation when possible
        frames: list[Path] = []

        # For progressive+guide we can show original crop -> stage2_pre -> final
        if args.progressive and args.guide:
            # Recompute bbox + center in base space
            gpath = Path(args.guide).expanduser().resolve()
            gx, gy, gw, gh = detect_annotation_bbox(inp, gpath, diff_thresh=args.diff_thresh, blur_ksize=args.diff_blur)
            bbox_side = max(gw, gh)
            cx_px = gx + gw / 2
            cy_px = gy + gh / 2

            # Original-space "final" crop (before any GenAI), padded with final_pad
            side0 = ensure_even(max(64, int(round(bbox_side * float(args.final_pad)))))
            side0 = min(side0, w, h)
            x0 = int(round(cx_px - side0 / 2))
            y0 = int(round(cy_px - side0 / 2))
            x0 = max(0, min(x0, w - side0))
            y0 = max(0, min(y0, h - side0))

            frame0 = tmp_dir / "gif-0-orig.png"
            crop_and_preup(inp, frame0, x0, y0, side0)

            # Stage2 pre-upscale (already exists in tmp_dir when progressive+guide)
            frame1 = tmp_dir / "stage2-preup.png"

            frames = [frame0, frame1, out_path]
        else:
            # Fallback: just blink from the single-stage preup crop to final
            frame0 = crop_path
            frames = [frame0, out_path]

        # Normalize all frames to the same size (use the smallest side among them)
        dims = [ffprobe_dims(p) for p in frames]
        side = min(min(w0, h0) for (w0, h0) in dims)
        side = max(256, min(side, 1024))  # keep GIF reasonable

        norm_dir = tmp_dir / "gif-frames"
        norm_dir.mkdir(parents=True, exist_ok=True)
        norm_frames: list[Path] = []
        for i, fp in enumerate(frames):
            out_fp = norm_dir / f"f{i:02d}.png"
            run(["ffmpeg", "-y", "-i", str(fp), "-vf", f"scale={side}:{side}:flags=bilinear", str(out_fp)])
            norm_frames.append(out_fp)

        # Build a short mp4 with crossfades, then convert to GIF with palette
        mp4_path = tmp_dir / "enhance.mp4"
        palette = tmp_dir / "palette.png"

        # Each still shown for (gif_seconds / len(frames)) seconds, with small crossfade
        n = len(norm_frames)
        seg = max(0.5, float(args.gif_seconds) / n)
        fade = min(0.25, seg * 0.35)

        # Create inputs
        ff = ["ffmpeg", "-y"]
        for nf in norm_frames:
            ff += ["-loop", "1", "-t", f"{seg}", "-i", str(nf)]

        # xfade chain
        # offsets are where each transition starts in the accumulating timeline
        filt = ""
        if n == 2:
            filt = f"[0:v][1:v]xfade=transition=fade:duration={fade}:offset={seg-fade},format=yuv420p[v]"
        else:
            # [0][1] -> v01, then v01 with [2] ...
            offset1 = seg - fade
            filt = f"[0:v][1:v]xfade=transition=fade:duration={fade}:offset={offset1}[v01];"
            cur = "v01"
            for idx in range(2, n):
                # timeline length grows by seg each time, but overlap by fade
                offset = idx * seg - fade * (idx)
                nxt = f"v{idx:02d}"
                filt += f"[{cur}][{idx}:v]xfade=transition=fade:duration={fade}:offset={offset}[{nxt}];"
                cur = nxt
            filt += f"[{cur}]format=yuv420p[v]"

        ff += ["-filter_complex", filt, "-map", "[v]", "-r", str(args.gif_fps), str(mp4_path)]
        run(ff)

        # Palette + GIF
        fps = max(6, int(args.gif_fps))
        run(["ffmpeg", "-y", "-i", str(mp4_path), "-vf", f"fps={fps},scale=640:-1:flags=lanczos,palettegen", str(palette)])
        run([
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-i",
            str(palette),
            "-lavfi",
            f"fps={fps},scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(gif_path),
        ])

        print(f"MEDIA:{gif_path}")

    # Best-effort cleanup
    try:
        if not args.gif:
            for p in tmp_dir.glob("*"):
                p.unlink(missing_ok=True)
            tmp_dir.rmdir()
    except Exception:
        pass

    # Print for OpenClaw to auto-attach on supported surfaces
    print(f"MEDIA:{out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        cmd = " ".join(shlex.quote(c) for c in e.cmd) if isinstance(e.cmd, (list, tuple)) else str(e.cmd)
        print(f"ERROR: command failed (exit {e.returncode}): {cmd}", file=sys.stderr)
        return_code = e.returncode or 1
        raise SystemExit(return_code)
