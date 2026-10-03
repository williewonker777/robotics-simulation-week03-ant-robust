#!/usr/bin/env python3
"""Compose v28 demo media with the ffmpeg bundled in imageio-ffmpeg.

* ``pair``: two 16 s rollouts side by side (same terrain, same start), 1920x540.
* ``grid``: up to six rollouts in a 3x2 grid, 1920x720.
* ``gif`` : a light README preview (palette-optimised GIF) of any MP4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: list[str]) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def pair(left: Path, right: Path, out: Path) -> None:
    run(["-i", str(left), "-i", str(right), "-filter_complex",
         "[0:v]scale=960:540[l];[1:v]scale=960:540[r];[l][r]hstack=inputs=2[v]",
         "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", str(out)])


def grid(inputs: list[Path], out: Path) -> None:
    if not 1 <= len(inputs) <= 6:
        raise ValueError("grid takes 1-6 inputs")
    args, scaled = [], []
    for index, path in enumerate(inputs):
        args += ["-i", str(path)]
        scaled.append(f"[{index}:v]scale=640:360[v{index}]")
    layout = "|".join(["0_0", "640_0", "1280_0", "0_360", "640_360", "1280_360"][: len(inputs)])
    names = "".join(f"[v{index}]" for index in range(len(inputs)))
    filt = ";".join(scaled) + f";{names}xstack=inputs={len(inputs)}:layout={layout}:fill=black[v]"
    run([*args, "-filter_complex", filt, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "27", "-preset", "slow", str(out)])


def gif(source: Path, out: Path, width: int, fps: int, start: float, duration: float) -> None:
    filt = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    run(["-ss", str(start), "-t", str(duration), "-i", str(source), "-filter_complex", filt, str(out)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("pair")
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("out", type=Path)
    g = sub.add_parser("grid")
    g.add_argument("out", type=Path)
    g.add_argument("inputs", type=Path, nargs="+")
    f = sub.add_parser("gif")
    f.add_argument("source", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--width", type=int, default=720)
    f.add_argument("--fps", type=int, default=10)
    f.add_argument("--start", type=float, default=0.0)
    f.add_argument("--duration", type=float, default=16.0)
    args = parser.parse_args()
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "pair":
        pair(args.left, args.right, out)
        sources = [args.left, args.right]
    elif args.mode == "grid":
        grid(args.inputs, out)
        sources = args.inputs
    else:
        gif(args.source, out, args.width, args.fps, args.start, args.duration)
        sources = [args.source]
    print(json.dumps({"output": str(out), "sha256": sha256(out), "bytes": out.stat().st_size,
                      "sources": [str(s) for s in sources]}))


if __name__ == "__main__":
    main()
