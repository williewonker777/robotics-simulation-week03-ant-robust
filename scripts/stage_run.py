#!/usr/bin/env python3
"""Copy one validated training run into the submission artifact tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("label")
    parser.add_argument("variant", choices=("baseline", "friction", "robust", "terrain"))
    parser.add_argument("seed", type=int)
    parser.add_argument("--checkpoint", default="model_999.pt")
    parser.add_argument("--destination", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()

    source = args.source.resolve()
    destination = (args.destination / args.label).resolve()
    checkpoint = source / args.checkpoint
    event_files = sorted(source.glob("events.out.tfevents.*"))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event file in {source}")

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination / checkpoint.name)
    for event_file in event_files:
        shutil.copy2(event_file, destination / event_file.name)
    if (source / "params").is_dir():
        shutil.copytree(source / "params", destination / "params", dirs_exist_ok=True)

    staged_checkpoint = destination / checkpoint.name
    manifest = {
        "label": args.label,
        "variant": args.variant,
        "seed": args.seed,
        "source_run": str(source),
        "checkpoint": staged_checkpoint.name,
        "checkpoint_bytes": staged_checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(staged_checkpoint),
        "event_files": [path.name for path in event_files],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
