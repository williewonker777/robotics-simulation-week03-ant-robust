#!/usr/bin/env python3
"""Stage chosen v28 runs into ``artifacts/combo_v28/`` for publication (CPU only).

For every run name given with ``--runs``:

* ``runs/<run>/model_<iter>.pt`` - the final checkpoint, byte-identical (cached and O_DIRECT SHA-256
  must both equal the SHA recorded right after training);
* ``runs/<run>/params/{agent,env}.yaml`` and ``runs/<run>/run.json`` - with this machine's absolute
  repository path replaced by a repository-relative one (the only change);
* ``runs/<run>/curves.csv`` - per-iteration training scalars read from the TensorBoard event file
  (the event file itself stays local, like every earlier run in ``artifacts/runs/``);
* ``runs/<run>/manifest.json`` - checkpoint SHA, recipe, seed and the parent checkpoint SHA.

``--submission`` also copies one run's checkpoint to ``submission/`` (the path used by the README
evaluation command), ``--plot`` draws the training curves of the published runs and ``--media`` copies the
composed videos/GIF previews of ``outputs/combo_v28/media`` with a manifest of every source clip.
``publication_metadata.json`` lists original and published SHA-256 of every staged file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "combo_v28"
ART = ROOT / "artifacts" / "combo_v28"
PREFIX = str(ROOT) + "/"

CURVE_TAGS = {
    "mean_reward": "Train/mean_reward",
    "mean_episode_length": "Train/mean_episode_length",
    "fall_share": "Episode_Termination/torso_height",
    "time_out_share": "Episode_Termination/time_out",
    "action_noise_std": "Policy/mean_noise_std",
    "learning_rate": "Loss/learning_rate",
    "value_loss": "Loss/value_function",
    "surrogate_loss": "Loss/surrogate",
    "entropy": "Loss/entropy",
}


def _runner():
    spec = importlib.util.spec_from_file_location("run_combo_v28", ROOT / "scripts" / "run_combo_v28.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def project(text: str) -> str:
    """Repository-absolute paths of this machine -> repository-relative paths."""
    return text.replace(PREFIX, "")


def split_run(run_name: str) -> tuple[str, int, str]:
    """``v28s2_lim_e5_d1+stock_s43`` -> (``lim_e5_d1+stock``, 43, ``stage2``)."""
    stage, rest = run_name.split("_", 1)
    recipe, seed = rest.rsplit("_s", 1)
    return recipe, int(seed), {"v28s1": "stage1", "v28s2": "stage2"}[stage]


def read_curves(run_dir: Path) -> list[dict]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = sorted(run_dir.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise SystemExit(f"expected one event file in {run_dir}, found {len(events)}")
    accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    columns = {name: {event.step: event.value for event in accumulator.Scalars(tag)} for name, tag in CURVE_TAGS.items()}
    steps = sorted(columns["mean_reward"])
    return [{"iteration": step, **{name: columns[name].get(step) for name in CURVE_TAGS}} for step in steps]


def stage_run(run_name: str, runner, files: list[dict]) -> dict:
    record_path = OUT / "run_records" / f"{run_name}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    final = Path(record["final_checkpoint"]["path"])
    run_dir = Path(record["run_dir"])
    expected = record["final_checkpoint"]["sha256"]
    if not (runner.sha256(final) == runner.direct_sha256(final) == expected):
        raise SystemExit(f"checkpoint SHA mismatch before copy: {final}")

    dest = ART / "runs" / run_name
    (dest / "params").mkdir(parents=True, exist_ok=True)
    model = dest / final.name
    shutil.copyfile(final, model)
    if not (runner.sha256(model) == runner.direct_sha256(model) == expected):
        raise SystemExit(f"checkpoint SHA mismatch after copy: {model}")
    files.append({"path": project(str(model)), "original_sha256": expected, "published_sha256": expected,
                  "transformed": False})

    for name in ("agent.yaml", "env.yaml"):
        original = (run_dir / "params" / name).read_bytes()
        published = project(original.decode("utf-8")).encode("utf-8")
        (dest / "params" / name).write_bytes(published)
        files.append({"path": project(str(dest / "params" / name)), "original_sha256": sha256_bytes(original),
                      "published_sha256": sha256_bytes(published), "transformed": original != published})

    original = record_path.read_bytes()
    published = project(original.decode("utf-8")).encode("utf-8")
    (dest / "run.json").write_bytes(published)
    files.append({"path": project(str(dest / "run.json")), "original_sha256": sha256_bytes(original),
                  "published_sha256": sha256_bytes(published), "transformed": original != published})

    curves = read_curves(run_dir)
    if len(curves) != record["max_iterations"]:
        raise SystemExit(f"{run_name}: {len(curves)} curve rows for {record['max_iterations']} iterations")
    with (dest / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["iteration", *CURVE_TAGS], lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)
    files.append({"path": project(str(dest / "curves.csv")), "original_sha256": None,
                  "published_sha256": sha256_bytes((dest / "curves.csv").read_bytes()), "transformed": False,
                  "derived_from": "TensorBoard event file of the run (kept local)"})

    recipe, seed, stage = split_run(run_name)
    init = record.get("init_checkpoint") or {}
    manifest = {
        "run": run_name,
        "stage": stage,
        "recipe": recipe,
        "seed": seed,
        "task": record["task"],
        "entropy_coef": record["entropy_coef"],
        "iterations": record["max_iterations"],
        "transitions": record["transitions"],
        "checkpoint": final.name,
        "checkpoint_bytes": record["final_checkpoint"]["bytes"],
        "checkpoint_sha256": expected,
        "direct_read_sha256_match": record.get("direct_sha256_match"),
        "parent_checkpoint_sha256": init.get("sha256"),
        "parent_run": f"v28s1_{recipe.split('+')[0]}_s{seed}" if stage == "stage2" else None,
        "source_run": project(str(run_dir)),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files.append({"path": project(str(dest / "manifest.json")), "original_sha256": None,
                  "published_sha256": sha256_bytes((dest / "manifest.json").read_bytes()), "transformed": False})
    return manifest


def stage_submission(run_name: str, runner, files: list[dict]) -> dict:
    source_dir = ART / "runs" / run_name
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    dest = ART / "submission"
    dest.mkdir(parents=True, exist_ok=True)
    model = dest / manifest["checkpoint"]
    shutil.copyfile(source_dir / manifest["checkpoint"], model)
    if not (runner.sha256(model) == runner.direct_sha256(model) == manifest["checkpoint_sha256"]):
        raise SystemExit("submission checkpoint SHA mismatch")
    shutil.copytree(source_dir / "params", dest / "params", dirs_exist_ok=True)
    submission = dict(manifest, submission=True, published_run_dir=project(str(source_dir)))
    (dest / "manifest.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    for path in [model, dest / "params" / "agent.yaml", dest / "params" / "env.yaml", dest / "manifest.json"]:
        files.append({"path": project(str(path)), "original_sha256": None,
                      "published_sha256": sha256_bytes(path.read_bytes()), "transformed": False,
                      "copy_of": project(str(source_dir))})
    return submission


def lineage(manifests: list[dict], submission: str) -> list[dict]:
    """The submission recipe's runs plus, for a continuation, its stage-1 parent runs."""
    recipe = next(manifest["recipe"] for manifest in manifests if manifest["run"] == submission)
    wanted = {recipe, recipe.split("+")[0]}
    return [manifest for manifest in manifests if manifest["recipe"] in wanted]


def plot_curves(manifests: list[dict], path: Path) -> None:
    """Mean training reward and fall share per seed (thin) and their mean (bold), one panel each.

    ``manifests`` must be one lineage (see ``lineage``): stage-2 rows are drawn after iteration 1000.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from combo_v28_plots import ACCENT, GRID, INK, INK2, OTHER, plt  # noqa: E402

    by_recipe: dict[str, dict[int, list[dict]]] = {}
    for manifest in manifests:
        rows = list(csv.DictReader((ART / "runs" / manifest["run"] / "curves.csv").open(encoding="utf-8")))
        offset = 1000 if manifest["stage"] == "stage2" else 0
        series = by_recipe.setdefault(manifest["recipe"].split("+")[0], {}).setdefault(manifest["seed"], [])
        series += [dict(row, iteration=int(row["iteration"]) + offset) for row in rows]
    panels = [("mean_reward", "학습 평균 episode 보상"), ("fall_share", "넘어짐으로 끝난 episode 비율")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (key, title) in zip(axes, panels):
        for recipe, seeds in by_recipe.items():
            curves = []
            for seed, rows in sorted(seeds.items()):
                rows = sorted(rows, key=lambda row: row["iteration"])
                x = [row["iteration"] for row in rows]
                y = [float(row[key]) for row in rows]
                ax.plot(x, y, color=OTHER, linewidth=1.0, zorder=2)
                curves.append((x, y))
            length = min(len(y) for _, y in curves)
            mean = [sum(y[i] for _, y in curves) / len(curves) for i in range(length)]
            ax.plot(curves[0][0][:length], mean, color=ACCENT, linewidth=2.0, zorder=3)
        if any(manifest["stage"] == "stage2" for manifest in manifests):
            ax.axvline(1000, color=INK2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
            ax.text(1010, ax.get_ylim()[1], " +600 it 이어 학습", va="top", fontsize=8.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("iteration (4,096 env × 32 step)")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    from matplotlib.lines import Line2D  # noqa: E402

    axes[0].legend(handles=[Line2D([], [], color=OTHER, linewidth=1.0, label="seed별"),
                            Line2D([], [], color=ACCENT, linewidth=2.0, label="3 seed 평균")],
                   frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def stage_media(files: list[dict]) -> dict:
    """Composed MP4/GIF files -> ``media/``; per-clip metadata (paths made relative) -> ``media/manifest.json``."""
    source = OUT / "media"
    dest = ART / "media"
    dest.mkdir(parents=True, exist_ok=True)
    published = []
    for path in sorted(list(source.glob("*.mp4")) + list(source.glob("*.gif"))):
        shutil.copyfile(path, dest / path.name)
        digest = sha256_bytes((dest / path.name).read_bytes())
        if digest != sha256_bytes(path.read_bytes()):
            raise SystemExit(f"media copy mismatch: {path}")
        published.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
        files.append({"path": project(str(dest / path.name)), "original_sha256": digest, "published_sha256": digest,
                      "transformed": False})
    clips = []
    for meta in sorted((source / "clips").glob("*.json")):
        record = json.loads(project(meta.read_text(encoding="utf-8")))
        clips.append(record)
    manifest = {
        "rule": "Week03-Ant-Combo-v28-Play, terrain seed 2028, evaluation seed 24, env 0 of 16, full 16 s at 30 fps; "
                "after the first episode ends the robot is reset automatically and the caption keeps the first-episode return",
        "composed": published,
        "clips": clips,
        "note": "Qualitative demos; scores come from the 100-environment evaluation. Source clips (1280x720) stay local.",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files.append({"path": project(str(dest / "manifest.json")), "original_sha256": None,
                  "published_sha256": sha256_bytes((dest / "manifest.json").read_bytes()), "transformed": False})
    return manifest


def main() -> None:
    global ART
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run names, e.g. v28s1_lim_e5_d1_s42")
    parser.add_argument("--submission", default=None, help="one of --runs; copied to submission/")
    parser.add_argument("--plot", action="store_true", help="draw plots/training_curves.png (needs --submission)")
    parser.add_argument("--media", action="store_true", help="copy composed videos and write media/manifest.json")
    parser.add_argument("--art", type=Path, default=ART, help="output folder (default artifacts/combo_v28)")
    args = parser.parse_args()
    ART = args.art
    if args.submission and args.submission not in args.runs:
        raise SystemExit("--submission must be one of --runs")

    runner = _runner()
    files: list[dict] = []
    manifests = [stage_run(run, runner, files) for run in args.runs]
    submission = stage_submission(args.submission, runner, files) if args.submission else None
    if args.plot and args.submission:
        plot_curves(lineage(manifests, args.submission), ART / "plots" / "training_curves.png")
    media = stage_media(files) if args.media else None
    metadata = {
        "schema": "week03_ant_combo_v28_publication_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "projection": "Only the absolute repository path prefix was removed from text files; checkpoint bytes are unchanged.",
        "runs": [manifest["run"] for manifest in manifests],
        "submission": submission["run"] if submission else None,
        "media": [item["file"] for item in media["composed"]] if media else [],
        "files": files,
    }
    (ART / "publication_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for manifest in manifests:
        print(f"{manifest['run']:<36s} {manifest['checkpoint_sha256'][:16]}  {manifest['stage']}")
    if submission:
        print(f"submission: {submission['run']} -> artifacts/combo_v28/submission/{submission['checkpoint']}")


if __name__ == "__main__":
    main()
