"""Uncut, predeclared seed42 vs v5 comparison; never added to benchmark counts."""

import argparse
import fcntl
import json
import subprocess

from run_foothold_experiment import ROOT, WORK, ARTIFACT, PYTHON, check_sources, run, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    frozen = json.loads((ARTIFACT / "frozen.json").read_text())
    check_sources(frozen["source_sha256"])
    if not (ARTIFACT / "verification.json").exists():
        raise ValueError("audit primary evaluation before making comparison videos")
    candidate = next(r for r in frozen["runs"] if r["group"] == "guided" and r["training_seed"] == 42)
    videos = []
    with (WORK / "gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for record, name in ((frozen["reference"], "frozen_v5_stones10"),
                             (candidate, "guided_seed42_stones10")):
            checkpoint = ROOT / record["checkpoint"]
            if sha(checkpoint) != record["sha256"]:
                raise ValueError("frozen checkpoint changed")
            output = ARTIFACT / "videos" / f"{name}.mp4"
            if output.exists():
                raise FileExistsError(output)
            command = [str(PYTHON), "scripts/foothold_v9.py", "demo", "--task", "Week03-Ant-Foothold-Lanes-Demo-v9",
                       "--headless", "--device", args.device, "--seed", "38", "--family", "stepping_stones",
                       "--level", "4", "--cycle-seconds", "16", "--checkpoint", str(checkpoint), "--record", str(output),
                       "env.scene.terrain.terrain_generator.seed=64"]
            run(command, f"video_{name}")
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)],
                                   check=True, capture_output=True, text=True)
            metadata = json.loads(probe.stdout)
            stream = metadata["streams"][0]
            if (int(stream["nb_frames"]), stream["r_frame_rate"], stream["width"], stream["height"]) != (480, "30/1", 1280, 720):
                raise ValueError("unexpected video duration/frame format")
            run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(output), "-f", "null", "-"], f"decode_{name}")
            videos.append({"file": str(output.relative_to(ROOT)), "sha256": sha(output), "checkpoint": record,
                           "command": command, "metadata": metadata, "full_decode_passed": True})
    write_json(ARTIFACT / "video_manifest.json", {
        "geometry": 64, "reset": 38, "family": "stepping_stones", "difficulty": 1.0,
        "selection": "predeclared seed42; uncut16s including any failures/resets; no selection by results",
        "interpretation": "Separate one-environment qualitative rollouts, not primary benchmark episodes",
        "videos": videos,
    })


if __name__ == "__main__":
    main()
