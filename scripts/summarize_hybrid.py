"""Recompute strict v11 metrics from first-episode arrays, never saved aggregates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import struct


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(data):
    if data.get("schema") != "week03_ant_depth_switch_v11_v1":
        raise ValueError("unexpected v11 schema")
    if data.get("mode") not in ("v5", "v10", "hybrid"):
        raise ValueError("unknown policy mode")
    n = data["num_envs"]
    condition = data["condition"]
    if type(n) is not int or n <= 0 or condition["seconds"] not in (16, 64):
        raise ValueError("invalid episode contract")
    if (condition["one_threshold"], condition["six_threshold"], condition["footprint_margin"],
            condition["observations"], condition["v5_observations"], condition["actions"]) != (13.1, 53.1, 1.1, 88, 60, 8):
        raise ValueError("changed benchmark contract")
    if condition["max_steps"] != condition["seconds"] * 60 or not math.isclose(condition["dt"], 1 / 60):
        raise ValueError("invalid episode time")
    vector_keys = (
        "forward_distance", "episode_lengths", "episode_terminated", "episode_out_of_lane",
        "episode_world_exit", "family_indices", "level_indices", "episode_strict_one_tile_success",
        "episode_strict_all_tiles_success", "episode_full_horizon_survival", "episode_switch_count",
        "episode_v10_duty", "episode_switch_steps", "episode_switch_to_v10", "episode_uncertain_steps",
        "episode_fall_within_switch_window", "episode_max_action_jump_rms", "episode_mean_action_disagreement_rms",
        "episode_alpha_sum", "episode_active_steps", "first_hit_one_seconds", "first_hit_six_seconds",
        "distance_at_snapshot_m", "maximum_distance_m",
    )
    if any(not isinstance(data.get(k), list) or len(data[k]) != n for k in vector_keys):
        raise ValueError("missing or malformed episode array")
    bool_keys = ("episode_terminated", "episode_out_of_lane", "episode_world_exit",
                 "episode_strict_one_tile_success", "episode_strict_all_tiles_success",
                 "episode_full_horizon_survival", "episode_fall_within_switch_window")
    one, six = (struct.unpack("<f", struct.pack("<f", condition[k]))[0]
                for k in ("one_threshold", "six_threshold"))
    rows = []
    for i in range(n):
        if any(type(data[k][i]) is not bool for k in bool_keys):
            raise ValueError("terminal/success flags must be booleans")
        distance, length = data["forward_distance"][i], data["episode_lengths"][i]
        if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
            raise ValueError("nonfinite distance")
        if type(length) is not int or not 0 < length <= condition["max_steps"] or type(data["episode_active_steps"][i]) is not int or data["episode_active_steps"][i] != length:
            raise ValueError("first episode duration mismatch")
        family_id, level = data["family_indices"][i], data["level_indices"][i]
        if type(family_id) is not int or not 0 <= family_id < len(data["family_names"]) or type(level) is not int or not 0 <= level < len(data["difficulties"]):
            raise ValueError("invalid reporting family/level")
        fall, lane, world = (data[k][i] for k in ("episode_terminated", "episode_out_of_lane", "episode_world_exit"))
        safe = not (fall or lane or world)
        success_one, success_six = safe and distance >= one, safe and distance >= six
        if (success_one, success_six) != (data["episode_strict_one_tile_success"][i], data["episode_strict_all_tiles_success"][i]):
            raise ValueError("strict success mismatch")
        survived = length == condition["max_steps"] and not fall
        if data["episode_full_horizon_survival"][i] != survived:
            raise ValueError("survival mismatch")
        switches, targets = data["episode_switch_steps"][i], data["episode_switch_to_v10"][i]
        if not isinstance(switches, list) or not isinstance(targets, list) or type(data["episode_switch_count"][i]) is not int or len(switches) != data["episode_switch_count"][i] or len(switches) != len(targets):
            raise ValueError("switch event count mismatch")
        if any(type(s) is not int or not 1 <= s <= length for s in switches) or any(a >= b for a, b in zip(switches, switches[1:])):
            raise ValueError("switch outside first episode or nonmonotonic")
        if any(type(t) is not bool for t in targets) or any(a == b for a, b in zip(targets, targets[1:])) or (targets and not targets[0]):
            raise ValueError("switch targets must alternate from v5 to v10")
        near = fall and any(0 <= (length - (s - 1)) * condition["dt"] <= .5 + 1.e-8 for s in switches)
        if near != data["episode_fall_within_switch_window"][i]:
            raise ValueError("switch-associated fall mismatch")
        duty, uncertain = data["episode_v10_duty"][i], data["episode_uncertain_steps"][i]
        if isinstance(duty, bool) or not isinstance(duty, (int, float)) or not math.isfinite(duty) or not -1.e-7 <= duty <= 1 + 1.e-7 or type(uncertain) is not int or not 0 <= uncertain <= length:
            raise ValueError("invalid duty or unknown count")
        if isinstance(data["episode_alpha_sum"][i], bool) or not isinstance(data["episode_alpha_sum"][i], (int, float)):
            raise ValueError("alpha sum must be numeric")
        if not math.isclose(duty * length, data["episode_alpha_sum"][i], rel_tol=1.e-6, abs_tol=1.e-5):
            raise ValueError("duty denominator mismatch")
        if data["mode"] in ("v5", "v10") and (switches or not math.isclose(duty, float(data["mode"] == "v10"), abs_tol=1.e-7)):
            raise ValueError("fixed baseline is not an exact endpoint")
        for key in ("episode_max_action_jump_rms", "episode_mean_action_disagreement_rms"):
            if isinstance(data[key][i], bool) or not isinstance(data[key][i], (int, float)) or not math.isfinite(data[key][i]) or data[key][i] < 0:
                raise ValueError("invalid action diagnostics")
        snapshot = data["distance_at_snapshot_m"][i]
        censored = length * condition["dt"] <= condition["snapshot_seconds"] + 1.e-8
        if (snapshot is None) != censored or (snapshot is not None and (isinstance(snapshot, bool) or not math.isfinite(snapshot))):
            raise ValueError("snapshot censoring mismatch")
        maximum = data["maximum_distance_m"][i]
        if isinstance(maximum, bool) or not math.isfinite(maximum) or maximum + 1.e-5 < distance or (snapshot is not None and snapshot > maximum + 1.e-5):
            raise ValueError("invalid maximum distance")
        for key, threshold in (("first_hit_one_seconds", one), ("first_hit_six_seconds", six)):
            hit = data[key][i]
            if (hit is not None) != (maximum >= threshold):
                raise ValueError("first hit inconsistent with maximum distance")
            if hit is not None and (isinstance(hit, bool) or not math.isfinite(hit) or not 0 < hit <= length * condition["dt"] + 1.e-7):
                raise ValueError("first hit outside first episode")
        if data["first_hit_six_seconds"][i] is not None and data["first_hit_six_seconds"][i] < data["first_hit_one_seconds"][i]:
            raise ValueError("six-tile hit precedes one-tile hit")
        rows.append(dict(family=data["family_names"][family_id], level=level, one=int(success_one),
                         six=int(success_six), falls=int(fall), lane=int(lane), world=int(world),
                         survival=int(survived), distance=distance, steps=length, duty=duty,
                         switches=len(switches), near_switch_fall=int(near), uncertain=uncertain,
                         jump=data["episode_max_action_jump_rms"][i], disagreement=data["episode_mean_action_disagreement_rms"][i]))
    return rows


def aggregate(rows):
    terrain = [r for r in rows if r["family"] != "flat"]
    flat = [r for r in rows if r["family"] == "flat"]
    result = {"n": len(terrain), "flat_n": len(flat), "flat_falls": sum(r["falls"] for r in flat),
              "flat_world": sum(r["world"] for r in flat)}
    for key in ("one", "six", "falls", "lane", "world", "survival", "switches", "near_switch_fall"):
        result[key] = sum(r[key] for r in terrain)
    result.update(
        mean_final_distance=mean(r["distance"] for r in terrain) if terrain else None,
        mean_v10_duty=mean(r["duty"] for r in terrain) if terrain else None,
        mean_max_action_jump=mean(r["jump"] for r in terrain) if terrain else None,
        mean_action_disagreement=mean(r["disagreement"] for r in terrain) if terrain else None,
        uncertain_steps=sum(r["uncertain"] for r in terrain),
        flat_mean_v10_duty=mean(r["duty"] for r in flat) if flat else None,
    )
    return result


def promotion(primary, horizon):
    def rate(group, key, denominator="n"):
        if group[denominator] <= 0:
            raise ValueError("empty promotion denominator")
        return Fraction(group[key], group[denominator])
    hybrid, v5, v10 = (primary[k] for k in ("hybrid", "v5", "v10"))
    checks = {f"{metric}_vs_{label}": rate(hybrid, metric) >= rate(other, metric)
              for metric in ("one", "six") for label, other in (("v5", v5), ("v10", v10))}
    checks.update({f"{metric}_vs_v5": rate(hybrid, metric) <= rate(v5, metric) for metric in ("falls", "lane")})
    checks["world_zero"] = hybrid["world"] + hybrid["flat_world"] == 0
    checks["flat_falls_vs_v5"] = rate(hybrid, "flat_falls", "flat_n") <= rate(v5, "flat_falls", "flat_n")
    checks["64s_stone_six_vs_v5"] = rate(horizon["hybrid"], "six") > rate(horizon["v5"], "six")
    checks["64s_stone_falls_vs_v5"] = rate(horizon["hybrid"], "falls") <= rate(horizon["v5"], "falls")
    return {"passed": all(checks.values()), "checks": checks}


def summarize(directory):
    directory = Path(directory)
    frozen = json.loads((directory / "frozen.json").read_text())
    summaries = {}
    sources = []
    for scenario, folder, envs, seconds in (("mixed", "evaluations", 175, 16), ("stones", "horizon", 10, 64)):
        groups = defaultdict(list)
        seeds = defaultdict(lambda: defaultdict(list))
        initial = {}
        for mode, policy_seed in [("v5", 42), *[(m, s) for s in (42, 43, 44) for m in ("v10", "hybrid")]]:
            for geom, reset in frozen["holdouts"]:
                path = directory / folder / f"{mode}_seed{policy_seed}__geometry{geom}_reset{reset}.json"
                data = json.loads(path.read_text())
                expected = (mode, None if mode == "v5" else policy_seed, geom, reset, scenario, envs, seconds, "holdout")
                actual = (data["mode"], data["policy_seed"], data["geometry_seed"], data["reset_seed"], data["scenario"], data["num_envs"], data["condition"]["seconds"], data["phase"])
                if actual != expected or data["gate_config"] != frozen["gate_config"]:
                    raise ValueError(f"frozen evaluation contract mismatch: {path}")
                if data["checkpoint_sha256"] != frozen["checkpoints"][str(policy_seed)]["sha256"] or data["v5_sha256"] != frozen["v5_sha256"]:
                    raise ValueError("model hash mismatch")
                if data["started_utc"] <= frozen["frozen_at"]:
                    raise ValueError("holdout precedes freeze")
                pair = (geom, reset)
                initial.setdefault(pair, data["initial_state_sha256"])
                if data["initial_state_sha256"] != initial[pair]:
                    raise ValueError("paired policies do not share initial states")
                rows = audit(data)
                groups[mode].extend(rows)
                if mode != "v5":
                    seeds[str(policy_seed)][mode].extend(rows)
                sources.append({"path": str(path), "sha256": sha(path), "episodes": len(rows)})
        summaries[folder] = {
            "groups": {k: aggregate(v) for k, v in groups.items()},
            "seeds": {seed: {k: aggregate(v) for k, v in entries.items()} for seed, entries in seeds.items()},
            "family_level": {k: {f"{f}/level{level}": aggregate([r for r in v if r["family"] == f and r["level"] == level])
                                 for f, level in sorted({(r["family"], r["level"]) for r in v})} for k, v in groups.items()},
        }
    summaries["promotion"] = promotion(summaries["evaluations"]["groups"], summaries["horizon"]["groups"])
    summaries["files"] = sources
    summaries["interpretation"] = "Same two fresh maps; v5 once/map, each candidate3 frozen seeds. Not thousands of independent maps. No retraining. Ideal depth, not real RGB-D."
    return summaries


def markdown(result):
    lines = ["# v11: depth-gated frozen v5/v10", "", result["interpretation"], ""]
    for key, title in (("evaluations", "16s mixed terrain"), ("horizon", "Separate64s stones1.0")):
        lines += [f"## {title}", "", "| Policy | One tile | Six tiles | Falls | Lane | Flat falls | Mean v10 duty | Switches | Falls near switch |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for mode in ("v5", "v10", "hybrid"):
            d = result[key]["groups"][mode]
            lines.append(f"| {mode} | {d['one']}/{d['n']} | {d['six']}/{d['n']} | {d['falls']}/{d['n']} | {d['lane']}/{d['n']} | {d['flat_falls']}/{d['flat_n']} | {d['mean_v10_duty']:.3f} | {d['switches']} | {d['near_switch_fall']} |")
        lines.append("")
    lines += [f"Promotion: **{'PASS' if result['promotion']['passed'] else 'FAIL — retain v5'}**", ""]
    lines += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in result["promotion"]["checks"].items()]
    lines += ["", "Falls near a switch are temporal associations, not causal attribution. See JSON for perseed and family/level results."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    for name, text in (("summary.json", json.dumps(result, indent=2, allow_nan=False) + "\n"), ("summary.md", markdown(result))):
        with (args.directory / name).open("x") as stream:
            stream.write(text)
