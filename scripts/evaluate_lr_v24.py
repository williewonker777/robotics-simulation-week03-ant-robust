"""v24 passive 16s/64s windows in the unchanged 91D physical evaluation scene.

The action/sensor/step loop is retained from frozen evaluate_seed_v22.py.
One physical 64s first episode supplies two scoring windows without a 16s reset;
exact development replay and snapshot passivity gate the fresh-map evaluation.
"""

import argparse
import importlib
from dataclasses import asdict
from pathlib import Path
import sys
import time

from week03_ant.lr_study_v24 import (
    ART, CONTROLLERS, ROOT, SCHEMA, BUNDLE_SCHEMA, TASK, V5_SHA, PLAN,
    command_mode, model_key, policy_mode, models, legacy_hashes, training_seed,
    save_json, sha, tensor_sha, utc, validate_teacher,
    evaluation_inputs, validate_request, verify_frozen,
)
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--controller", choices=CONTROLLERS, required=True)
parser.add_argument("--geometry", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--scenario", choices=("mixed",), default="mixed")
parser.add_argument("--seconds", type=int, choices=(64,), default=64)
parser.add_argument("--num_envs", type=int, default=175)
parser.add_argument("--phase", choices=("prepare_development", "prepare", "reference", "smoke", "holdout"), required=True)
parser.add_argument("--no-prefixes", action="store_true", help="Development-only uninstrumented 64s reference")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
if unknown or args.output.exists() or min(args.geometry, args.seed) < 0 or args.num_envs <= 0:
    parser.error("no overrides, overwrites, negative seeds or nonpositive env counts")
validate_request(args.controller, args.geometry, args.seed, args.scenario,
                 args.seconds, args.num_envs, args.phase, instrumented=not args.no_prefixes)
if args.device is None:
    args.device = "cuda:1"
if args.device != "cuda:1":
    parser.error("v24 is assigned to cuda:1")
legacy = legacy_hashes()
key = model_key(args.controller)
access = command_mode(args.controller)
entry = models()[key]
if key != 'parent' and (entry.get('iteration') != 249 or entry.get('transitions') != 32768000
                        or entry.get('training_seed') != training_seed(key)):
    raise ValueError('v24 evaluates final full-budget checkpoints only')
checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else ROOT / entry["checkpoint"]
if checkpoint != ROOT / entry["checkpoint"] or sha(checkpoint) != entry["sha256"]:
    raise ValueError("v24 allows only its predeclared historical checkpoints")
freeze_sha = input_sha = cache_sha = None
if args.phase in ("prepare", "holdout"):
    verify_frozen()
    freeze_sha = sha(ART / "frozen.json")
if args.phase == "holdout":
    inputs = evaluation_inputs()
    input_sha = sha(ART / "evaluation_inputs.json")
    cache_sha = inputs["terrain_cache_manifest_sha256"]
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app

import gymnasium as gym
import torch
from week03_ant.prior_policy import PriorActorCritic
from week03_ant.command_policy import CommandPriorActorCritic
from isaaclab.managers import RewardTermCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
importlib.import_module("isaaclab_tasks")
importlib.import_module("week03_ant.tasks.contact_v16")
from week03_ant.depth_math import encode_height_scan
from week03_ant.evaluation import LaneTraversalGeometry
from week03_ant.foothold_math import GRID_RAYS
from week03_ant.history_gate import HistoryDepthPolicyGate, HistoryGateConfig
from week03_ant.paired_horizon_v23 import PairedHorizonTracker, immutable_snapshot, snapshot_passivity
from week03_ant.hybrid_telemetry import HybridTelemetry
from week03_ant.posture_math import PostureConfig
from week03_ant.posture_telemetry import PostureTelemetry
from week03_ant.contact_telemetry import ContactTelemetry
from week03_ant.tasks.contact_v16_cfg import ContactSlipReward, validate_contact_config_parity
from week03_ant.tasks.lanes import ANT_FOOT_REACH
from week03_ant.tasks.posture_v13_cfg import PostureReward
from week03_ant.tasks.rough_v5_cfg import FLAT_PLANE_HEIGHT
from evaluate_history_hybrid import RecordedGate


def scan(env):
    sensor = env.scene["height_scanner"]
    sensor.update(0.0, force_recompute=True)
    encoded = encode_height_scan(env.scene["robot"].data.root_pos_w[:, 2], sensor.data.ray_hits_w[..., 2],
                                 ray_offset_z=sensor.cfg.offset.pos[2], max_distance=sensor.cfg.max_distance,
                                 plane_height=FLAT_PLANE_HEIGHT)
    if encoded.shape[-1] != GRID_RAYS * 2:
        raise ValueError("scanner shape changed")
    height = encoded[:, :GRID_RAYS]
    return -height - .5, (encoded[:, GRID_RAYS:] > .5) & (height.abs() < .99)


def state_value(value):
    """Hash every tensor and preserve every scalar in the mutable gate tree."""
    if torch.is_tensor(value):
        return {"tensor_sha256": tensor_sha(value), "shape": list(value.shape), "dtype": str(value.dtype)}
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, (list, tuple)):
        return [state_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): state_value(item) for key, item in value.items()}
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "__dict__"):
        return {key: state_value(item) for key, item in vars(value).items()}
    raise TypeError(f"unrecognized mutable gate state: {type(value).__name__}")


def physical_state(raw, robot, observations, gate, policy, tracker, mode):
    import hashlib
    import json

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {"root_state_sha256": tensor_sha(robot.root_state_w),
            "joint_pos_sha256": tensor_sha(robot.joint_pos), "joint_vel_sha256": tensor_sha(robot.joint_vel),
            "episode_length_buf_sha256": tensor_sha(raw.episode_length_buf),
            "common_step_counter": int(raw.common_step_counter),
            "observations_sha256": tensor_sha(observations["policy"]),
            "cpu_rng_sha256": tensor_sha(torch.get_rng_state()),
            "cuda_rng_sha256": tensor_sha(torch.cuda.get_rng_state(raw.device)),
            "policy_mode": mode, "policy_training": bool(policy.training),
            "policy_state_sha256": digest({key: tensor_sha(value) for key, value in policy.state_dict().items()}),
            "gate_state_sha256": digest(state_value(gate)),
            "full_first_episode_active_sha256": tensor_sha(~tracker.finished)}


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    started, begin = utc(), time.monotonic()
    if validate_contact_config_parity() is not True:
        raise ValueError("v16 evaluation task parity failed")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = float(args.seconds)
    env_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device = args.device
    env_cfg.scene.terrain.terrain_generator.seed = args.geometry
    env_cfg.log_dir = str(checkpoint.parent)
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        raw = env.unwrapped
        dt, max_steps = float(raw.step_dt), int(raw.max_episode_length)
        if max_steps != args.seconds * 60 or abs(dt - 1 / 60) > 1.e-8:
            raise ValueError("physical control-time contract changed")
        lane = raw._week03_lane_state
        geometry = LaneTraversalGeometry(lane.tile_length, lane.terrain_tiles, ANT_FOOT_REACH)
        if (geometry.one_tile_clearance, geometry.all_tiles_clearance) != (13.1, 53.1):
            raise ValueError("strict evaluation distances changed")
        family, level = (t.clone() for t in lane.family_level())
        # Mirror the frozen runner's two observation queries and CPU construction.
        # Only the explicit input view differs; no second physical env or PPO update.
        constructor_obs = env.get_observations()["policy"]
        selected = {"policy": constructor_obs if access else constructor_obs[:, :88].contiguous()}
        policy_cfg = agent_cfg.policy.to_dict()
        policy_cfg.pop("class_name")
        policy_cfg["require_config_match"] = True
        if access:
            policy_cfg["command_mode"] = access
            cls = CommandPriorActorCritic
        else:
            policy_cfg.pop("command_mode")
            cls = PriorActorCritic
        policy = cls(selected, {"policy": ["policy"], "critic": ["policy"]}, 8, **policy_cfg).to(raw.device)
        policy.load_state_dict(torch.load(checkpoint, map_location=raw.device, weights_only=False)["model_state_dict"])
        if policy.prior_mode != "anchored" or policy.input_mode != "targets":
            raise ValueError("anchored targets policy required")
        if access and policy.command_mode != access:
            raise ValueError("checkpoint command mode differs from controller")
        validate_teacher(policy.state_dict())
        policy.eval().requires_grad_(False)
        def selected_obs(obs):
            return obs if access else {"policy": obs["policy"][:, :88].contiguous()}
        observations = env.get_observations()
        if observations["policy"].shape != (args.num_envs, 91):
            raise ValueError("91D observation contract changed")
        robot = raw.scene["robot"].data
        initial = {"root_state": tensor_sha(robot.root_state_w), "joint_pos": tensor_sha(robot.joint_pos),
                   "joint_vel": tensor_sha(robot.joint_vel), "observations": tensor_sha(observations["policy"])}
        mode = policy_mode(args.controller)
        config = HistoryGateConfig()
        common = {"schema": SCHEMA, "task": TASK, "controller": args.controller, "mode": mode,
                  "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint),
                  "v5_sha256": V5_SHA, "teacher_tensor_identity_verified": True,
                  "phase": args.phase, "scenario": args.scenario, "geometry_seed": args.geometry,
                  "reset_seed": args.seed, "num_envs": args.num_envs, "initial_state_sha256": initial,
                  "experiment_freeze_sha256": freeze_sha, "gate_config": asdict(config),
                  "evaluation_inputs_sha256": input_sha, "terrain_cache_manifest_sha256": cache_sha,
                  "posture_config": asdict(PostureConfig()), "contact_config": {"force_threshold_n": 2.0, "speed_scale_m_s": 1.0, "contact_targets": 2},
                  "evaluation_plan_sha256": sha(PLAN), "started_utc": started,
                  "command_mode": access, "command_schema_version": 1,
                  "initial_prefix_sha256": tensor_sha(observations["policy"][:, :88]),
                  "initial_rng_sha256": {"cpu": tensor_sha(torch.get_rng_state()),
                                          "cuda": tensor_sha(torch.cuda.get_rng_state(raw.device))},
                  "default_config_parity": True, "new_training_transitions": 0,
                  "policy_training_transitions": 0 if key == "parent" else entry["transitions"],
                  "training_seed": None if key == "parent" else training_seed(key),
                  "lr_condition": "not_applicable" if key == "parent" else ("low" if key.startswith("low") else "high"),
                  "model_learning_rate": None if key == "parent" else (1.e-5 if key.startswith("low") else 1.e-4)}
        if args.phase in ("prepare", "prepare_development"):
            save_json(args.output, dict(common, scored_episodes=0, finished_utc=utc()))
            return
        gate = RecordedGate(HistoryDepthPolicyGate(args.num_envs, raw.device, dt, config), args.num_envs)
        routing = HybridTelemetry(args.num_envs, raw.device, dt=dt)
        posture = PostureTelemetry(args.num_envs, raw.device)
        measure = PostureReward(RewardTermCfg(func=PostureReward, weight=0.), raw)
        contact = ContactTelemetry(args.num_envs, raw.device)
        contact_measure = ContactSlipReward(RewardTermCfg(func=ContactSlipReward, weight=0.), raw)
        post_wrap = torch.ones(args.num_envs, dtype=torch.bool, device=raw.device)
        paired = PairedHorizonTracker(args.num_envs, raw.device, instrumented=not args.no_prefixes)
        tracker = paired.full
        windows, proofs = {}, {}
        for step in range(1, max_steps + 1):
            active = (~tracker.finished).clone()
            with torch.inference_mode():
                old_action, new_action = policy.teacher_mean(selected_obs(observations)), policy.act_inference(selected_obs(observations))
                height, valid = scan(raw)
                choice = gate.step(height, valid, old_action, new_action, mode=mode)
                routing.before_step(step, active, choice, old_action, new_action)
                posture.before_step(active, measure.geometry(raw))
                contact.before_step(active, contact_measure.geometry(raw), sample_valid=~post_wrap)
                wrap_before = lane.wrap_count.clone()
                if not torch.isfinite(choice["actions"]).all():
                    raise ValueError("nonfinite action")
                observations, rewards, dones, _ = env.step(choice["actions"])
            post_wrap = lane.wrap_count != wrap_before
            done, terminated = dones.reshape(-1).bool(), raw.reset_terminated.reshape(-1).bool()
            distance = lane.odometer(robot.root_pos_w[:, 0]) - lane.spawn_x
            paired.update(step=step, rewards=rewards, dones=done, reset_terminated=terminated,
                           current_distance=distance, last_distance=lane.last_distance,
                           current_out_of_lane=lane.out_of_lane, last_out_of_lane=lane.last_out_of_lane,
                           current_world_exit=lane.world_exit, last_world_exit=lane.last_world_exit)
            routing.after_step(step, active, done, terminated)
            gate.reset(done)
            for seconds in paired.ready_windows():
                before = physical_state(raw, robot, observations, gate, policy, tracker, mode)
                window_tracker = paired.trackers[seconds]
                data = paired.window_data(seconds)
                data.update(routing.to_dict(window_tracker.steps))
                result = {**common, **data, "condition": {"seconds": seconds, "max_steps": seconds * 60,
                          "dt": dt, "snapshot_seconds": 8.0 if seconds == 16 else 16.0,
                          "one_threshold": 13.1, "six_threshold": 53.1,
                          "footprint_margin": ANT_FOOT_REACH, "observations": 91,
                          "expert_observations": 91 if access else 88, "v5_observations": 60, "actions": 8},
                          "family_names": lane.family_names, "difficulties": lane.difficulties,
                          "family_indices": family.cpu().tolist(), "level_indices": level.cpu().tolist(),
                          "contact_telemetry": contact.to_dict(window_tracker.steps),
                          "contact_interpretation": "Pre-action first-episode visited states only, excluding initial reset and portal-wrap stale samples. Filtered foot contact is not whole-body airborne or matched-state causality. World-XY distal foot-tip speed is a kinematic proxy, not actual sole slip; cost is unweighted and not dt-integrated.",
                          "history_switch_events": gate.events,
                          "posture_telemetry": posture.to_dict(window_tracker.steps),
                          "routing_inputs": "Unchanged v12 history gate uses depth only. Contact and posture metrics are passive.",
                          "posture_interpretation": "Conditional on visited states, not a matched-state causal estimate. Kinematic swing proxy, not contact detection.",
                          "finished_utc": utc(), "wall_seconds": time.monotonic() - begin}
                windows[str(seconds)] = immutable_snapshot(result)
                after = physical_state(raw, robot, observations, gate, policy, tracker, mode)
                proofs[str(seconds)] = snapshot_passivity(before, after)
                paired.mark_captured(seconds)
            if tracker.complete:
                break
        expected_windows = {"64"} if args.no_prefixes else {"16", "64"}
        if set(windows) != expected_windows or set(proofs) != expected_windows:
            raise ValueError("physical rollout ended before immutable window capture")
        result = {**common, "schema": BUNDLE_SCHEMA, "instrumented": not args.no_prefixes,
                  "physical_condition": {"seconds": 64, "max_steps": max_steps, "dt": dt},
                  "windows": windows, "snapshot_passivity": proofs,
                  "physical_steps_executed": step, "executed_first_episodes": args.num_envs,
                  "window_observations": args.num_envs * len(windows),
                  "finished_utc": utc(), "wall_seconds": time.monotonic() - begin}
        save_json(args.output, result)
        print(f"[v24] {args.controller}: {len(windows)} windows, {step} physical steps", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
