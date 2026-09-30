"""v19 fixed-policy comparison in the unchanged 91D physical evaluation scene.

The rollout is adapted from frozen evaluate_style_v18.py, without editing it.
Only controller/provenance binding changes; validate development parity first.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

from week03_ant.rebaseline_study_v19 import (
    ART, CONTROLLERS, HOLDOUTS, ROOT, SCHEMA, TASK, V5_SHA, PLAN,
    command_mode, model_key, policy_mode, models, legacy_hashes,
    save_json, sha, tensor_sha, utc, validate_teacher, verify_hashes,
    evaluation_inputs, validate_request, verify_frozen,
)
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--controller", choices=CONTROLLERS, required=True)
parser.add_argument("--geometry", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--scenario", choices=("mixed", "stones"), default="mixed")
parser.add_argument("--seconds", type=int, choices=(16, 64), default=16)
parser.add_argument("--num_envs", type=int, default=175)
parser.add_argument("--phase", choices=("prepare_development", "prepare", "smoke", "holdout"), required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
if unknown or args.output.exists() or min(args.geometry, args.seed) < 0 or args.num_envs <= 0:
    parser.error("no overrides, overwrites, negative seeds or nonpositive env counts")
validate_request(args.controller, args.geometry, args.seed, args.scenario,
                 args.seconds, args.num_envs, args.phase)
if args.device is None:
    args.device = "cuda:1"
legacy = legacy_hashes()
key = model_key(args.controller)
access = command_mode(args.controller)
entry = models()[key]
checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else ROOT / entry["checkpoint"]
if checkpoint != ROOT / entry["checkpoint"] or sha(checkpoint) != entry["sha256"]:
    raise ValueError("v19 allows only its predeclared historical checkpoints")
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
import isaaclab_tasks  # noqa: F401
import week03_ant.tasks.contact_v16  # noqa: F401
from week03_ant.depth_math import encode_height_scan
from week03_ant.evaluation import LaneTraversalGeometry
from week03_ant.foothold_math import GRID_RAYS
from week03_ant.history_gate import HistoryDepthPolicyGate, HistoryGateConfig
from week03_ant.horizon import HorizonEpisodeTracker
from week03_ant.hybrid_telemetry import HybridTelemetry
from week03_ant.posture_math import PostureConfig
from week03_ant.posture_telemetry import PostureTelemetry
from week03_ant.contact_telemetry import ContactTelemetry
from week03_ant.tasks.contact_v16_cfg import ContactSlipReward, validate_contact_config_parity
from week03_ant.tasks.lanes import ANT_FOOT_REACH, lane_assign
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
    if args.scenario == "stones":
        env_cfg.events.lane_layout.func = lane_assign
        env_cfg.events.lane_layout.params = {"family_levels": [("stepping_stones", 4)]}
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
                  "default_config_parity": True, "new_training_transitions": 0}
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
        snapshot = 8.0 if args.seconds == 16 else 16.0
        tracker = HorizonEpisodeTracker(args.num_envs, raw.device, dt=dt, max_steps=max_steps,
                                         snapshot_seconds=snapshot, one_tile_threshold=13.1,
                                         all_tiles_threshold=53.1)
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
            tracker.update(step=step, rewards=rewards, dones=done, reset_terminated=terminated,
                           current_distance=distance, last_distance=lane.last_distance,
                           current_out_of_lane=lane.out_of_lane, last_out_of_lane=lane.last_out_of_lane,
                           current_world_exit=lane.world_exit, last_world_exit=lane.last_world_exit)
            routing.after_step(step, active, done, terminated)
            gate.reset(done)
            if tracker.complete:
                break
        data = tracker.to_dict()
        data["distance_at_snapshot_m"] = data.pop("distance_at_16s")
        data["episode_full_horizon_survival"] = data.pop("episode_full_64s_survival")
        data["aggregates"]["full_horizon_survivals"] = data["aggregates"].pop("full_64s_survivals")
        data.update(routing.to_dict(tracker.steps))
        result = {**common, **data, "condition": {"seconds": args.seconds, "max_steps": max_steps,
                  "dt": dt, "snapshot_seconds": snapshot, "one_threshold": 13.1, "six_threshold": 53.1,
                  "footprint_margin": ANT_FOOT_REACH, "observations": 91, "expert_observations": 91 if access else 88, "v5_observations": 60, "actions": 8},
                  "family_names": lane.family_names, "difficulties": lane.difficulties,
                  "family_indices": family.cpu().tolist(), "level_indices": level.cpu().tolist(),
                  "contact_telemetry": contact.to_dict(tracker.steps),
                  "contact_interpretation": "Pre-action first-episode visited states only, excluding initial reset and portal-wrap stale samples. Filtered foot contact is not whole-body airborne or matched-state causality. World-XY distal foot-tip speed is a kinematic proxy, not actual sole slip; cost is unweighted and not dt-integrated.",
                  "history_switch_events": gate.events, "posture_telemetry": posture.to_dict(tracker.steps),
                  "routing_inputs": "Unchanged v12 history gate uses depth only. Contact and posture metrics are passive.",
                  "posture_interpretation": "Conditional on visited states, not a matched-state causal estimate. Kinematic swing proxy, not contact detection.",
                  "finished_utc": utc(), "wall_seconds": time.monotonic() - begin}
        save_json(args.output, result)
        print(f"[v19] {args.controller}: {result['aggregates']}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
