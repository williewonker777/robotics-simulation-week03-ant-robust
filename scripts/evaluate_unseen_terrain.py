# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in unseen-layout evaluation and uncut demo; never a historical holdout.

64-second runs score independent 16/64-second windows of the same first
physical episode. Rendering retains later resets, which never enter scores.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import hashlib
import importlib
from pathlib import Path
import sys
import time

CONTROLLERS = ('v5', 'v16_control', 'history_control', 'high53', 'history_high53')
SCHEMA = 'week03_ant_unseen_layout_v1'
TASK = 'Week03-Ant-Contact-v16-Eval-v0'


def controller_binding(controller):
    if controller not in CONTROLLERS:
        raise ValueError('unknown controller')
    key = 'original' if controller == 'v5' else ('seed53' if 'high53' in controller else 'control')
    mode = 'hybrid' if controller.startswith('history_') else ('v5' if controller == 'v5' else 'v10')
    return key, mode, None if controller == 'v5' else 'conditioned'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True, choices=CONTROLLERS)
    parser.add_argument('--geometry', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--scenario', choices=('mixed', 'obstacles', 'stones', 'flat'), default='mixed')
    parser.add_argument('--level', type=int, choices=range(5), default=4)
    parser.add_argument('--seconds', type=int, choices=(16, 64), default=64)
    parser.add_argument('--num-envs', '--num_envs', dest='num_envs', type=int, default=175)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--kit-args', '--kit_args', dest='kit_args',
                        default='--/renderer/multiGpu/enabled=false')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--real-time', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(argv)
    if min(args.geometry, args.seed) < 0 or args.num_envs <= 0:
        parser.error('nonnegative seeds and positive environment count required')
    if args.scenario == 'mixed' and args.num_envs % 35:
        parser.error('mixed requires a multiple of 35 environments')
    if args.prepare and args.record:
        parser.error('prepare performs zero scored steps and cannot record')
    if args.record and args.record.suffix.lower() != '.mp4':
        parser.error('record must be an MP4 path')
    paths = [p for p in (args.output, args.record) if p is not None]
    if len({p.resolve() for p in paths}) != len(paths) or any(p.exists() for p in paths):
        parser.error('output paths must be distinct and new; refusing overwrite')
    return args


def caption(args, family, step, alpha, resets, distance):
    return (
        f'{args.controller} | {family} | geometry={args.geometry} reset={args.seed} | t={step / 60:.2f}s',
        f'alpha={alpha:.2f} | resets={resets} | current episode={distance:.1f}m',
        'UNCUT qualitative replay; later resets are NOT benchmark successes',
    )


def single_window(tracker):
    data = tracker.to_dict()
    data['distance_at_snapshot_m'] = data.pop('distance_at_16s')
    data['episode_full_horizon_survival'] = data.pop('episode_full_64s_survival')
    data['aggregates']['full_horizon_survivals'] = data['aggregates'].pop('full_64s_survivals')
    data['window_seconds'] = 16
    return data


def resolve_model(controller):
    from week03_ant.rebaseline_study_v19 import ROOT, models, sha
    from week03_ant.seed_study_v22 import ART

    key, mode, access = controller_binding(controller)
    if key == 'seed53':
        # The historical freeze, not new evaluation outputs, fixes seed53 final249.
        entry = json.loads((ART / 'frozen.json').read_text())['models']['seed53']
        if (entry['sha256'] != '5c86a9b6efebf02bca4b37609acf688d69514d1887fb5b85cf28b79f5dc17c65'
                or not entry['checkpoint'].endswith('/seed53/model_249.pt')):
            raise ValueError('historical seed53 identity changed')
    else:
        entry = models()[key]
    checkpoint = ROOT / entry['checkpoint']
    if sha(checkpoint) != entry['sha256']:
        raise ValueError('checkpoint checksum mismatch')
    return checkpoint, entry['sha256'], mode, access


def launch(argv=None):
    args = parse_args(argv)
    checkpoint, expected_sha, mode, access = resolve_model(args.controller)
    from isaaclab.app import AppLauncher

    args.enable_cameras = bool(args.record)
    sys.argv = [sys.argv[0]]
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.hydra import hydra_task_config
        importlib.import_module('isaaclab_tasks')
        importlib.import_module('week03_ant.tasks.contact_v16')
        from week03_ant.command_policy import CommandPriorActorCritic
        from week03_ant.prior_policy import PriorActorCritic
        from week03_ant.depth_math import encode_height_scan
        from week03_ant.evaluation import LaneTraversalGeometry
        from week03_ant.foothold_math import GRID_RAYS
        from week03_ant.history_gate import HistoryDepthPolicyGate, HistoryGateConfig
        from week03_ant.horizon import HorizonEpisodeTracker
        from week03_ant.hybrid_telemetry import HybridTelemetry
        from week03_ant.paired_horizon_v23 import PairedHorizonTracker, immutable_snapshot
        from week03_ant.rebaseline_study_v19 import ROOT, sha, tensor_sha, utc, validate_teacher, V5_SHA
        from week03_ant.tasks.contact_v16_cfg import validate_contact_config_parity
        from week03_ant.tasks.lanes import ANT_FOOT_REACH, lane_assign
        from week03_ant.tasks.rough_v5_cfg import FLAT_PLANE_HEIGHT

        @hydra_task_config(TASK, 'rsl_rl_cfg_entry_point')
        def run(env_cfg, agent_cfg):
            if validate_contact_config_parity() is not True:
                raise ValueError('v16 scene parity failed')
            env_cfg.scene.num_envs = args.num_envs
            env_cfg.episode_length_s = float(args.seconds)
            env_cfg.seed = agent_cfg.seed = args.seed
            env_cfg.sim.device = agent_cfg.device = args.device
            env_cfg.scene.terrain.terrain_generator.seed = args.geometry
            env_cfg.log_dir = str(checkpoint.parent)
            if args.scenario != 'mixed':
                family_name = {'stones': 'stepping_stones'}.get(args.scenario, args.scenario)
                env_cfg.events.lane_layout.func = lane_assign
                env_cfg.events.lane_layout.params = {'family_levels': [(family_name, args.level)]}
            env_cfg.viewer.resolution = (1280, 720)
            env_cfg.viewer.origin_type = 'asset_root'
            env_cfg.viewer.asset_name = 'robot'
            env_cfg.viewer.env_index = 0
            source_paths = ('src/week03_ant/tasks/contact_v16_cfg.py',
                'src/week03_ant/tasks/rough_v5_cfg.py', 'src/week03_ant/tasks/lanes.py',
                'src/week03_ant/history_gate.py', 'src/week03_ant/paired_horizon_v23.py')
            source_hashes = {p: sha(ROOT / p) for p in source_paths}
            terrain_config = json.dumps(env_cfg.scene.terrain.terrain_generator.to_dict(),
                sort_keys=True, default=lambda x: f'{x.__module__}.{x.__qualname__}' if callable(x) else str(x))
            terrain_config_sha = hashlib.sha256(terrain_config.encode()).hexdigest()
            env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg,
                **({'render_mode': 'rgb_array'} if args.record else {})), clip_actions=agent_cfg.clip_actions)
            writer = None
            try:
                raw = env.unwrapped
                dt, max_steps = float(raw.step_dt), int(raw.max_episode_length)
                if abs(dt - 1 / 60) > 1.e-8 or max_steps != args.seconds * 60:
                    raise ValueError('60Hz physical horizon contract changed')
                lane = raw._week03_lane_state
                geometry = LaneTraversalGeometry(lane.tile_length, lane.terrain_tiles, ANT_FOOT_REACH)
                if (geometry.one_tile_clearance, geometry.all_tiles_clearance) != (13.1, 53.1):
                    raise ValueError('strict success thresholds changed')
                family, level = (t.clone() for t in lane.family_level())
                constructor_obs = env.get_observations()['policy']
                selected = {'policy': constructor_obs if access else constructor_obs[:, :88].contiguous()}
                policy_cfg = agent_cfg.policy.to_dict()
                policy_cfg.pop('class_name')
                policy_cfg['require_config_match'] = True
                if access:
                    policy_cfg['command_mode'] = access
                else:
                    policy_cfg.pop('command_mode')
                cls = CommandPriorActorCritic if access else PriorActorCritic
                policy = cls(selected, {'policy': ['policy'], 'critic': ['policy']}, 8, **policy_cfg).to(raw.device)
                policy.load_state_dict(torch.load(checkpoint, map_location=raw.device, weights_only=False)['model_state_dict'])
                if policy.prior_mode != 'anchored' or policy.input_mode != 'targets' or (access and policy.command_mode != access):
                    raise ValueError('policy contract changed')
                validate_teacher(policy.state_dict())
                policy.eval().requires_grad_(False)
                observations = env.get_observations()
                if observations['policy'].shape != (args.num_envs, 91):
                    raise ValueError('91D observation contract changed')
                robot = raw.scene['robot'].data
                config = HistoryGateConfig()
                result = dict(schema=SCHEMA, task=TASK, controller=args.controller, mode=mode,
                    command_mode=access, checkpoint=str(checkpoint), checkpoint_sha256=expected_sha,
                    v5_sha256=V5_SHA, teacher_tensor_identity_verified=True, gate_config=asdict(config),
                    geometry_seed=args.geometry, reset_seed=args.seed, scenario=args.scenario,
                    num_envs=args.num_envs, seconds=args.seconds, dt=dt, started_utc=utc(),
                    source_sha256=source_hashes, terrain_config_sha256=terrain_config_sha,
                    purpose='New unseen layouts, not a historical frozen holdout; no seed reselection',
                    routing_inputs='Depth history only; no family labels', new_training_transitions=0,
                    family_names=lane.family_names, difficulties=lane.difficulties,
                    family_indices=family.cpu().tolist(), level_indices=level.cpu().tolist(),
                    initial_state_sha256={'root_state': tensor_sha(robot.root_state_w),
                        'joint_pos': tensor_sha(robot.joint_pos), 'joint_vel': tensor_sha(robot.joint_vel),
                        'observations': tensor_sha(observations['policy'])},
                    initial_prefix_sha256=tensor_sha(observations['policy'][:, :88]),
                    initial_rng_sha256={'cpu': tensor_sha(torch.get_rng_state()),
                        'cuda': tensor_sha(torch.cuda.get_rng_state(raw.device))},
                    success_thresholds_m={'one_tile': 13.1, 'all_tiles': 53.1})
                gate = HistoryDepthPolicyGate(args.num_envs, raw.device, dt, config)
                routing = HybridTelemetry(args.num_envs, raw.device, dt=dt)
                paired = PairedHorizonTracker(args.num_envs, raw.device) if args.seconds == 64 else None
                tracker = paired.full if paired else HorizonEpisodeTracker(args.num_envs, raw.device,
                    dt=dt, max_steps=max_steps, snapshot_seconds=8, one_tile_threshold=13.1, all_tiles_threshold=53.1)
                windows, events = {}, []
                resets, frames, step = 0, 0, 0
                if args.record:
                    import imageio.v2 as imageio
                    import numpy as np
                    from PIL import Image, ImageDraw
                    args.record.parent.mkdir(parents=True, exist_ok=True)
                    with args.record.open('xb'):
                        pass
                    writer = imageio.get_writer(str(args.record), fps=30, macro_block_size=None)
                start = time.monotonic()
                for step in range(1, (0 if args.prepare else max_steps) + 1):
                    scoring = not tracker.complete
                    active = (~tracker.finished).clone()
                    with torch.inference_mode():
                        view = observations if access else {'policy': observations['policy'][:, :88].contiguous()}
                        old, new = policy.teacher_mean(view), policy.act_inference(view)
                        sensor = raw.scene['height_scanner']
                        sensor.update(0., force_recompute=True)
                        encoded = encode_height_scan(robot.root_pos_w[:, 2], sensor.data.ray_hits_w[..., 2],
                            ray_offset_z=sensor.cfg.offset.pos[2], max_distance=sensor.cfg.max_distance,
                            plane_height=FLAT_PLANE_HEIGHT)
                        if encoded.shape[-1] != GRID_RAYS * 2:
                            raise ValueError('scanner shape changed')
                        heights = encoded[:, :GRID_RAYS]
                        choice = gate.step(-heights - .5, (encoded[:, GRID_RAYS:] > .5) & (heights.abs() < .99), old, new, mode=mode)
                        if not torch.isfinite(choice['actions']).all():
                            raise ValueError('nonfinite action')
                        if scoring:
                            routing.before_step(step, active, choice, old, new)
                        observations, rewards, dones, _ = env.step(choice['actions'])
                    done, terminated = dones.reshape(-1).bool(), raw.reset_terminated.reshape(-1).bool()
                    distance = lane.odometer(robot.root_pos_w[:, 0]) - lane.spawn_x
                    if scoring:
                        update = dict(step=step, rewards=rewards, dones=done, reset_terminated=terminated,
                            current_distance=distance, last_distance=lane.last_distance,
                            current_out_of_lane=lane.out_of_lane, last_out_of_lane=lane.last_out_of_lane,
                            current_world_exit=lane.world_exit, last_world_exit=lane.last_world_exit)
                        (paired if paired else tracker).update(**update)
                        routing.after_step(step, active, done, terminated)
                        ready = paired.ready_windows() if paired else ((16,) if tracker.complete else ())
                        for seconds in ready:
                            data = paired.window_data(seconds) if paired else single_window(tracker)
                            data['routing'] = routing.to_dict((paired.trackers[seconds] if paired else tracker).steps)
                            windows[str(seconds)] = immutable_snapshot(data)
                            if paired:
                                paired.mark_captured(seconds)
                    if bool(done[0]):
                        resets += 1
                        events.append(dict(step=step, terminated=bool(terminated[0]),
                            distance_m=float(lane.last_distance[0]), out_of_lane=bool(lane.last_out_of_lane[0]),
                            world_exit=bool(lane.last_world_exit[0])))
                    gate.reset(done)
                    if writer and step % 2 == 0:
                        state, tick = robot.root_state_w.clone(), raw.sim.current_time_step_index
                        pos = robot.root_pos_w[0].detach().cpu().numpy()
                        raw.sim.set_camera_view(pos + np.array([-4., -4., 2.5]), pos)
                        raw.sim.render()
                        frame = raw.render(recompute=True)
                        if frame is None or frame.shape != (720, 1280, 3):
                            raise RuntimeError('expected 1280x720 RGB frame')
                        if tick != raw.sim.current_time_step_index or not torch.equal(state, robot.root_state_w):
                            raise RuntimeError('render advanced physical state')
                        image = Image.fromarray(frame)
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, 1280, 76), fill='black')
                        for line, y in zip(caption(args, lane.family_names[int(family[0])], step,
                                float(choice['alpha'][0]), resets, float(distance[0])), (5, 28, 51)):
                            draw.text((8, y), line, fill='white')
                        writer.append_data(np.asarray(image))
                        frames += 1
                    if args.real_time:
                        time.sleep(max(0., start + step * dt - time.monotonic()))
                    if tracker.complete and args.headless and not args.record:
                        break
                if writer:
                    writer.close()
                    writer = None
                if not args.prepare and set(windows) != ({'16', '64'} if paired else {'16'}):
                    raise RuntimeError('missing scoring windows')
                if {p: sha(ROOT / p) for p in source_paths} != source_hashes:
                    raise ValueError('evaluation sources changed during run')
                if sha(checkpoint) != expected_sha:
                    raise ValueError('checkpoint changed during evaluation')
                result.update(source_sha256_after={p: sha(ROOT / p) for p in source_paths},
                    checkpoint_sha256_after=sha(checkpoint), windows=windows,
                    scored_episodes=0 if args.prepare else args.num_envs, physics_steps=step,
                    finished_utc=utc(), wall_seconds=time.monotonic() - start,
                    demo={'uncut': True, 'env_index': 0, 'reset_events': events, 'resets': resets,
                        'frames': frames, 'fps': 30, 'duration_seconds': frames / 30,
                        'record': str(args.record) if args.record else None,
                        'sha256': sha(args.record) if args.record else None,
                        'render_physics_unchanged': True if args.record else None})
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open('x') as stream:
                    json.dump(result, stream, indent=2, allow_nan=False)
                    stream.write('\n')
                print(f'[unseen] {args.controller}: {step} steps, windows={list(windows)}', flush=True)
            finally:
                if writer:
                    writer.close()
                env.close()
        run()
    finally:
        app.close()


if __name__ == '__main__':
    launch()
