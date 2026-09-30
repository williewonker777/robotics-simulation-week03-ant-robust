"""Signed learning-progress lane sampling, with no simulator import for CPU tests."""

import torch

FAMILIES = ('rough', 'slope', 'stairs', 'waves', 'obstacles', 'stepping_stones', 'flat')


def base_probabilities(device='cpu'):
    """Five levels per family; preserve the inherited 1:1:1:1:1:4:2 mixture."""
    return torch.tensor([1.] * 25 + [4.] * 5 + [2.] * 5,
                        dtype=torch.float64, device=device) / 55.


def progress_probabilities(progress):
    """Signed, bounded exponential reweighting with a 40% base-mixture floor."""
    progress = torch.as_tensor(progress, dtype=torch.float64)
    if progress.shape != (30,) or not torch.isfinite(progress).all():
        raise ValueError('learning progress must contain 30 finite task returns')
    base = base_probabilities(progress.device)
    mesh_base = base[:30] / base[:30].sum()
    q = mesh_base * torch.exp(torch.clamp(progress / 5., -2., 2.))
    q /= q.sum()
    result = base.clone()
    result[:30] = (9. / 11.) * (0.4 * mesh_base + 0.6 * q)
    return result


def translated_reset_pose(pose, old_lanes, new_lanes, entrance, center, ground):
    """Translate only XYZ; preserve reset jitter, quaternion, and velocity fields."""
    result = pose.clone()
    for axis, origins in enumerate((entrance, center, ground)):
        result[:, axis] += origins[new_lanes] - origins[old_lanes]
    return result


class LPSamplerV17:
    """Episode-credit ledger; updates use complete simultaneous-reset batches."""

    def __init__(self, num_envs, device, mode='fixed', stage_size=1024):
        if isinstance(stage_size, bool) or not isinstance(stage_size, int) or stage_size <= 0:
            raise ValueError('stage_size must be a positive integer')
        self.stage_size = stage_size
        if mode not in ('fixed', 'lp'):
            raise ValueError('mode must be fixed or lp')
        self.mode, self.device = mode, device
        self.generator = torch.Generator(device=device).manual_seed(49017)
        self.probabilities = base_probabilities(device)
        self.first_completed = torch.zeros(num_envs, dtype=torch.bool)
        self.window_counts = torch.zeros(30, dtype=torch.long)
        self.window_sums = torch.zeros(30, dtype=torch.float64)
        self.previous_means = None
        self.stages = []
        self.excluded = 0
        self.completed_counts = torch.zeros(35, dtype=torch.long)
        self.sampled_counts = torch.zeros(35, dtype=torch.long)
        self.duration_steps = torch.zeros(35, dtype=torch.long)
        self.eligible_counts = torch.zeros(35, dtype=torch.long)
        self.terminated_count = 0
        self.timeout_count = 0
        self.invalid_count = 0

    def complete(self, ids, old_lanes, returns, lengths, terminated, timeouts):
        """Credit old tasks, then draw new tasks in sorted environment-ID order."""
        ids, lanes, lengths = [torch.as_tensor(x).detach().cpu().long()
                               for x in (ids, old_lanes, lengths)]
        returns = torch.as_tensor(returns).detach().cpu().double()
        terminated = torch.as_tensor(terminated).detach().cpu().bool()
        timeouts = torch.as_tensor(timeouts).detach().cpu().bool()
        n = ids.numel()
        valid = (all(x.shape == (n,) for x in (ids, lanes, returns, lengths, terminated, timeouts))
                 and len(ids.unique()) == n and bool(((ids >= 0) & (ids < len(self.first_completed))).all())
                 and bool(((lanes >= 0) & (lanes < 35)).all())
                 and bool((lengths > 0).all()) and bool((terminated | timeouts).all())
                 and bool(torch.isfinite(returns).all()))
        if not valid:
            self.invalid_count += 1
            raise ValueError('invalid/nonfinite completed episode batch')
        order = torch.argsort(ids)
        for k in order.tolist():
            i, lane = int(ids[k]), int(lanes[k])
            self.completed_counts[lane] += 1
            self.duration_steps[lane] += lengths[k]
            self.terminated_count += int(terminated[k])
            self.timeout_count += int(timeouts[k])
            if not self.first_completed[i]:
                self.first_completed[i] = True
                self.excluded += 1
                continue
            self.eligible_counts[lane] += 1
            if lane < 30:
                self.window_counts[lane] += 1
                self.window_sums[lane] += returns[k]
        if self.window_counts.sum() >= self.stage_size and bool((self.window_counts >= 4).all()):
            means = self.window_sums / self.window_counts
            progress = torch.zeros_like(means) if self.previous_means is None else means - self.previous_means
            if not torch.isfinite(means).all() or not torch.isfinite(progress).all():
                self.invalid_count += 1
                raise ValueError('nonfinite learning-progress statistics')
            if self.mode == 'lp':
                self.probabilities = progress_probabilities(progress).to(self.device)
            self.stages.append({'counts': self.window_counts.tolist(), 'mean_returns': means.tolist(),
                                'signed_lp': progress.tolist(), 'probabilities': self.probabilities.cpu().tolist()})
            self.previous_means = means.clone()
            self.window_counts.zero_()
            self.window_sums.zero_()
        sampled = torch.multinomial(self.probabilities, n, replacement=True, generator=self.generator)
        result = torch.empty(n, dtype=torch.long, device=self.device)
        result[order.to(self.device)] = sampled
        self.sampled_counts += torch.bincount(sampled.cpu(), minlength=35)
        return result

    def summary(self):
        return {'mode': self.mode, 'seed': 49017, 'stages': self.stages,
                'stage_updates': len(self.stages), 'stage_size': self.stage_size,
                'probabilities': self.probabilities.cpu().tolist(),
                'window_counts': self.window_counts.tolist(), 'window_return_sums': self.window_sums.tolist(),
                'first_episode_exclusions': self.excluded, 'completed_counts': self.completed_counts.tolist(),
                'sampled_episode_counts': self.sampled_counts.tolist(),
                'completed_duration_steps': self.duration_steps.tolist(),
                'duration_note': 'Includes first randomized episode counters; first completions excluded from LP returns.',
                'eligible_counts': self.eligible_counts.tolist(),
                'terminated_count': self.terminated_count, 'timeout_count': self.timeout_count,
                'invalid_nonfinite_count': self.invalid_count}


def lp_reset_root_state(env, env_ids, pose_range, velocity_range, asset_cfg=None, mode='fixed', stage_size=1024):
    """Reset once on the old lane, then translate the completed subset to new lanes."""
    from week03_ant.tasks.lanes import get_lane_state, lane_reset_root_state
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg('robot')
    ids = (torch.arange(env.num_envs, device=env.device)[env_ids]
           if isinstance(env_ids, slice) else
           torch.arange(env.num_envs, device=env.device) if env_ids is None else
           torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    ids = ids.sort().values
    state = get_lane_state(env)
    if tuple(state.family_names) != FAMILIES or state.num_levels != 5 or state.num_mesh_lanes != 30:
        raise ValueError('v17 requires the frozen seven-family five-level lane layout')
    sampler = getattr(env, '_week03_lp_sampler_v17', None)
    if sampler is None:
        sampler = env._week03_lp_sampler_v17 = LPSamplerV17(env.num_envs, env.device, mode, stage_size)
    if sampler.mode != mode or sampler.stage_size != stage_size:
        raise ValueError('sampler mode/stage_size cannot change mid-run')
    lengths = env.episode_length_buf[ids].clone()
    terminated = env.termination_manager.terminated[ids].clone()
    timeouts = env.termination_manager.time_outs[ids].clone()
    completed = (lengths > 0) & (terminated | timeouts)
    completed_ids = ids[completed]
    old_lanes = state.lane[completed_ids].clone()
    returns = torch.zeros(len(completed_ids), device=env.device, dtype=torch.float64)
    for values in env.reward_manager._episode_sums.values():
        returns += values[completed_ids].double()
    lane_reset_root_state(env, ids, pose_range, velocity_range, asset_cfg)
    if len(completed_ids):
        new_lanes = sampler.complete(completed_ids, old_lanes, returns, lengths[completed],
                                     terminated[completed], timeouts[completed])
        asset = env.scene[asset_cfg.name]
        pose = translated_reset_pose(asset.data.root_link_pose_w[completed_ids], old_lanes, new_lanes,
                                     state.x_entrance, state.center_y, state.ground_z)
        asset.write_root_pose_to_sim(pose, env_ids=completed_ids)
        state.lane[completed_ids] = new_lanes
        state.spawn_x[completed_ids] = pose[:, 0]
