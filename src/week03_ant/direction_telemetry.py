"""Passive directional diagnostics; invalid samples never enter tracking means."""

import math
import torch

FIELDS = ('absolute_lateral_velocity', 'absolute_heading_error', 'absolute_yaw_error', 'reward')


class DirectionTelemetry:
    def __init__(self, num_envs, device):
        self.active_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.valid_steps = torch.zeros_like(self.active_steps)
        self.sums = {key + '_sum': torch.zeros(num_envs, dtype=torch.float64, device=device) for key in FIELDS}

    @torch.inference_mode()
    def before_step(self, active, metrics):
        if active.shape != self.active_steps.shape or active.dtype != torch.bool:
            raise ValueError('expected boolean first-episode active mask')
        valid = metrics['valid']
        if valid.shape != active.shape or valid.dtype != torch.bool:
            raise ValueError('expected boolean direction validity')
        selected = active & valid
        values = dict(absolute_lateral_velocity=metrics['lateral_velocity'].abs(),
                      absolute_heading_error=metrics['heading_error'].abs(),
                      absolute_yaw_error=metrics['yaw_error'].abs(), reward=metrics['reward'])
        for key, value in values.items():
            if value.shape != active.shape or not torch.isfinite(value[selected]).all():
                raise ValueError('malformed/nonfinite valid directional metric')
        reward = values['reward'][selected]
        if ((reward < -1.) | (reward > 0.)).any() or (values['absolute_heading_error'][selected] > math.pi + 1.e-6).any():
            raise ValueError('direction metric outside bounds')
        self.active_steps += active.long()
        self.valid_steps += selected.long()
        for key, value in values.items():
            self.sums[key + '_sum'] += torch.where(selected, value, 0.).double()

    def to_dict(self, episode_lengths):
        if not torch.equal(self.active_steps, episode_lengths):
            raise ValueError('direction duration differs from first episodes')
        return dict(pre_action_first_episode_only=True, reward_unweighted_not_dt_integrated=True,
                    active_steps=self.active_steps.cpu().tolist(), valid_steps=self.valid_steps.cpu().tolist(),
                    **{key: value.cpu().tolist() for key, value in self.sums.items()})


def audit_direction(data, rows):
    telemetry = data.get('direction_telemetry')
    keys = {'pre_action_first_episode_only', 'reward_unweighted_not_dt_integrated',
            'active_steps', 'valid_steps', *(key + '_sum' for key in FIELDS)}
    if not isinstance(telemetry, dict) or set(telemetry) != keys:
        raise ValueError('unexpected direction telemetry schema')
    if telemetry['pre_action_first_episode_only'] is not True or telemetry['reward_unweighted_not_dt_integrated'] is not True:
        raise ValueError('direction sampling/reward convention changed')
    arrays = {k: v for k, v in telemetry.items() if k not in ('pre_action_first_episode_only', 'reward_unweighted_not_dt_integrated')}
    if any(not isinstance(v, list) or len(v) != len(rows) for v in arrays.values()):
        raise ValueError('malformed direction telemetry arrays')
    for i, row in enumerate(rows):
        state = {key: value[i] for key, value in arrays.items()}
        active, valid = state['active_steps'], state['valid_steps']
        if type(active) is not int or type(valid) is not int or active != row['steps'] or not 0 <= valid <= active:
            raise ValueError('direction count/coverage mismatch')
        for key in FIELDS:
            value = state[key + '_sum']
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError('nonfinite direction sums')
            low = -valid if key == 'reward' else 0
            high = 0 if key == 'reward' else (math.pi + 1.e-6) * valid if key == 'absolute_heading_error' else math.inf
            if value < low - 1.e-7 or value > high + 1.e-7 or (valid == 0 and value != 0):
                raise ValueError('direction sum outside count/physical bounds')
        row['direction'] = state


def aggregate_direction(rows):
    sums = {key: sum(row['direction'][key] for row in rows)
            for key in ('active_steps', 'valid_steps', *(field + '_sum' for field in FIELDS))}
    valid, active = sums['valid_steps'], sums['active_steps']
    return dict(sums=sums, means={field: sums[field + '_sum'] / valid if valid else None for field in FIELDS},
                valid_coverage=valid / active if active else None, invalid_steps=active - valid)
