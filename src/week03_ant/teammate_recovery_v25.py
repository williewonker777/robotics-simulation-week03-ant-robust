"""Stick-inspired continuous reward rates; RewardManager alone integrates dt.

Adapted concepts: Stick-0/isaac-ant-rough-terrain at 3cc718a (BSD-3-Clause).
No simulator dependencies, filtering, terrain labels, or action-observation reuse.
"""
import torch


def recovery_terms(clearance, target, valid, upright, action, previous_action, angular_velocity):
    n = clearance.shape[0] if clearance.ndim == 1 else -1
    vectors = (clearance, target, upright)
    if (n < 1 or any(x.shape != (n,) for x in vectors) or valid.shape != (n,)
            or valid.dtype != torch.bool or action.shape != (n, 8)
            or previous_action.shape != (n, 8) or angular_velocity.shape != (n, 3)):
        raise ValueError('expected scalar[N], boolean valid[N], action[N,8], angular velocity[N,3]')
    tensors = (*vectors, valid, action, previous_action, angular_velocity)
    if len({x.device for x in tensors}) != 1 or any(not x.is_floating_point() for x in (*vectors, action, previous_action, angular_velocity)):
        raise ValueError('inputs must share device and physical values must be floating point')
    # Invalid scans may supply nonfinite clearance/target; physical state never may.
    if (any(not torch.isfinite(x).all() for x in (upright, action, previous_action, angular_velocity))
            or not torch.isfinite(clearance[valid]).all() or not torch.isfinite(target[valid]).all()
            or bool((target[valid] <= .31).any())):
        raise ValueError('nonfinite physical state or invalid valid-scan target')
    safe = torch.where(valid, target.clamp(max=.48), torch.full_like(target, .44))
    measured = torch.where(valid, clearance, safe)
    height_risk = ((safe - measured) / (safe - .31)).clamp(0, 1).square()
    tilt_risk = ((.93 - upright) / (.93 - .5)).clamp(0, 1).square()
    action_delta = (action - previous_action).square().sum(-1)
    angular_xy = angular_velocity[:, :2].square().sum(-1)
    rate = -2 * height_risk -2 * tilt_risk -.01 * action_delta -.025 * angular_xy
    if not torch.isfinite(rate).all():
        raise ValueError('nonfinite recovery rate')
    return dict(rate=rate, clearance_risk=height_risk, tilt_risk=tilt_risk,
                action_delta_squared_sum=action_delta, angular_xy_squared_sum=angular_xy,
                clearance_abstained=~valid, safe_height=safe)


def applied_rate(terms, enabled):
    if type(enabled) is not bool:
        raise ValueError('enabled must be a boolean treatment')
    return terms['rate'] if enabled else torch.zeros_like(terms['rate'])
