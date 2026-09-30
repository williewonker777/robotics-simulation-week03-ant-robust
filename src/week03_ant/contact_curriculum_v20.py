"""Pure global-policy-step contact coefficient; no RNG or episode state."""


def coefficient(mode: str, common_step_counter: int, ramp_steps: int = 4000) -> float:
    if mode not in ('immediate', 'ramped'):
        raise ValueError('unknown contact curriculum mode')
    if type(common_step_counter) is not int or common_step_counter < 1:
        raise ValueError('common step counter must be a positive integer')
    if type(ramp_steps) is not int or ramp_steps < 2:
        raise ValueError('ramp steps must be an integer >= 2')
    return 1. if mode == 'immediate' else min((common_step_counter - 1) / (ramp_steps - 1), 1.)


def audit_schedule(record, *, mode, policy_steps, ramp_steps):
    """Recompute every actually applied coefficient and bounded reward summary."""
    import math

    if type(policy_steps) is not int or policy_steps <= 0:
        raise ValueError('positive policy-step budget required')
    coefficient(mode, 1, ramp_steps)
    if (record.get('mode') != mode or record.get('ramp_steps') != ramp_steps
            or record.get('policy_steps') != policy_steps):
        raise ValueError('schedule metadata differs')
    counters = record.get('common_step_counters')
    expected = list(range(1, policy_steps + 1))
    if counters != expected or any(type(v) is not int for v in counters):
        raise ValueError('global reward clock incomplete or reset')
    factors = [coefficient(mode, step, ramp_steps) for step in expected]
    if (record.get('coefficients') != factors
            or any(type(v) not in (int, float) for v in record.get('coefficients', []))):
        raise ValueError('actual coefficient trajectory differs')
    for key in ('raw_reward_mean', 'scaled_reward_mean'):
        values = record.get(key)
        if (not isinstance(values, list) or len(values) != policy_steps
                or any(type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 0 for v in values)):
            raise ValueError('invalid applied reward evidence')
    if any(not math.isclose(scaled, raw * factor, abs_tol=2.e-7, rel_tol=2.e-6)
           for raw, scaled, factor in zip(record['raw_reward_mean'], record['scaled_reward_mean'], factors)):
        raise ValueError('scaled reward does not match coefficient')
    if not math.isclose(record.get('coefficient_sum', -1), sum(factors), abs_tol=1.e-9):
        raise ValueError('coefficient dose differs')
    envs = record.get('num_envs')
    if type(envs) is not int or envs <= 0:
        raise ValueError('missing environment count')
    for key in ('valid_rows', 'step_dt', 'raw_reward_dt_sum', 'scaled_reward_dt_sum'):
        if not isinstance(record.get(key), list) or len(record[key]) != policy_steps:
            raise ValueError('missing reward-state accounting: ' + key)
    for i, factor in enumerate(factors):
        valid, dt = record['valid_rows'][i], record['step_dt'][i]
        if type(valid) is not int or not 0 <= valid <= envs or not math.isclose(dt, 1 / 60):
            raise ValueError('invalid validity/dt evidence')
        for prefix in ('raw', 'scaled'):
            total = record[prefix + '_reward_dt_sum'][i]
            expected_sum = record[prefix + '_reward_mean'][i] * envs * dt
            if not math.isfinite(total) or not math.isclose(total, expected_sum, abs_tol=2.e-7, rel_tol=2.e-6):
                raise ValueError('realized reward dt accounting differs')
    if not math.isclose(record.get('realized_penalty_sum', -1), -sum(record['scaled_reward_dt_sum']),
                        abs_tol=1.e-9, rel_tol=1.e-9):
        raise ValueError('realized penalty total differs')
    return True
