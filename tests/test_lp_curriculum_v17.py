"""CPU proofs for signed LP, episode accounting, and reset geometry."""
import json
import sys
import types

import pytest
import torch

from week03_ant.lp_curriculum_v17 import (
    FAMILIES, LPSamplerV17, base_probabilities, lp_reset_root_state,
    progress_probabilities, translated_reset_pose,
)


def batch(sampler, returns=None, lanes=None, reverse=False):
    n = len(sampler.first_completed)
    ids = torch.arange(n)
    lanes = ids % 30 if lanes is None else lanes
    returns = torch.ones(n) if returns is None else returns
    order = ids.flip(0) if reverse else ids
    return sampler.complete(ids[order], lanes[order], returns[order], torch.full((n,), 10)[order],
                            torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))


def test_base_and_flat_mass():
    p = base_probabilities()
    assert p.sum().item() == pytest.approx(1)
    assert p[30:].sum().item() == pytest.approx(2 / 11)
    torch.testing.assert_close(p[25:30], p[:5] * 4)
    torch.testing.assert_close(progress_probabilities(torch.zeros(30)), p)


def test_signed_progress_floor_clip_and_flat_invariant():
    lp = torch.zeros(30); lp[0] = 10; lp[1] = -10
    p, base = progress_probabilities(lp), base_probabilities()
    assert p[0] > p[2] > p[1]
    assert (p[:30] >= .4 * base[:30]).all()
    assert p.sum().item() == pytest.approx(1)
    torch.testing.assert_close(p[30:], base[30:])
    lp[0] = 10000; lp[1] = -10000
    torch.testing.assert_close(progress_probabilities(lp), p)


@pytest.mark.parametrize('bad', [torch.zeros(29), torch.full((30,), float('nan')),
                                 torch.full((30,), float('inf'))])
def test_invalid_progress(bad):
    with pytest.raises(ValueError):
        progress_probabilities(bad)


def test_first_completions_resample_but_do_not_credit():
    s = LPSamplerV17(1050, 'cpu', 'lp')
    batch(s)
    assert s.excluded == 1050 and not s.window_counts.any() and not s.stages
    assert s.sampled_counts.sum() == 1050
    batch(s)
    assert len(s.stages) == 1
    assert s.stages[0]['counts'] == [35] * 30
    assert s.stages[0]['mean_returns'] == [1.] * 30
    assert s.stages[0]['signed_lp'] == [0.] * 30
    values = torch.ones(1050); values[torch.arange(1050) % 30 == 0] = 6
    batch(s, values)
    assert len(s.stages) == 2 and s.stages[1]['signed_lp'][0] == 5
    assert s.probabilities[0] > s.probabilities[1]
    json.dumps(s.summary(), allow_nan=False)


def test_missing_task_delays_and_flat_does_not_trigger_stage():
    s = LPSamplerV17(1050, 'cpu', 'lp')
    lanes = torch.arange(1050) % 29
    batch(s, lanes=lanes); batch(s, lanes=lanes)
    assert s.window_counts.sum() == 1050 and not s.stages
    batch(s, lanes=torch.full((1050,), 30))
    assert s.window_counts.sum() == 1050 and not s.stages
    batch(s, lanes=torch.full((1050,), 29))
    assert len(s.stages) == 1 and sum(s.stages[0]['counts']) == 2100


def test_minimum_total_and_per_task_gate():
    s = LPSamplerV17(120, 'cpu', 'lp')
    batch(s)
    for _ in range(8):
        batch(s)
    assert not s.stages and s.window_counts.sum() == 960
    batch(s)
    assert len(s.stages) == 1


def test_batch_permutation_invariance_and_private_rng():
    a, b = LPSamplerV17(1050, 'cpu', 'lp'), LPSamplerV17(1050, 'cpu', 'lp')
    for j in range(3):
        before = torch.get_rng_state().clone()
        x = batch(a, returns=torch.arange(1050).float() * j)
        assert torch.equal(before, torch.get_rng_state())
        y = batch(b, returns=torch.arange(1050).float() * j, reverse=True)
        assert torch.equal(x, y.flip(0))
        assert a.summary() == b.summary()


def test_fixed_arm_updates_ledger_not_probabilities():
    s = LPSamplerV17(1050, 'cpu', 'fixed')
    for j in range(3):
        batch(s, returns=torch.arange(1050).float() * j)
    assert len(s.stages) == 2 and any(s.stages[1]['signed_lp'])
    torch.testing.assert_close(s.probabilities, base_probabilities())


@pytest.mark.parametrize('field', ['returns', 'ids', 'lengths', 'lanes', 'termination'])
def test_invalid_batch_fails_closed(field):
    s = LPSamplerV17(2, 'cpu')
    args = dict(ids=torch.arange(2), old_lanes=torch.arange(2), returns=torch.ones(2),
                lengths=torch.ones(2), terminated=torch.ones(2).bool(), timeouts=torch.zeros(2).bool())
    if field == 'returns': args['returns'][0] = float('nan')
    if field == 'ids': args['ids'][1] = 0
    if field == 'lengths': args['lengths'][0] = 0
    if field == 'lanes': args['old_lanes'][0] = 35
    if field == 'termination': args['terminated'][0] = False
    with pytest.raises(ValueError): s.complete(**args)
    assert s.invalid_count == 1 and s.sampled_counts.sum() == 0


def test_translation_preserves_jitter_quaternion_velocity_and_odometer():
    pose = torch.tensor([[10.2, 20.4, .6, 1., 0., 0., 0., 3., 4., 5., 6., 7., 8.]])
    moved = translated_reset_pose(pose, torch.tensor([0]), torch.tensor([1]),
                                  torch.tensor([10., 0.]), torch.tensor([20., 0.]),
                                  torch.tensor([0., -50.]))
    torch.testing.assert_close(moved[0, :3], torch.tensor([.2, .4, -49.4]))
    torch.testing.assert_close(moved[:, 3:], pose[:, 3:])
    spawn_x = moved[:, 0].clone()
    assert (moved[:, 0] - spawn_x).item() == 0
    # RewardManager resets its previous XY after reset events, so teleport is not progress.
    prev_xy = moved[:, :2].clone()
    assert torch.equal(moved[:, :2] - prev_xy, torch.zeros(1, 2))


def fake_env(monkeypatch):
    ns = types.SimpleNamespace
    pose = torch.zeros(3, 7); pose[:, 3] = 1
    pose[:, 0] = torch.tensor([7., 8., 9.])
    asset = ns(data=ns(root_link_pose_w=pose))
    asset.write_root_pose_to_sim = lambda value, env_ids: pose.__setitem__(env_ids, value)
    state = ns(family_names=FAMILIES, num_levels=5, num_mesh_lanes=30,
               lane=torch.tensor([0, 10, 30]), spawn_x=torch.zeros(3),
               x_entrance=torch.arange(35).float() * 10,
               center_y=torch.arange(35).float() * 100,
               ground_z=torch.cat((torch.zeros(30), torch.full((5,), -50.))),
               last_distance=torch.zeros(3))
    env = ns(num_envs=3, device='cpu', scene={'robot': asset}, state=state,
             episode_length_buf=torch.zeros(3, dtype=torch.long),
             reward_manager=ns(_episode_sums={'one': torch.tensor([3., 7., 11.]),
                                            'two': torch.tensor([2., 4., 5.])}),
             termination_manager=ns(terminated=torch.zeros(3).bool(), time_outs=torch.zeros(3).bool()),
             calls=0)
    def reset(env, ids, pose_range, velocity_range, asset_cfg):
        env.calls += 1
        state.last_distance[ids] = pose[ids, 0] - state.spawn_x[ids]
        lane = state.lane[ids]
        pose[ids, 0] = state.x_entrance[lane] + .1
        pose[ids, 1] = state.center_y[lane] + .2
        pose[ids, 2] = state.ground_z[lane] + .5
        state.spawn_x[ids] = pose[ids, 0]
    monkeypatch.setitem(sys.modules, 'week03_ant.tasks.lanes', ns(get_lane_state=lambda env: state,
                                                                lane_reset_root_state=reset))
    monkeypatch.setitem(sys.modules, 'isaaclab.managers', ns(SceneEntityCfg=lambda name: ns(name=name)))
    return env


def test_wrapper_initial_reset_and_old_lane_credit(monkeypatch):
    env = fake_env(monkeypatch)
    initial = env.state.lane.clone()
    lp_reset_root_state(env, None, {}, {}, mode='lp')
    s = env._week03_lp_sampler_v17
    assert env.calls == 1 and s.sampled_counts.sum() == 0
    assert torch.equal(env.state.lane, initial)
    untouched = env.scene['robot'].data.root_link_pose_w[2].clone()
    env.episode_length_buf[:2] = 123
    env.termination_manager.terminated[:2] = True
    lp_reset_root_state(env, [1, 0], {}, {}, mode='lp')
    assert env.calls == 2 and s.excluded == 2
    old = env.state.lane[:2].clone()
    lp_reset_root_state(env, [0, 1], {}, {}, mode='lp')
    assert s.completed_counts.sum() == 4 and s.eligible_counts.sum() == 2
    for lane in old.tolist():
        assert s.eligible_counts[lane] >= 1
    expected = torch.zeros(30, dtype=torch.float64)
    for lane, value in zip(old.tolist(), [5., 11.]):
        if lane < 30: expected[lane] += value
    torch.testing.assert_close(s.window_sums, expected)
    torch.testing.assert_close(env.scene['robot'].data.root_link_pose_w[2], untouched)
    torch.testing.assert_close(env.state.spawn_x[:2], env.scene['robot'].data.root_link_pose_w[:2, 0])
    assert env.state.lane[2] == initial[2]


def test_nonterminal_manual_reset_not_completion(monkeypatch):
    env = fake_env(monkeypatch); env.episode_length_buf[:] = 50
    lp_reset_root_state(env, [0, 1], {}, {})
    assert env._week03_lp_sampler_v17.sampled_counts.sum() == 0


@pytest.mark.parametrize('stage_size', [0, -1, 1.5, True, '16', None])
def test_stage_size_invalid(stage_size):
    with pytest.raises(ValueError, match='positive integer'):
        LPSamplerV17(120, 'cpu', 'lp', stage_size=stage_size)


def test_default_stage_size_and_development_coverage_gate():
    assert LPSamplerV17(1, 'cpu').summary()['stage_size'] == 1024
    s = LPSamplerV17(30, 'cpu', 'lp', stage_size=16)
    assert s.summary()['stage_size'] == 16
    batch(s)  # Excluded first completions.
    for _ in range(3):
        batch(s)
        assert not s.stages  # Total >=16 does not waive four samples/task.
    batch(s)
    assert s.summary()['stage_updates'] == 1
    assert s.stages[0]['counts'] == [4] * 30


def test_wrapper_development_stage_size_forwarded(monkeypatch):
    env = fake_env(monkeypatch)
    lp_reset_root_state(env, None, {}, {}, mode='lp', stage_size=16)
    assert env._week03_lp_sampler_v17.summary()['stage_size'] == 16
    with pytest.raises(ValueError, match='cannot change'):
        lp_reset_root_state(env, None, {}, {}, mode='lp', stage_size=1024)
