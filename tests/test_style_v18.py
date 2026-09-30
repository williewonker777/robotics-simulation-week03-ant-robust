"""CPU proofs for v18 binary scan mask, loss gradient and frozen interfaces."""

from __future__ import annotations

import importlib.util
import sys

import pytest
import torch
from tensordict import TensorDict

from week03_ant.style_math_v18 import teacher_mask_from_scan
from week03_ant.foothold_math import GRID_RAYS, foothold_grid
from week03_ant.style_policy_v18 import StyleCommandPriorActorCritic, StylePriorPPO
from week03_ant.command_policy import CommandPriorActorCritic
from week03_ant.prior_ppo import PriorPPO, fixed_gaussian_prior_loss
from week03_ant.style_study_v18 import ROOT, V16_CONTROL, prepare_checkpoint

sys.path.insert(0, str(ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("v18_launcher_test", ROOT / "scripts/style_v18.py")
launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launcher)


def _scan(height=0., valid=True):
    h = torch.full((1, GRID_RAYS), float(height))
    v = torch.full_like(h, bool(valid), dtype=torch.bool)
    return h, v


def test_gate_flat_rough_unknown_and_vertical_translation():
    flat, valid = _scan()
    assert teacher_mask_from_scan(flat, valid).item() == 1.
    assert teacher_mask_from_scan(flat, torch.zeros_like(valid)).item() == 1.
    xy = foothold_grid().reshape(-1, 2)
    slope = .18 * xy[:, 0].reshape(1, -1)
    assert teacher_mask_from_scan(slope, valid).item() == 0.
    assert teacher_mask_from_scan(slope + 13.4, valid).item() == 0.
    # A sparse observed stone patch is rough; nearly all unknown is conservative.
    partial = valid.clone()
    roi = ((xy[:, 0] >= 0.) & (xy[:, 0] <= 1.5) & (xy[:, 1].abs() <= .8))
    ids = roi.nonzero().flatten()
    partial[:, ids[::3]] = False
    assert teacher_mask_from_scan(flat, partial).item() == 0.
    sparse = torch.zeros_like(valid)
    sparse[:, ids[:4]] = True
    assert teacher_mask_from_scan(flat, sparse).item() == 1.
    clipped = flat.clone()
    clipped[:, ids] = torch.nan
    assert teacher_mask_from_scan(clipped, valid).item() == 1.


def test_gate_rejects_bad_shapes():
    height, valid = _scan()
    for bad_height, bad_valid in ((height[:, :10], valid), (height, valid.float())):
        with pytest.raises(ValueError):
            teacher_mask_from_scan(bad_height, bad_valid)


def _policy(style_mode):
    obs = {"policy": torch.zeros(4, 91)}
    groups = {"policy": ["policy"], "critic": ["policy"]}
    return StyleCommandPriorActorCritic(
        obs, groups, 8, command_mode="conditioned", prior_mode="anchored",
        input_mode="targets", require_config_match=True, style_mode=style_mode,
    )


def test_checkpoint_compatibility_and_inference_without_gate():
    source = torch.load(V16_CONTROL, map_location="cpu", weights_only=False)["model_state_dict"]
    policy = _policy("gated")
    assert source.keys() == policy.state_dict().keys()
    policy.load_state_dict(source)
    baseline = CommandPriorActorCritic(
        {"policy": torch.zeros(4, 91)}, {"policy": ["policy"], "critic": ["policy"]}, 8,
        command_mode="conditioned", prior_mode="anchored", input_mode="targets",
        require_config_match=True,
    )
    baseline.load_state_dict(source)
    obs = {"policy": torch.randn(4, 91)}
    with torch.no_grad():
        assert torch.equal(policy.act_inference(obs), baseline.act_inference(obs))
        assert torch.equal(policy.evaluate(obs), baseline.evaluate(obs))
        assert torch.equal(policy.teacher_mean(obs), baseline.teacher_mean(obs))


def test_binary_target_equals_explicit_masked_prior_loss_and_gradient():
    policy = _policy("gated")
    obs = {"policy": torch.randn(4, 91), "style_gate": torch.tensor([[1.], [0.], [1.], [0.]])}
    policy.act(obs)
    student = policy.action_mean
    teacher = CommandPriorActorCritic.teacher_mean(policy, obs)
    policy._style_loss_active = True
    target = policy.teacher_mean(obs)
    assert torch.equal(target[::2], teacher[::2])
    assert torch.equal(target[1::2], student.detach()[1::2])
    masked = ((student - teacher).square() * obs["style_gate"]) / (2 * .2**2)
    obtained = fixed_gaussian_prior_loss(student, target)
    expected = masked.mean()
    assert torch.allclose(obtained, expected)
    gradient, = torch.autograd.grad(obtained, student, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, student)
    assert torch.allclose(gradient, expected_gradient)
    assert torch.equal(gradient[1::2], torch.zeros_like(gradient[1::2]))


def test_always_matches_real_teacher_even_during_update():
    policy = _policy("always")
    obs = {"policy": torch.randn(4, 91), "style_gate": torch.tensor([[1.], [0.], [0.], [1.]])}
    policy.act(obs)
    policy._style_loss_active = True
    assert torch.equal(policy.teacher_mean(obs), CommandPriorActorCritic.teacher_mean(policy, obs))
    assert (policy._style_gate_sum, policy._style_gate_count) == (2, 4)


@pytest.mark.parametrize("bad", [torch.tensor([[2.], [0.], [1.], [0.]]), torch.zeros(4, 2)])
def test_invalid_training_mask_rejected(bad):
    policy = _policy("gated")
    obs = {"policy": torch.randn(4, 91), "style_gate": bad}
    policy.act(obs)
    policy._style_loss_active = True
    with pytest.raises(ValueError):
        policy.teacher_mean(obs)


def test_launch_contract_and_shared_checkpoint(tmp_path, monkeypatch):
    a = launcher.training_arguments([], "always")
    b = launcher.training_arguments([], "gated")
    assert [(x, y) for x, y in zip(a, b) if x != y] == [
        ("agent.policy.style_mode=always", "agent.policy.style_mode=gated")]
    assert "Week03-Ant-Style-v18-Train-v0" in a
    for changed in ("env.rewards.contact_slip.weight=1.0", "agent.algorithm.teacher_coef=0.0",
                    "agent.policy.command_mode=masked", "env.events.reset_base.func=other"):
        with pytest.raises(ValueError):
            launcher.training_arguments([changed], "gated")
    base = ["--seed", "50", "--resume", "--load_run", "v18_init", "--checkpoint", "model_0.pt",
            "env.scene.terrain.terrain_generator.seed=104"]
    for envs, iters in ((256, 16), (4096, 2), (4096, 250)):
        launcher.validate_budget(base + ["--num_envs", str(envs), "--max_iterations", str(iters)])
    with pytest.raises(ValueError):
        launcher.validate_budget(base + ["--num_envs", "4096", "--max_iterations", "251"])
    import week03_ant.style_study_v18 as study

    monkeypatch.setattr(study, "ROOT", tmp_path)
    prepared = prepare_checkpoint(tmp_path / "model_0.pt")
    assert prepared["all_non_std_tensors_identical"] and prepared["optimizer_state_empty"]
    state = torch.load(tmp_path / "model_0.pt", map_location="cpu", weights_only=False)
    assert state["iter"] == 0 and not state["optimizer_state_dict"]["state"]


def _algorithm(cls, policy):
    return cls(
        policy, teacher_coef=.02, num_learning_epochs=2, num_mini_batches=2,
        clip_param=.2, gamma=.995, lam=.95, value_loss_coef=1., entropy_coef=.002,
        learning_rate=1.e-4, max_grad_norm=1., use_clipped_value_loss=True,
        schedule="fixed", desired_kl=None, device="cpu",
        normalize_advantage_per_mini_batch=False,
    )


def _rollout(policy):
    generator = torch.Generator().manual_seed(18018)
    obs = TensorDict({
        "policy": torch.randn(3, 4, 91, generator=generator),
        "style_gate": torch.tensor([[[1.], [0.], [1.], [0.]],
                                    [[0.], [1.], [0.], [1.]],
                                    [[1.], [1.], [0.], [0.]]]),
    }, batch_size=[3, 4])
    flat = obs.flatten(0, 1)
    with torch.no_grad():
        torch.manual_seed(18019)
        actions = policy.act(flat).reshape(3, 4, 8)
        log_prob = policy.get_actions_log_prob(actions.flatten(0, 1)).reshape(3, 4, 1)
        mu = policy.action_mean.reshape(3, 4, 8).clone()
        sigma = policy.action_std.reshape(3, 4, 8).clone()
        values = policy.evaluate(flat).reshape(3, 4, 1)
    return {"observations": obs, "actions": actions, "values": values,
            "actions_log_prob": log_prob, "mu": mu, "sigma": sigma,
            "returns": torch.randn(3, 4, 1, generator=generator),
            "advantages": torch.randn(3, 4, 1, generator=generator),
            "rewards": torch.randn(3, 4, 1, generator=generator),
            "dones": torch.zeros(3, 4, 1, dtype=torch.uint8)}


def _fill(algorithm, data):
    algorithm.init_storage("rl", 4, 3, data["observations"][0], [8])
    for name, value in data.items():
        getattr(algorithm.storage, name).copy_(value)
    algorithm.storage.step = 3


def test_always_update_exactly_matches_frozen_prior_ppo():
    torch.manual_seed(18020)
    candidate = _policy("always")
    reference = CommandPriorActorCritic(
        {"policy": torch.zeros(4, 91)}, {"policy": ["policy"], "critic": ["policy"]}, 8,
        command_mode="conditioned", prior_mode="anchored", input_mode="targets",
        require_config_match=True,
    )
    reference.load_state_dict(candidate.state_dict())
    control = _algorithm(PriorPPO, reference)
    experiment = _algorithm(StylePriorPPO, candidate)
    data = _rollout(reference)
    _fill(control, data)
    _fill(experiment, data)
    torch.manual_seed(18021)
    expected = control.update()
    torch.manual_seed(18021)
    actual = experiment.update()
    for key, value in expected.items():
        assert actual[key] == value, key
    assert actual["style_teacher_mask_fraction"] == .5
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], atol=0, rtol=0)


def test_rollout_minibatch_keeps_policy_and_style_gate_paired():
    policy = _policy("gated")
    algorithm = _algorithm(StylePriorPPO, policy)
    data = _rollout(policy)
    marker = data["observations"]["style_gate"].clone()
    data["observations"]["policy"][..., 0:1] = marker
    _fill(algorithm, data)
    for batch in algorithm.storage.mini_batch_generator(2, 2):
        selected = batch[0]
        assert torch.equal(selected["policy"][:, 0:1], selected["style_gate"])
