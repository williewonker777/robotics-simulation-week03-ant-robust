import copy

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from week03_ant.prior_policy import PriorActorCritic
from week03_ant.prior_ppo import PriorPPO, fixed_gaussian_prior_loss


GROUPS = {"policy": ["policy"], "critic": ["policy"]}


def make_policy(mode="free"):
    obs = TensorDict({"policy": torch.randn(4, 88)}, batch_size=[4])
    return PriorActorCritic(obs, GROUPS, 8, prior_mode=mode, init_noise_std=0.2)


def test_fixed_gaussian_loss_zero_scale_finite_and_gradient_boundaries():
    student = torch.zeros(3, 8, requires_grad=True)
    teacher = torch.zeros(3, 8, requires_grad=True)
    assert fixed_gaussian_prior_loss(student, teacher).item() == 0
    shifted = torch.full((3, 8), 0.2, requires_grad=True)
    loss = fixed_gaussian_prior_loss(shifted, teacher)
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert torch.isfinite(shifted.grad).all() and shifted.grad.abs().sum() > 0
    assert teacher.grad is None
    live_std = torch.tensor(9.0, requires_grad=True)
    (fixed_gaussian_prior_loss(shifted, teacher) + 0.0 * live_std).backward()
    assert live_std.grad.item() == 0.0
    with pytest.raises(ValueError):
        fixed_gaussian_prior_loss(student, teacher, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"teacher_coef": 0.01, "schedule": "fixed", "desired_kl": None},
        {"teacher_coef": 0.0, "schedule": "adaptive", "desired_kl": None},
        {"teacher_coef": 0.0, "schedule": "fixed", "desired_kl": 0.01},
        {"teacher_coef": 0.0, "schedule": "fixed", "desired_kl": None, "rnd_cfg": {}},
        {"teacher_coef": 0.0, "schedule": "fixed", "desired_kl": None, "symmetry_cfg": {}},
        {"teacher_coef": 0.0, "schedule": "fixed", "desired_kl": None, "multi_gpu_cfg": {}},
    ],
)
def test_rejects_invalid_or_unsupported_configuration(kwargs):
    with pytest.raises((ValueError, KeyError)):
        PriorPPO(make_policy(), **kwargs)


def fill_storage(algorithm, data):
    obs = TensorDict({"policy": torch.zeros(3, 4, 88)}, batch_size=[3, 4])
    algorithm.init_storage("rl", 4, 3, obs[0], [8])
    for key, value in data.items():
        getattr(algorithm.storage, key).copy_(value)
    algorithm.storage.step = 3


def storage_data(policy):
    generator = torch.Generator().manual_seed(91)
    obs = TensorDict({"policy": torch.randn(3, 4, 88, generator=generator)}, batch_size=[3, 4])
    flat = obs.flatten(0, 1)
    with torch.no_grad():
        torch.manual_seed(92)
        actions = policy.act(flat).reshape(3, 4, 8)
        log_prob = policy.get_actions_log_prob(actions.flatten(0, 1)).reshape(3, 4, 1)
        mu = policy.action_mean.reshape(3, 4, 8).clone()
        sigma = policy.action_std.reshape(3, 4, 8).clone()
        values = policy.evaluate(flat).reshape(3, 4, 1)
    return {
        "observations": obs,
        "actions": actions,
        "values": values,
        "actions_log_prob": log_prob,
        "mu": mu,
        "sigma": sigma,
        "returns": torch.randn(3, 4, 1, generator=generator),
        "advantages": torch.randn(3, 4, 1, generator=generator),
        "rewards": torch.randn(3, 4, 1, generator=generator),
        "dones": torch.zeros(3, 4, 1, dtype=torch.uint8),
    }


def algorithm(cls, policy, coef=0.0):
    kwargs = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=None,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
    )
    return cls(policy, teacher_coef=coef, **kwargs) if cls is PriorPPO else cls(policy, **kwargs)


def test_free_one_update_numerical_parity_with_installed_ppo():
    torch.manual_seed(5)
    reference_policy = make_policy("free")
    prior_policy = copy.deepcopy(reference_policy)
    reference = algorithm(PPO, reference_policy)
    candidate = algorithm(PriorPPO, prior_policy, 0.0)
    data = storage_data(reference_policy)
    fill_storage(reference, data)
    fill_storage(candidate, data)
    torch.manual_seed(100)
    expected_losses = reference.update()
    torch.manual_seed(100)
    actual_losses = candidate.update()
    for key in ("value_function", "surrogate", "entropy"):
        assert actual_losses[key] == expected_losses[key]
    assert actual_losses["weighted_prior"] == 0.0
    for key, value in reference_policy.state_dict().items():
        torch.testing.assert_close(value, prior_policy.state_dict()[key], atol=0, rtol=0)


def test_anchored_update_keeps_teacher_bit_identical_and_logs_metrics():
    torch.manual_seed(7)
    policy = make_policy("anchored")
    before = {key: value.clone() for key, value in policy.teacher.state_dict().items()}
    algo = algorithm(PriorPPO, policy, 0.02)
    fill_storage(algo, storage_data(policy))
    losses = algo.update()
    assert set(("prior_loss", "prior_mse", "weighted_prior", "action_mean_abs_gt1_fraction",
                "student_action_mean_rms", "teacher_action_mean_rms")) <= losses.keys()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    for key, value in before.items():
        torch.testing.assert_close(value, policy.teacher.state_dict()[key], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in policy.teacher.parameters())


def test_update_rejects_mode_coefficient_mismatch():
    algo = algorithm(PriorPPO, make_policy("anchored"), 0.0)
    with pytest.raises(ValueError, match="mode"):
        algo.update()

