import hashlib
from pathlib import Path

import pytest
import torch

from rsl_rl.modules import ActorCritic

from scripts.prepare_prior_checkpoint import SOURCE_SHA256, prepare_checkpoint
from week03_ant.prior_policy import (
    PRIOR_MODES,
    PriorActorCritic,
    register_prior_components,
    warmstart_prior_from_60d,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
GROUPS = {"policy": ["policy"], "critic": ["policy"]}


def observations(batch=7):
    return {"policy": torch.randn(batch, 88)}


def policy(mode="free", **kwargs):
    return PriorActorCritic(
        observations(), GROUPS, 8, prior_mode=mode, input_mode="targets", init_noise_std=0.2, **kwargs
    )


def test_real_v5_student_value_and_teacher_identity():
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)["model_state_dict"]
    obs = observations(13)
    old_obs = {"policy": obs["policy"][:, :60]}
    old = ActorCritic(
        old_obs,
        GROUPS,
        8,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )
    old.load_state_dict(source)
    new = policy()
    warmstart_prior_from_60d(new, source)
    torch.testing.assert_close(new.act_inference(obs), old.act_inference(old_obs), atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(new.evaluate(obs), old.evaluate(old_obs), atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(new.teacher_mean(obs), old.act_inference(old_obs), atol=0, rtol=0)
    assert new.actor[0].weight[:, 60:].eq(0).all()
    assert new.critic[0].weight[:, 60:].eq(0).all()


def test_teacher_frozen_eval_and_inference_does_not_call_it(monkeypatch):
    model = policy()
    model.train()
    assert not model.teacher.training
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())
    monkeypatch.setattr(model.teacher, "forward", lambda _: (_ for _ in ()).throw(AssertionError()))
    assert model.act_inference(observations()).shape == (7, 8)


def test_mode_load_restore_guard_and_persisted_scalars():
    state = policy("anchored").state_dict()
    assert state["prior_mode_code"].dtype == torch.int64
    assert state["prior_teacher_coef"].dtype == torch.float32
    assert state["prior_reference_std"].dtype == torch.float32
    evaluator = policy("free")
    evaluator.load_state_dict(state)
    assert evaluator.prior_mode == "anchored"
    with pytest.raises(ValueError, match="training config"):
        policy("free", require_config_match=True).load_state_dict(state)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("prior_mode_code", None),
        ("prior_mode_code", torch.tensor(9, dtype=torch.int64)),
        ("prior_teacher_coef", torch.tensor(0.01, dtype=torch.float32)),
        ("prior_reference_std", torch.tensor(0.3, dtype=torch.float32)),
    ],
)
def test_missing_or_invalid_contract_state_rejected(key, bad):
    model = policy()
    state = model.state_dict()
    if bad is None:
        del state[key]
    else:
        state[key] = bad
    with pytest.raises(ValueError):
        model.load_state_dict(state)
    with pytest.raises(ValueError):
        model.load_state_dict(model.state_dict(), strict=False)


def test_dimensions_modes_normalization_and_std_rejected():
    with pytest.raises(ValueError, match="targets"):
        PriorActorCritic(observations(), GROUPS, 8, prior_mode="free", input_mode="feet")
    with pytest.raises(ValueError):
        PriorActorCritic({"policy": torch.zeros(2, 60)}, GROUPS, 8)
    with pytest.raises(ValueError):
        PriorActorCritic(observations(), GROUPS, 7)
    with pytest.raises(ValueError, match="normalization"):
        PriorActorCritic(observations(), GROUPS, 8, actor_obs_normalization=True)
    with pytest.raises(ValueError):
        warmstart_prior_from_60d(policy(), {}, std=float("nan"))


def test_paired_checkpoint_identity_and_source_preservation(tmp_path):
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert before == SOURCE_SHA256
    paths = [tmp_path / f"{mode}.pt" for mode in PRIOR_MODES]
    for mode, path in zip(PRIOR_MODES, paths):
        prepare_checkpoint(SOURCE, path, mode)
    states = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    differing = {"prior_mode_code", "prior_teacher_coef"}
    for key, value in states[0]["model_state_dict"].items():
        other = states[1]["model_state_dict"][key]
        if key in differing:
            assert not torch.equal(value, other)
        else:
            torch.testing.assert_close(value, other, atol=0, rtol=0)
    assert states[0]["optimizer_state_dict"]["state"] == {}
    assert states[0]["iter"] == states[1]["iter"] == 0
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        prepare_checkpoint(SOURCE, paths[0], "free")


def test_registration_exposes_policy_and_algorithm():
    import rsl_rl.runners.on_policy_runner as runner_module

    from week03_ant.prior_ppo import PriorPPO

    register_prior_components()
    assert runner_module.PriorActorCritic is PriorActorCritic
    assert runner_module.PriorPPO is PriorPPO

