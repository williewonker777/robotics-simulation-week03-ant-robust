import hashlib
from pathlib import Path

import pytest
import torch

from rsl_rl.modules import ActorCritic

from scripts.prepare_foothold_checkpoint import prepare_checkpoint
from week03_ant.foothold_math import OBSERVATIONS
from week03_ant.foothold_policy import (
    FootholdActorCritic, INPUT_MODES, register_foothold_policy, warmstart_from_60d,
)


def observations(batch=5):
    return {"policy": torch.randn(batch, OBSERVATIONS)}


def policy(mode="guided", **kwargs):
    obs = observations()
    return FootholdActorCritic(
        obs, {"policy": ["policy"], "critic": ["policy"]}, 8,
        input_mode=mode, init_noise_std=0.2, **kwargs,
    )


@pytest.mark.parametrize("mode", INPUT_MODES)
def test_valid_88d_policy_has_eight_actions_and_finite_gradients(mode):
    p = policy(mode)
    obs = observations()
    assert p.act(obs).shape == (5, 8)
    loss = p.action_mean.square().mean() + p.evaluate(obs).square().mean() - .01 * p.entropy.mean()
    loss.backward()
    assert all(torch.isfinite(value.grad).all() for value in p.parameters() if value.grad is not None)


def test_mode_persists_eval_restores_and_training_guard_rejects_mismatch():
    saved = policy("targets").state_dict()
    evaluator = policy("feet")
    assert evaluator.load_state_dict(saved) is True
    assert evaluator.input_mode == "targets"
    trainer = policy("feet", require_config_match=True)
    with pytest.raises(ValueError, match="training config"):
        trainer.load_state_dict(saved)


@pytest.mark.parametrize("bad", [None, torch.tensor([0], dtype=torch.int64),
                                  torch.tensor(0, dtype=torch.int32), torch.tensor(99, dtype=torch.int64)])
def test_malformed_or_missing_mode_and_nonstrict_load_rejected(bad):
    p = policy()
    state = p.state_dict()
    if bad is None:
        del state["input_mode_code"]
    else:
        state["input_mode_code"] = bad
    with pytest.raises(ValueError):
        p.load_state_dict(state)
    with pytest.raises(ValueError):
        p.load_state_dict(p.state_dict(), strict=False)


def test_feet_mode_masks_targets_including_nan_for_actor_and_critic():
    p = policy("feet")
    a = observations()
    b = {"policy": a["policy"].clone()}
    b["policy"][:, -16:] = float("nan")
    torch.testing.assert_close(p.act_inference(a), p.act_inference(b), atol=0, rtol=0)
    torch.testing.assert_close(p.evaluate(a), p.evaluate(b), atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["targets", "guided"])
def test_target_modes_backpropagate_into_added_first_layer_columns(mode):
    p = policy(mode)
    obs = observations(11)
    (p.act_inference(obs).square().mean() + p.evaluate(obs).square().mean()).backward()
    assert p.actor[0].weight.grad[:, 60:].abs().sum() > 0
    assert p.critic[0].weight.grad[:, 60:].abs().sum() > 0


def test_paired_prepared_states_identical_except_mode_buffer(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
    a_path, b_path = tmp_path / "feet.pt", tmp_path / "targets.pt"
    prepare_checkpoint(source, a_path, "feet")
    prepare_checkpoint(source, b_path, "targets")
    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)
    assert a["iter"] == b["iter"] == 0
    assert a["optimizer_state_dict"]["state"] == b["optimizer_state_dict"]["state"] == {}
    for key, value in a["model_state_dict"].items():
        if key == "input_mode_code":
            assert int(value) != int(b["model_state_dict"][key])
        else:
            torch.testing.assert_close(value, b["model_state_dict"][key], atol=0, rtol=0)


def test_real_v5_warmstart_preserves_outputs_and_source(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source_path = root / "artifacts/terrain_demo/runs/rough_v5_portal_rehearsal4_seed43/model_round_4.pt"
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    obs = observations(13)
    old = ActorCritic(
        {"policy": obs["policy"][:, :60]}, {"policy": ["policy"], "critic": ["policy"]}, 8,
        actor_hidden_dims=[400, 200, 100], critic_hidden_dims=[400, 200, 100], activation="elu",
    )
    old.load_state_dict(source["model_state_dict"])
    new = policy("guided")
    warmstart_from_60d(new, source["model_state_dict"])
    base_obs = {"policy": obs["policy"][:, :60]}
    torch.testing.assert_close(new.act_inference(obs), old.act_inference(base_obs), atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(new.evaluate(obs), old.evaluate(base_obs), atol=1e-5, rtol=1e-6)
    assert new.actor[0].weight[:, 60:].eq(0).all()
    assert new.critic[0].weight[:, 60:].eq(0).all()
    destination = tmp_path / "prepared.pt"
    prepare_checkpoint(source_path, destination, "guided")
    prepared = torch.load(destination, map_location="cpu", weights_only=False)
    assert prepared["infos"]["source_sha256"] == before
    assert prepared["infos"]["observation_dimensions"] == 88
    assert prepared["infos"]["optimizer"] == {"name": "Adam", "learning_rate": 1e-4}
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        prepare_checkpoint(source_path, destination, "guided")


def test_dimensions_actions_and_normalization_contracts_rejected():
    groups = {"policy": ["policy"], "critic": ["policy"]}
    with pytest.raises(ValueError):
        FootholdActorCritic({"policy": torch.zeros(2, 60)}, groups, 8)
    with pytest.raises(ValueError):
        FootholdActorCritic({"policy": torch.zeros(2, 88)}, groups, 7)
    with pytest.raises(ValueError, match="normalization"):
        FootholdActorCritic({"policy": torch.zeros(2, 88)}, groups, 8,
                            actor_obs_normalization=True)


def test_registration_exposes_policy_to_rsl_runner():
    import rsl_rl.runners.on_policy_runner as runner_module

    register_foothold_policy()
    assert runner_module.FootholdActorCritic is FootholdActorCritic
