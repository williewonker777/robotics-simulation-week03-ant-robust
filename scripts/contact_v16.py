"""Opt-in v16 paired continuation launcher with contact-sensor preflight evidence."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import runpy
import sys

from week03_ant.contact_study import (
    ARMS, ROOT, TRAIN_TASK, V15, check_initial, prepare_checkpoint, save_json, state_hashes,
    tensor_sha, training_parameters, utc, validate_training_start,
)


def training_arguments(forwarded, arm: str) -> list[str]:
    """Append the fixed v16 contract; reject all treatment/config substitutions."""
    if arm not in ARMS:
        raise ValueError("unknown contact arm")
    forwarded = list(forwarded)
    if "--task" in forwarded:
        index = forwarded.index("--task")
        if forwarded.count("--task") != 1 or index + 1 == len(forwarded) or forwarded[index + 1] != TRAIN_TASK:
            raise ValueError("v16 requires its opt-in training task")
    elif any(value.startswith("--task=") for value in forwarded):
        raise ValueError("use separate --task and task-name arguments")
    else:
        forwarded += ["--task", TRAIN_TASK]
    if any(arg.lstrip("+").startswith("env.rewards.directional_stability.") for arg in forwarded):
        raise ValueError("v16 must not carry the v15 directional reward")
    fixed = (
        ("env.rewards.adaptive_posture.weight", 1.0),
        ("env.rewards.contact_slip.weight", float(arm == "slip")),
        ("agent.policy.command_mode", "conditioned"), ("agent.policy.prior_mode", "anchored"),
        ("agent.policy.input_mode", "targets"), ("agent.policy.class_name", "CommandPriorActorCritic"),
        ("agent.algorithm.class_name", "PriorPPO"), ("agent.algorithm.teacher_coef", 0.02),
        ("env.rewards.foothold_support.weight", 0.0), ("agent.experiment_name", "week03_ant_contact_v16"),
        ("agent.policy.require_config_match", "true"),
    )
    for key, value in fixed:
        supplied = [arg.split("=", 1)[1] for arg in forwarded if arg.lstrip("+").startswith(key + "=")]
        actual = lambda raw: float(raw) if isinstance(value, float) else raw
        if len(supplied) > 1 or (supplied and actual(supplied[0]) != value):
            raise ValueError(f"v16 arm/config mismatch: {key}")
        if not supplied:
            forwarded.append(f"{key}={value}")
    return forwarded


def initial_policy_parity(policy, observations):
    """Prove same-device actor/critic/teacher identity before either arm learns."""
    import torch
    from week03_ant.command_policy import CommandPriorActorCritic

    device, selected = observations.device, {"policy": observations}
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        source = CommandPriorActorCritic(
            selected, {"policy": ["policy"], "critic": ["policy"]}, 8,
            input_mode="targets", prior_mode="anchored", command_mode="conditioned", require_config_match=True,
        ).to(device)
        source.load_state_dict(torch.load(V15, map_location=device, weights_only=False)["model_state_dict"])
        source.eval()
        with torch.no_grad():
            for name, old, new in (
                ("actor", source.act_inference(selected), policy.act_inference(selected)),
                ("critic", source.evaluate(selected), policy.evaluate(selected)),
                ("teacher", source.teacher_mean(selected), policy.teacher_mean(selected)),
            ):
                if not torch.equal(old, new):
                    raise ValueError(f"initial same-device v15 91D {name} output differs")
    return {"actor": True, "critic": True, "teacher": True, "device": str(device)}


def contact_geometry(env, term=None, row_mask=None):
    """Exercise passive contact diagnostics, optionally excluding reset/wrap rows."""
    from week03_ant.tasks.contact_v16_cfg import ContactSlipReward
    from isaaclab.managers import RewardTermCfg
    import torch

    term = term or ContactSlipReward(RewardTermCfg(func=ContactSlipReward, weight=0.), env)
    values = term.geometry(env)
    required = {"reward", "bounded_cost", "valid", "contact", "contact_fraction", "no_contact", "contacted_tip_speed", "force_norm", "tip_speed"}
    if not isinstance(values, dict) or not required <= values.keys():
        raise ValueError("contact reward diagnostics schema differs")
    for name in ("reward", "bounded_cost", "contact_fraction", "no_contact", "contacted_tip_speed"):
        value = values[name]
        if not isinstance(value, torch.Tensor) or value.shape != (env.num_envs,) or not torch.isfinite(value).all():
            raise ValueError(f"invalid contact reward diagnostic: {name}")
    if (values["valid"].shape != (env.num_envs,) or values["contact"].shape != (env.num_envs, 4)
            or values["force_norm"].shape != (env.num_envs, 4) or values["tip_speed"].shape != (env.num_envs, 4)):
        raise ValueError("contact reward diagnostic shape differs")
    if bool((values["reward"] > 0).any()) or bool((values["reward"] < -1).any()):
        raise ValueError("contact reward escaped declared [-1, 0] bound")

    forces = []
    for sensor in term.sensors:
        force = sensor.data.force_matrix_w
        if force is None or force.shape != (env.num_envs, 1, 2, 3):
            raise ValueError("contact sensor force matrix shape differs")
        forces.append(force[:, 0])
    forces = torch.stack(forces, dim=1)  # [N, four feet, mesh/plane, xyz]
    finite = torch.isfinite(forces).all(dim=(-1, -2, -3))
    if values["valid"].dtype is not torch.bool:
        raise ValueError("contact reward validity mask dtype differs")
    finite &= values["valid"]
    if row_mask is None:
        sampled = torch.ones(env.num_envs, dtype=torch.bool, device=forces.device)
    else:
        sampled = torch.as_tensor(row_mask, dtype=torch.bool, device=forces.device)
        if sampled.shape != (env.num_envs,):
            raise ValueError("contact sample row mask shape differs")
    sampled_rows = int(sampled.sum())
    valid_rows = sampled & finite
    valid_count = int(valid_rows.sum())
    invalid_count = sampled_rows - valid_count
    invalid_fraction = invalid_count / sampled_rows if sampled_rows else 0.
    if sampled_rows and invalid_fraction > .01:
        raise ValueError("contact diagnostic invalid fraction exceeds one percent")
    target_contact = torch.linalg.vector_norm(forces, dim=-1) > 2.
    target_contact &= torch.isfinite(forces).all(dim=-1)

    # Invalid rows are a separately recorded failure condition, not zero-contact
    # samples.  This also excludes returned auto-reset and lane-wrap rows, whose
    # observation/contact state is not a co-timed solved physics state.
    def mean(value):
        if valid_count == 0:
            return 0.
        return float(value[valid_rows].float().mean())

    return {
        "sampled_rows": sampled_rows, "valid_rows": valid_count, "invalid_rows": invalid_count,
        "reward": mean(values["reward"]), "bounded_cost": mean(values["bounded_cost"]),
        "contact_fraction": mean(values["contact_fraction"]),
        "no_contact_fraction": mean(values["no_contact"]),
        "contacted_tip_speed": mean(values["contacted_tip_speed"]), "invalid_fraction": invalid_fraction,
        "terrain_contact_fraction": mean(target_contact[:, :, 0]),
        "flat_contact_fraction": mean(target_contact[:, :, 1]),
        "per_foot_contact_fraction": [mean(v) for v in target_contact.any(dim=-1).transpose(0, 1)],
        "per_target_per_foot_contact_fraction": [
            [mean(target_contact[:, foot, target]) for target in range(2)] for foot in range(4)
        ],
    }


class ContactSamplingEnv:
    """Transparent vector-env proxy which samples post-physics contact metrics."""

    _SCALAR_METRICS = (
        "reward", "bounded_cost", "contact_fraction", "no_contact_fraction", "contacted_tip_speed",
        "terrain_contact_fraction", "flat_contact_fraction",
    )
    _LIST_METRICS = ("per_foot_contact_fraction", "per_target_per_foot_contact_fraction")

    def __init__(self, vector_env, raw_env, wrap_count_getter=None):
        self._vector_env, self._raw_env = vector_env, raw_env
        self._term = None
        self._steps = self._sampled_rows = self._valid_rows = self._invalid_rows = 0
        self._done_rows = self._wrap_rows = 0
        self._sums = {}
        if wrap_count_getter is None:
            # This is intentionally not optional in a live Isaac run: silently
            # disabling the portal mask would reintroduce stale-contact samples.
            from week03_ant.tasks.lanes import get_lane_state
            wrap_count_getter = lambda: get_lane_state(raw_env).wrap_count
        self._wrap_count_getter = wrap_count_getter
        self._last_wrap_count = self._wrap_count_getter().clone()

    def __getattr__(self, name):
        return getattr(self._vector_env, name)

    def _wrapped_rows(self):
        import torch
        current = self._wrap_count_getter()
        wrapped = current != self._last_wrap_count
        self._last_wrap_count = current.clone()
        return wrapped

    def step(self, actions):
        import torch

        result = self._vector_env.step(actions)
        if self._term is None:
            from isaaclab.managers import RewardTermCfg
            from week03_ant.tasks.contact_v16_cfg import ContactSlipReward
            self._term = ContactSlipReward(RewardTermCfg(func=ContactSlipReward, weight=0.), self._raw_env)
        done = torch.as_tensor(result[2], dtype=torch.bool, device=self._raw_env.device).reshape(-1)
        if done.shape != (self._raw_env.num_envs,):
            raise ValueError("vector environment done mask shape differs")
        wrapped = self._wrapped_rows()
        metrics = contact_geometry(self._raw_env, self._term, ~(done | wrapped))
        self._steps += 1
        self._sampled_rows += metrics["sampled_rows"]
        self._valid_rows += metrics["valid_rows"]
        self._invalid_rows += metrics["invalid_rows"]
        self._done_rows += int(done.sum())
        self._wrap_rows += int(wrapped.sum())
        if metrics["valid_rows"]:
            weight = metrics["valid_rows"]
            for name in self._SCALAR_METRICS:
                self._sums[name] = self._sums.get(name, 0.) + metrics[name] * weight
            for name in self._LIST_METRICS:
                tensor = torch.tensor(metrics[name], dtype=torch.float64)
                self._sums[name] = self._sums.get(name, torch.zeros_like(tensor)) + tensor * weight
        return result

    def summary(self):
        if self._steps == 0 or self._sampled_rows == 0:
            raise ValueError("contact sampler saw no post-action non-reset rows")
        if self._valid_rows == 0:
            raise ValueError("contact sampler saw no valid post-action rows")
        output = {
            "samples": self._steps, "sampled_rows": self._sampled_rows, "valid_rows": self._valid_rows,
            "invalid_rows": self._invalid_rows, "invalid_fraction": self._invalid_rows / self._sampled_rows,
            "excluded_done_rows": self._done_rows, "excluded_wrap_rows": self._wrap_rows,
        }
        for name, value in self._sums.items():
            output[name] = (value / self._valid_rows).tolist() if hasattr(value, "tolist") else value / self._valid_rows
        return output


@contextmanager
def audited_runner(arm: str, output: Path, expected_path: Path | None = None):
    import torch
    import rsl_rl.runners as runners

    original = runners.OnPolicyRunner

    class AuditedRunner(original):
        def learn(self, *args, **kwargs):
            env = self.env.unwrapped
            with torch.no_grad():
                obs = self.env.get_observations()["policy"]
            robot = env.scene["robot"].data
            from week03_ant.tasks.contact_v16_cfg import validate_contact_config_parity
            if validate_contact_config_parity() is not True:
                raise ValueError("v16 config parity check did not pass")
            contact = contact_geometry(env)  # Required even when RewardManager skips weight-zero terms.
            parity = initial_policy_parity(self.alg.policy, obs)
            saved = training_parameters(self.log_dir, arm)
            data = {
                "arm": arm, "created_utc": utc(), "log_dir": str(Path(self.log_dir).relative_to(ROOT)),
                "normalized_parameters": saved["normalized"], "saved_parameter_sha256": saved["sha256"],
                "num_envs": env.num_envs, "dt": float(env.step_dt), "observation_dimensions": list(obs.shape),
                "initial_state_sha256": {"root_state": tensor_sha(robot.root_state_w),
                                         "joint_pos": tensor_sha(robot.joint_pos),
                                         "joint_vel": tensor_sha(robot.joint_vel), "observations": tensor_sha(obs)},
                "initial_prefix_sha256": tensor_sha(obs[:, :88]), "initial_policy_parity": parity,
                "policy_state_sha256": state_hashes(self.alg.policy.state_dict()),
                "rng_sha256": {"cpu": tensor_sha(torch.get_rng_state()),
                                "cuda": tensor_sha(torch.cuda.get_rng_state(env.device))},
                "contact_preflight": contact, "contact_slip_weight": float(env.cfg.rewards.contact_slip.weight),
                "iteration": self.current_learning_iteration,
                "optimizer_state_empty": not bool(self.alg.optimizer.state_dict()["state"]),
            }
            save_json(output, data)
            if (data["contact_slip_weight"] != float(arm == "slip") or data["iteration"] != 0
                    or not data["optimizer_state_empty"] or obs.shape != (env.num_envs, 91)):
                raise ValueError("actual starting training config differs from declared arm")
            validate_training_start(self.alg.policy.state_dict(), self.alg.optimizer, arm)
            if expected_path is not None:
                check_initial(data, json.loads(Path(expected_path).read_text()))
            sampled_env, self.env = ContactSamplingEnv(self.env, env), None
            self.env = sampled_env
            try:
                result = super().learn(*args, **kwargs)
            finally:
                self.env = sampled_env._vector_env
            save_json(output.with_name(output.stem + "_contact_rollout.json"), {
                "arm": arm, "created_utc": utc(), "source_initial_audit": str(output.relative_to(ROOT)),
                "rollout": sampled_env.summary(), "passive_post_action": True,
            })
            return result

    runners.OnPolicyRunner = AuditedRunner
    try:
        yield
    finally:
        runners.OnPolicyRunner = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "train"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--expected-initial", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if args.operation == "prepare":
        if args.output is None or forwarded or args.arm or args.audit_output or args.expected_initial:
            parser.error("prepare requires --output only")
        if not args.dry_run:
            print(json.dumps(prepare_checkpoint(args.output), indent=2))
        return
    if not args.arm or args.audit_output is None or args.output:
        parser.error("train requires --arm and --audit-output")
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    if args.expected_initial and not args.expected_initial.is_file():
        raise FileNotFoundError(args.expected_initial)
    forwarded = training_arguments(forwarded, args.arm)
    script = ROOT / "scripts/train.py"
    if args.dry_run:
        print(json.dumps({"arm": args.arm, "backend": [str(script), *forwarded]}, indent=2))
        return
    import week03_ant.tasks.contact_v16  # noqa: F401
    from week03_ant.command_policy import register_command_components

    register_command_components()
    old_argv = sys.argv
    try:
        sys.argv = [str(script), *forwarded]
        with audited_runner(args.arm, args.audit_output, args.expected_initial):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
