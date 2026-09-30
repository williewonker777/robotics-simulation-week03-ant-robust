"""Train one preregistered v17 reset-sampling arm; no scoring or selection."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import runpy
import sys

from week03_ant.lp_study_v17 import (
    ARMS, EXPERIMENT, ROOT, TRAIN_TASK, V16_CONTROL, prepare_checkpoint,
    save_json, state_hashes, tensor_sha, training_parameters, utc,
    validate_training_start,
)


def training_arguments(forwarded, arm: str, *, stage_size: int = 1024) -> list[str]:
    """Add fixed PPO/task/reward treatment and reject substituted Hydra values."""
    if arm not in ARMS or stage_size not in (16, 1024):
        raise ValueError("unknown v17 arm or unauthorized stage size")
    forwarded = list(forwarded)
    if "--task" in forwarded:
        index = forwarded.index("--task")
        if (forwarded.count("--task") != 1 or index + 1 == len(forwarded)
                or forwarded[index + 1] != TRAIN_TASK):
            raise ValueError("v17 requires its opt-in training task")
    elif any(value.startswith("--task=") for value in forwarded):
        raise ValueError("use separate --task and task-name arguments")
    else:
        forwarded += ["--task", TRAIN_TASK]
    if any(arg.lstrip("+").startswith("env.rewards.directional_stability.") for arg in forwarded):
        raise ValueError("v17 cannot carry the v15 directional reward")
    fixed = (
        ("env.rewards.adaptive_posture.weight", 1.0),
        ("env.rewards.contact_slip.weight", 0.0),
        ("env.events.reset_base.params.mode", arm),
        ("env.events.reset_base.params.stage_size", stage_size),
        ("agent.policy.command_mode", "conditioned"),
        ("agent.policy.prior_mode", "anchored"),
        ("agent.policy.input_mode", "targets"),
        ("agent.policy.class_name", "CommandPriorActorCritic"),
        ("agent.algorithm.class_name", "PriorPPO"),
        ("agent.algorithm.teacher_coef", 0.02),
        ("env.rewards.foothold_support.weight", 0.0),
        ("agent.experiment_name", EXPERIMENT),
        ("agent.policy.require_config_match", "true"),
    )
    allowed = {key for key, _ in fixed} | {
        "env.scene.terrain.terrain_generator.seed", "agent.device",
    }
    for arg in forwarded:
        if "=" in arg and arg.lstrip("+").startswith(("env.", "agent.")):
            if arg.lstrip("+").split("=", 1)[0] not in allowed:
                raise ValueError(f"undeclared v17 Hydra override: {arg}")
    for key, value in fixed:
        supplied = [arg.split("=", 1)[1] for arg in forwarded if arg.lstrip("+").startswith(key + "=")]
        if len(supplied) > 1 or (supplied and str(supplied[0]) != str(value)):
            raise ValueError(f"v17 fixed config mismatch: {key}")
        if not supplied:
            forwarded.append(f"{key}={value}")
    return forwarded


def validate_budget(forwarded, *, development: bool) -> None:
    """Do not allow a direct launcher call to evade preregistered stage budgets."""
    def value(flag):
        if forwarded.count(flag) != 1:
            raise ValueError(f"v17 requires exactly one {flag}")
        index = forwarded.index(flag)
        if index + 1 == len(forwarded):
            raise ValueError(f"v17 missing {flag} value")
        return forwarded[index + 1]

    if any(arg.startswith(("--num_envs=", "--max_iterations=", "--seed=")) for arg in forwarded):
        raise ValueError("v17 budget flags must use separate values")
    envs, iterations, seed = (int(value(flag)) for flag in ("--num_envs", "--max_iterations", "--seed"))
    if (seed != 49 or (development and (envs != 256 or not 1 <= iterations <= 300))
            or (not development and (envs, iterations) not in ((4096, 2), (4096, 250)))):
        raise ValueError("v17 development/full training budget or seed differs")
    if (value("--load_run") != "v17_init" or value("--checkpoint") != "model_0.pt"
            or forwarded.count("--resume") != 1
            or "env.scene.terrain.terrain_generator.seed=101" not in forwarded):
        raise ValueError("v17 starting checkpoint or training geometry differs")


def initial_policy_parity(policy, observations):
    """Prove actor, critic and frozen teacher identity to the v16 control."""
    import torch
    from week03_ant.command_policy import CommandPriorActorCritic

    device, selected = observations.device, {"policy": observations}
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        source = CommandPriorActorCritic(
            selected, {"policy": ["policy"], "critic": ["policy"]}, 8,
            input_mode="targets", prior_mode="anchored", command_mode="conditioned",
            require_config_match=True,
        ).to(device)
        source.load_state_dict(torch.load(V16_CONTROL, map_location=device, weights_only=False)["model_state_dict"])
        source.eval()
        with torch.no_grad():
            for name, old, new in (
                ("actor", source.act_inference(selected), policy.act_inference(selected)),
                ("critic", source.evaluate(selected), policy.evaluate(selected)),
                ("teacher", source.teacher_mean(selected), policy.teacher_mean(selected)),
            ):
                if not torch.equal(old, new):
                    raise ValueError(f"initial same-device v16 {name} output differs")
    return {"actor": True, "critic": True, "teacher": True, "device": str(device)}


class LaneOccupancyEnv:
    """Count task exposure before each transition; forward the RSL horizon setter."""

    def __init__(self, vector_env, raw_env):
        import torch
        from week03_ant.tasks.lanes import get_lane_state

        self._vector_env, self._raw_env = vector_env, raw_env
        self._state = get_lane_state(raw_env)
        self._occupancy = torch.zeros(35, dtype=torch.long, device=raw_env.device)
        self._steps = 0

    def __getattr__(self, name):
        return getattr(self._vector_env, name)

    @property
    def episode_length_buf(self):
        return self._vector_env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self._vector_env.episode_length_buf = value

    def step(self, actions):
        import torch

        lanes = self._state.lane
        if lanes.shape != (self._raw_env.num_envs,) or not bool(((lanes >= 0) & (lanes < 35)).all()):
            raise ValueError("invalid v17 pre-transition lane IDs")
        self._occupancy += torch.bincount(lanes, minlength=35)
        self._steps += 1
        return self._vector_env.step(actions)

    def summary(self):
        counts = self._occupancy.cpu().tolist()
        return {"transition_steps": sum(counts), "policy_steps": self._steps,
                "lane_transition_counts": counts,
                "family_transition_counts": [sum(counts[i:i + 5]) for i in range(0, 35, 5)],
                "flat_transition_fraction": sum(counts[30:]) / sum(counts)}


@contextmanager
def audited_runner(arm: str, stage_size: int, audit_output: Path, sampling_output: Path,
                   expected_path: Path | None = None):
    import torch
    import rsl_rl.runners as runners
    from week03_ant.lp_curriculum_v17 import base_probabilities

    original = runners.OnPolicyRunner

    class AuditedRunner(original):
        def learn(self, *args, **kwargs):
            env = self.env.unwrapped
            with torch.no_grad():
                obs = self.env.get_observations()["policy"]
            robot = env.scene["robot"].data
            from week03_ant.tasks.contact_v16_cfg import validate_contact_config_parity
            if validate_contact_config_parity() is not True:
                raise ValueError("inherited v16 contact scene parity failed")
            sampler = getattr(env, "_week03_lp_sampler_v17", None)
            if (sampler is None or sampler.mode != arm or sampler.stage_size != stage_size
                    or sampler.summary()["sampled_episode_counts"] != [0] * 35
                    or sampler.summary()["stage_updates"] != 0
                    or not torch.equal(sampler.probabilities, base_probabilities(env.device))):
                raise ValueError("v17 initial sampler state differs")
            saved = training_parameters(self.log_dir, arm, stage_size)
            data = {
                "arm": arm, "stage_size": stage_size, "created_utc": utc(),
                "log_dir": str(Path(self.log_dir).relative_to(ROOT)),
                "normalized_parameters": saved["normalized"], "saved_parameter_sha256": saved["sha256"],
                "num_envs": env.num_envs, "dt": float(env.step_dt),
                "observation_dimensions": list(obs.shape),
                "initial_state_sha256": {"root_state": tensor_sha(robot.root_state_w),
                                         "joint_pos": tensor_sha(robot.joint_pos),
                                         "joint_vel": tensor_sha(robot.joint_vel),
                                         "observations": tensor_sha(obs)},
                "initial_prefix_sha256": tensor_sha(obs[:, :88]),
                "initial_policy_parity": initial_policy_parity(self.alg.policy, obs),
                "policy_state_sha256": state_hashes(self.alg.policy.state_dict()),
                "rng_sha256": {"cpu": tensor_sha(torch.get_rng_state()),
                                "cuda": tensor_sha(torch.cuda.get_rng_state(env.device))},
                "initial_lane_sha256": tensor_sha(env._week03_lane_state.lane),
                "initial_sampler_probabilities": sampler.probabilities.cpu().tolist(),
                "contact_slip_weight": float(env.cfg.rewards.contact_slip.weight),
                "iteration": self.current_learning_iteration,
                "optimizer_state_empty": not bool(self.alg.optimizer.state_dict()["state"]),
            }
            save_json(audit_output, data)
            if (data["contact_slip_weight"] != 0. or data["iteration"] != 0
                    or not data["optimizer_state_empty"] or obs.shape != (env.num_envs, 91)):
                raise ValueError("actual starting training configuration differs")
            validate_training_start(self.alg.policy.state_dict(), self.alg.optimizer, arm)
            if expected_path is not None:
                from week03_ant.lp_study_v17 import check_initial
                check_initial(data, json.loads(Path(expected_path).read_text()))
                if data["initial_lane_sha256"] != json.loads(Path(expected_path).read_text())["initial_lane_sha256"]:
                    raise ValueError("paired initial lane assignment differs")
            tracked = LaneOccupancyEnv(self.env, env)
            self.env = tracked
            try:
                result = super().learn(*args, **kwargs)
            finally:
                self.env = tracked._vector_env
            evidence = sampler.summary() | tracked.summary()
            evidence.update(created_utc=utc(), arm=arm, initial_audit=str(audit_output.relative_to(ROOT)))
            save_json(sampling_output, evidence)
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
    parser.add_argument("--sampling-output", type=Path)
    parser.add_argument("--expected-initial", type=Path)
    parser.add_argument("--stage-size", type=int, default=1024)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if args.operation == "prepare":
        if args.output is None or forwarded or args.arm or args.audit_output or args.sampling_output or args.expected_initial or args.development or args.stage_size != 1024:
            parser.error("prepare requires --output only")
        if not args.dry_run:
            print(json.dumps(prepare_checkpoint(args.output), indent=2))
        return
    if not args.arm or not args.audit_output or not args.sampling_output or args.output:
        parser.error("train requires --arm, --audit-output, --sampling-output")
    if args.stage_size != (16 if args.development else 1024):
        parser.error("stage-size16 is development-only; full training uses1024")
    if args.audit_output.exists() or args.sampling_output.exists():
        raise FileExistsError("training evidence output already exists")
    if args.expected_initial is not None and not args.expected_initial.is_file():
        raise FileNotFoundError(args.expected_initial)
    forwarded = training_arguments(forwarded, args.arm, stage_size=args.stage_size)
    validate_budget(forwarded, development=args.development)
    script = ROOT / "scripts/train.py"
    if args.dry_run:
        print(json.dumps({"arm": args.arm, "stage_size": args.stage_size,
                          "backend": [str(script), *forwarded]}, indent=2))
        return
    import week03_ant.tasks.lp_v17  # noqa: F401
    from week03_ant.command_policy import register_command_components

    register_command_components()
    old_argv = sys.argv
    try:
        sys.argv = [str(script), *forwarded]
        with audited_runner(args.arm, args.stage_size, args.audit_output, args.sampling_output,
                            args.expected_initial):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
