"""Additive v13 fine-tuning launcher; archived training code remains unchanged."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import json
import runpy
import sys

from week03_ant.posture_study import (
    ARMS, ROOT, TRAIN_TASK, prepare_checkpoint, save_json, state_hashes, tensor_sha, utc, validate_training_start, training_parameters,
)


def training_arguments(forwarded, arm):
    from prior_v10 import training_arguments as original_arguments

    if arm not in ARMS:
        raise ValueError("unknown reward arm")
    forwarded = list(forwarded)
    if "--task" in forwarded:
        index = forwarded.index("--task")
        if forwarded.count("--task") != 1 or index + 1 == len(forwarded) or forwarded[index + 1] != TRAIN_TASK:
            raise ValueError("v13 requires its opt-in training task")
    elif any(value.startswith("--task=") for value in forwarded):
        raise ValueError("use separate --task and task-name arguments")
    else:
        forwarded += ["--task", TRAIN_TASK]
    for key, value in (("env.rewards.adaptive_posture.weight", float(arm == "adaptive")),
                       ("agent.policy.prior_mode", "anchored"),
                       ("agent.experiment_name", "week03_ant_adaptive_posture_v13")):
        supplied = [arg.split("=", 1)[1] for arg in forwarded if arg.lstrip("+").startswith(key + "=")]
        if len(supplied) > 1 or (supplied and (float(supplied[0]) if isinstance(value, float) else supplied[0]) != value):
            raise ValueError(f"v13 arm/config mismatch: {key}")
        if not supplied:
            forwarded.append(f"{key}={value}")
    return original_arguments(forwarded)


def check_initial(data, expected):
    for key in ("initial_state_sha256", "policy_state_sha256", "rng_sha256", "num_envs", "dt", "normalized_parameters"):
        if data[key] != expected[key]:
            raise ValueError(f"paired training initialization differs: {key}")


@contextmanager
def audited_runner(arm, output, expected_path=None):
    import torch
    import rsl_rl.runners as runners

    original = runners.OnPolicyRunner

    class AuditedRunner(original):
        def learn(self, *args, **kwargs):
            env = self.env.unwrapped
            # One extra observation query in BOTH arms, before the unchanged learn
            # loop. Training observation noise/RNG state is paired and recorded.
            with torch.no_grad():
                obs = self.env.get_observations()["policy"]
            robot = env.scene["robot"].data
            from week03_ant.tasks.foothold_v9_cfg import FootholdTrainAntEnvCfg
            from week03_ant.tasks.posture_v13_cfg import AdaptivePostureTrainEnvCfg
            original_cfg = FootholdTrainAntEnvCfg().to_dict()
            additive_cfg = AdaptivePostureTrainEnvCfg().to_dict()
            additive_cfg["rewards"].pop("adaptive_posture")
            if additive_cfg != original_cfg:
                raise ValueError("new task changes original configuration beyond additive reward")
            saved = training_parameters(self.log_dir, arm)
            data = {"arm": arm, "default_config_parity": True,
                    "normalized_parameters": saved["normalized"], "saved_parameter_sha256": saved["sha256"], "created_utc": utc(), "log_dir": str(Path(self.log_dir).relative_to(ROOT)),
                    "num_envs": env.num_envs, "dt": float(env.step_dt),
                    "initial_state_sha256": {"root_state": tensor_sha(robot.root_state_w),
                                             "joint_pos": tensor_sha(robot.joint_pos),
                                             "joint_vel": tensor_sha(robot.joint_vel),
                                             "observations": tensor_sha(obs)},
                    "policy_state_sha256": state_hashes(self.alg.policy.state_dict()),
                    "rng_sha256": {"cpu": tensor_sha(torch.get_rng_state()),
                                   "cuda": tensor_sha(torch.cuda.get_rng_state(env.device))},
                    "reward_weight": float(env.cfg.rewards.adaptive_posture.weight),
                    "observation_dimensions": list(obs.shape), "iteration": self.current_learning_iteration,
                    "optimizer_state_empty": not bool(self.alg.optimizer.state_dict()["state"])}
            save_json(output, data)
            if (data["reward_weight"] != float(arm == "adaptive") or data["iteration"] != 0
                    or not data["optimizer_state_empty"] or obs.shape != (env.num_envs, 88)):
                raise ValueError("actual starting training config differs from declared arm")
            validate_training_start(self.alg.policy.state_dict(), self.alg.optimizer)
            if expected_path is not None:
                check_initial(data, json.loads(Path(expected_path).read_text()))
            return super().learn(*args, **kwargs)

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
            parser.error("prepare accepts only --output and optional --dry-run")
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
    import week03_ant.tasks.posture_v13  # noqa: F401
    from week03_ant.prior_policy import register_prior_components

    register_prior_components()
    old_argv = sys.argv
    try:
        sys.argv = [str(script), *forwarded]
        with audited_runner(args.arm, args.audit_output, args.expected_initial):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
