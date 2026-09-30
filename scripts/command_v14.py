"""Opt-in v14 command-access fine tuning; frozen training code remains unchanged."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import json
import runpy
import sys

from week03_ant.command_study import (
    ARMS, ROOT, TRAIN_TASK, prepare_checkpoint, save_json, state_hashes, tensor_sha, utc, validate_training_start, training_parameters, check_initial, EXPERIMENT,
)


def training_arguments(forwarded, arm):
    if arm not in ARMS:
        raise ValueError("unknown command arm")
    forwarded = list(forwarded)
    if "--task" in forwarded:
        index = forwarded.index("--task")
        if forwarded.count("--task") != 1 or index + 1 == len(forwarded) or forwarded[index + 1] != TRAIN_TASK:
            raise ValueError("v14 requires its opt-in training task")
    elif any(value.startswith("--task=") for value in forwarded):
        raise ValueError("use separate --task and task-name arguments")
    else:
        forwarded += ["--task", TRAIN_TASK]
    for key, value in (
        ("env.rewards.adaptive_posture.weight", 1.), ("agent.policy.command_mode", arm),
        ("agent.policy.prior_mode", "anchored"), ("agent.policy.input_mode", "targets"),
        ("agent.policy.class_name", "CommandPriorActorCritic"), ("agent.algorithm.class_name", "PriorPPO"),
        ("agent.algorithm.teacher_coef", .02), ("env.rewards.foothold_support.weight", 0.),
        ("agent.experiment_name", EXPERIMENT), ("agent.policy.require_config_match", "true"),
    ):
        supplied = [arg.split("=", 1)[1] for arg in forwarded if arg.lstrip("+").startswith(key + "=")]
        if len(supplied) > 1 or (supplied and (float(supplied[0]) if isinstance(value, float) else supplied[0]) != value):
            raise ValueError(f"v14 arm/config mismatch: {key}")
        if not supplied:
            forwarded.append(f"{key}={value}")
    return forwarded


def initial_policy_parity(policy, observations):
    """Same-device bit parity before training, without altering RNG or observations."""
    import torch
    from week03_ant.prior_policy import PriorActorCritic
    from week03_ant.command_study import START

    device = observations.device
    prefix = {"policy": observations[:, :88].contiguous()}
    with torch.random.fork_rng(devices=[device]):
        original = PriorActorCritic(prefix, {"policy": ["policy"], "critic": ["policy"]}, 8,
            input_mode="targets", prior_mode="anchored", require_config_match=True).to(device)
        original.load_state_dict(torch.load(START, map_location=device, weights_only=False)["model_state_dict"])
        original.eval()
        with torch.no_grad():
            for label, old, new in (
                ("actor", original.act_inference(prefix), policy.act_inference({"policy": observations})),
                ("critic", original.evaluate(prefix), policy.evaluate({"policy": observations})),
                ("teacher", original.teacher_mean(prefix), policy.teacher_mean({"policy": observations})),
            ):
                if not torch.equal(old, new):
                    raise ValueError(f"initial same-device original88D {label} output differs")
    return {"actor": True, "critic": True, "teacher": True, "device": str(device)}


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
            from week03_ant.tasks.posture_v13_cfg import AdaptivePostureTrainEnvCfg
            from week03_ant.tasks.command_v14_cfg import CommandTrainAntEnvCfg
            original_cfg = AdaptivePostureTrainEnvCfg().to_dict()
            additive_cfg = CommandTrainAntEnvCfg().to_dict()
            additive_cfg["observations"]["policy"].pop("posture_command")
            if additive_cfg != original_cfg:
                raise ValueError("new task changes original configuration beyond appended observation")
            parity = initial_policy_parity(self.alg.policy, obs)
            saved = training_parameters(self.log_dir, arm)
            data = {"arm": arm, "default_config_parity": True,
                    "normalized_parameters": saved["normalized"], "saved_parameter_sha256": saved["sha256"], "created_utc": utc(), "log_dir": str(Path(self.log_dir).relative_to(ROOT)),
                    "num_envs": env.num_envs, "dt": float(env.step_dt),
                    "initial_state_sha256": {"root_state": tensor_sha(robot.root_state_w),
                                             "joint_pos": tensor_sha(robot.joint_pos),
                                             "joint_vel": tensor_sha(robot.joint_vel),
                                             "observations": tensor_sha(obs)},
                    "initial_prefix_sha256": tensor_sha(obs[:, :88]),
                    "initial_policy_parity": parity,
                    "policy_state_sha256": state_hashes({k: v for k, v in self.alg.policy.state_dict().items() if k != "command_mode_code"}),
                    "rng_sha256": {"cpu": tensor_sha(torch.get_rng_state()),
                                   "cuda": tensor_sha(torch.cuda.get_rng_state(env.device))},
                    "reward_weight": float(env.cfg.rewards.adaptive_posture.weight),
                    "observation_dimensions": list(obs.shape), "iteration": self.current_learning_iteration,
                    "optimizer_state_empty": not bool(self.alg.optimizer.state_dict()["state"])}
            save_json(output, data)
            if (data["reward_weight"] != 1. or data["iteration"] != 0
                    or not data["optimizer_state_empty"] or obs.shape != (env.num_envs, 91)):
                raise ValueError("actual starting training config differs from declared arm")
            validate_training_start(self.alg.policy.state_dict(), self.alg.optimizer, arm)
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
        if args.output is None or forwarded or not args.arm or args.audit_output or args.expected_initial:
            parser.error("prepare requires --output and --arm; optional --dry-run")
        if not args.dry_run:
            print(json.dumps(prepare_checkpoint(args.output, args.arm), indent=2))
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
    import week03_ant.tasks.command_v14  # noqa: F401
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
