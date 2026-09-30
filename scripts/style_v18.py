"""Prepare or train one guarded v18 arm; no checkpoint/holdout selection."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import runpy
import sys

from week03_ant.style_study_v18 import (
    ARMS, EXPERIMENT, ROOT, TRAIN_GEOMETRY, TRAIN_SEED, TRAIN_TASK, V16_CONTROL,
    normalize_training_parameters, prepare_checkpoint, save_json, state_hashes,
    tensor_sha, utc, validate_training_start,
)


def training_arguments(forwarded, arm):
    if arm not in ARMS:
        raise ValueError("unknown v18 arm")
    forwarded = list(forwarded)
    if "--task" in forwarded:
        i = forwarded.index("--task")
        if forwarded.count("--task") != 1 or i + 1 == len(forwarded) or forwarded[i + 1] != TRAIN_TASK:
            raise ValueError("v18 requires its opt-in training task")
    elif any(value.startswith("--task=") for value in forwarded):
        raise ValueError("use separate --task and task-name arguments")
    else:
        forwarded += ["--task", TRAIN_TASK]
    fixed = (
        ("env.rewards.adaptive_posture.weight", 1.0),
        ("env.rewards.contact_slip.weight", 0.0),
        ("env.rewards.foothold_support.weight", 0.0),
        ("agent.policy.command_mode", "conditioned"),
        ("agent.policy.prior_mode", "anchored"),
        ("agent.policy.input_mode", "targets"),
        ("agent.policy.class_name", "StyleCommandPriorActorCritic"),
        ("agent.policy.style_mode", arm),
        ("agent.policy.require_config_match", "true"),
        ("agent.algorithm.class_name", "StylePriorPPO"),
        ("agent.algorithm.teacher_coef", .02),
        ("agent.experiment_name", EXPERIMENT),
    )
    allowed = {key for key, _ in fixed} | {"env.scene.terrain.terrain_generator.seed", "agent.device"}
    for arg in forwarded:
        if "=" in arg and arg.lstrip("+").startswith(("env.", "agent.")):
            if arg.lstrip("+").split("=", 1)[0] not in allowed:
                raise ValueError(f"undeclared v18 Hydra override: {arg}")
    for key, expected in fixed:
        supplied = [arg.split("=", 1)[1] for arg in forwarded if arg.lstrip("+").startswith(key + "=")]
        if len(supplied) > 1 or (supplied and str(supplied[0]) != str(expected)):
            raise ValueError(f"v18 fixed configuration mismatch: {key}")
        if not supplied:
            forwarded.append(f"{key}={expected}")
    return forwarded


def validate_budget(forwarded):
    def value(flag):
        if forwarded.count(flag) != 1:
            raise ValueError(f"v18 requires exactly one {flag}")
        i = forwarded.index(flag)
        if i + 1 == len(forwarded):
            raise ValueError(f"v18 missing {flag} value")
        return forwarded[i + 1]

    if any(arg.startswith(("--num_envs=", "--max_iterations=", "--seed=")) for arg in forwarded):
        raise ValueError("v18 budget flags require separate values")
    envs, iterations, seed = (int(value(flag)) for flag in ("--num_envs", "--max_iterations", "--seed"))
    if (envs, iterations) not in ((256, 16), (4096, 2), (4096, 250)) or seed != TRAIN_SEED:
        raise ValueError("v18 training budget/seed differs")
    if (value("--load_run") != "v18_init" or value("--checkpoint") != "model_0.pt"
            or forwarded.count("--resume") != 1
            or f"env.scene.terrain.terrain_generator.seed={TRAIN_GEOMETRY}" not in forwarded):
        raise ValueError("v18 shared checkpoint or training terrain differs")


class StyleExposureEnv:
    """Count pre-transition lane/mask exposure without changing observations."""

    def __init__(self, vector_env, raw_env):
        import torch

        self._vector_env, self._raw_env = vector_env, raw_env
        self._last_obs = vector_env.get_observations()
        self._steps = 0
        self._count = torch.zeros(35, dtype=torch.long, device=raw_env.device)
        self._teacher = torch.zeros_like(self._count)

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
        from week03_ant.tasks.lanes import get_lane_state

        lanes = get_lane_state(self._raw_env).lane
        mask = self._last_obs["style_gate"]
        if (lanes.shape != (self._raw_env.num_envs,) or mask.shape != (self._raw_env.num_envs, 1)
                or not bool(((lanes >= 0) & (lanes < 35)).all())
                or not bool(((mask == 0.) | (mask == 1.)).all())):
            raise ValueError("invalid v18 lane/style observation pairing")
        self._count += torch.bincount(lanes, minlength=35)
        self._teacher += torch.bincount(lanes, weights=mask[:, 0], minlength=35).long()
        self._steps += 1
        result = self._vector_env.step(actions)
        self._last_obs = result[0]
        return result

    def summary(self):
        count, teacher = self._count.cpu().tolist(), self._teacher.cpu().tolist()
        return {"policy_steps": self._steps, "transition_steps": sum(count),
                "lane_transition_counts": count, "lane_teacher_counts": teacher,
                "family_transition_counts": [sum(count[i:i + 5]) for i in range(0, 35, 5)],
                "family_teacher_counts": [sum(teacher[i:i + 5]) for i in range(0, 35, 5)]}


@contextmanager
def audited_runner(arm, audit_output, exposure_output, expected_initial=None):
    import torch
    import rsl_rl.runners as runners
    from week03_ant.command_policy import CommandPriorActorCritic

    original = runners.OnPolicyRunner

    class AuditedRunner(original):
        def learn(self, *args, **kwargs):
            raw = self.env.unwrapped
            obs = self.env.get_observations()
            policy = self.alg.policy
            robot = raw.scene["robot"].data
            if (obs["policy"].shape != (raw.num_envs, 91)
                    or obs["style_gate"].shape != (raw.num_envs, 1)
                    or not bool(((obs["style_gate"] == 0.) | (obs["style_gate"] == 1.)).all())):
                raise ValueError("v18 train-only observation contract differs")
            if policy.style_mode != arm or policy._style_loss_active:
                raise ValueError("v18 policy treatment not paired")
            validate_training_start(policy.state_dict(), self.alg.optimizer)
            saved = normalize_training_parameters(self.log_dir, arm)
            if float(raw.cfg.rewards.contact_slip.weight) != 0.:
                raise ValueError("v18 contact reward is not zero")
            device = obs["policy"].device
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                source = CommandPriorActorCritic(
                    {"policy": obs["policy"]}, {"policy": ["policy"], "critic": ["policy"]}, 8,
                    input_mode="targets", prior_mode="anchored", command_mode="conditioned",
                    require_config_match=True,
                ).to(raw.device)
                source.load_state_dict(torch.load(V16_CONTROL, map_location=raw.device,
                                                  weights_only=False)["model_state_dict"])
                source.eval()
                with torch.no_grad():
                    selected = {"policy": obs["policy"]}
                    parity = {
                        "actor": torch.equal(source.act_inference(selected), policy.act_inference(obs)),
                        "critic": torch.equal(source.evaluate(selected), policy.evaluate(obs)),
                        "teacher": torch.equal(source.teacher_mean(selected), policy.teacher_mean(obs)),
                    }
            # Source std differs by design; no inference output depends on it.
            if not all(parity.values()):
                raise ValueError(f"v18 initial source parity failed: {parity}")
            data = {
                "arm": arm, "created_utc": utc(), "log_dir": str(Path(self.log_dir).relative_to(ROOT)),
                "normalized_parameters": saved["normalized"], "saved_parameter_sha256": saved["sha256"],
                "num_envs": raw.num_envs, "dt": float(raw.step_dt),
                "initial_state_sha256": {"root_state": tensor_sha(robot.root_state_w),
                                         "joint_pos": tensor_sha(robot.joint_pos),
                                         "joint_vel": tensor_sha(robot.joint_vel),
                                         "observations": tensor_sha(obs["policy"]),
                                         "style_gate": tensor_sha(obs["style_gate"])},
                "initial_prefix_sha256": tensor_sha(obs["policy"][:, :88]),
                "initial_policy_parity": parity, "policy_state_sha256": state_hashes(policy.state_dict()),
                "rng_sha256": {"cpu": tensor_sha(torch.get_rng_state()),
                               "cuda": tensor_sha(torch.cuda.get_rng_state(raw.device))},
                "initial_lane_sha256": tensor_sha(raw._week03_lane_state.lane),
                "optimizer_state_empty": not bool(self.alg.optimizer.state_dict()["state"]),
                "iteration": self.current_learning_iteration,
            }
            if data["iteration"] != 0 or not data["optimizer_state_empty"]:
                raise ValueError("v18 did not start from iteration0/fresh optimizer")
            if expected_initial is not None:
                reference = json.loads(Path(expected_initial).read_text())
                for key in ("normalized_parameters", "num_envs", "dt", "initial_state_sha256",
                            "initial_prefix_sha256", "policy_state_sha256", "rng_sha256",
                            "initial_lane_sha256", "optimizer_state_empty", "iteration"):
                    if data[key] != reference[key]:
                        raise ValueError(f"v18 paired start differs: {key}")
            save_json(audit_output, data)
            tracked = StyleExposureEnv(self.env, raw)
            self.env = tracked
            try:
                result = super().learn(*args, **kwargs)
            finally:
                self.env = tracked._vector_env
            save_json(exposure_output, {**tracked.summary(), "arm": arm, "created_utc": utc(),
                                        "initial_audit": str(audit_output.relative_to(ROOT))})
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
    parser.add_argument("--exposure-output", type=Path)
    parser.add_argument("--expected-initial", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if args.operation == "prepare":
        if args.output is None or forwarded or args.arm or args.audit_output or args.exposure_output:
            parser.error("prepare requires --output only")
        if not args.dry_run:
            print(json.dumps(prepare_checkpoint(args.output), indent=2))
        return
    if not args.arm or not args.audit_output or not args.exposure_output or args.output:
        parser.error("train requires --arm, --audit-output and --exposure-output")
    if args.audit_output.exists() or args.exposure_output.exists():
        raise FileExistsError("v18 train output exists")
    if args.expected_initial and not args.expected_initial.is_file():
        raise FileNotFoundError(args.expected_initial)
    forwarded = training_arguments(forwarded, args.arm)
    validate_budget(forwarded)
    script = ROOT / "scripts/train.py"
    if args.dry_run:
        print(json.dumps({"arm": args.arm, "backend": [str(script), *forwarded]}, indent=2))
        return
    import week03_ant.tasks.style_v18  # noqa: F401
    from week03_ant.style_policy_v18 import register_style_components

    register_style_components()
    old_argv = sys.argv
    try:
        sys.argv = [str(script), *forwarded]
        with audited_runner(args.arm, args.audit_output, args.exposure_output, args.expected_initial):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
