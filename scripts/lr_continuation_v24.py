"""Fixed no-cost v24 continuation with explicit seed, loaded-LR and per-Adam-step proof."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import importlib
from pathlib import Path
import runpy
import sys

from week03_ant.lr_study_v24 import (
    LearningRateAudit, learning_rate, TRAIN_SEEDS, ROOT, TRAIN_TASK, PARENT, EXPERIMENT, check_initial, prepare_checkpoint,
    save_json, state_hashes, tensor_sha, training_parameters, utc, validate_training_start,
)
from week03_ant.contact_continuation_v21 import audit_schedule
from contact_v16 import ContactSamplingEnv, contact_geometry


class ContinuationSamplingEnv(ContactSamplingEnv):
    """Forward RSL's random-horizon assignment to the real vector environment."""
    randomization_proof = None

    @property
    def episode_length_buf(self):
        return self._vector_env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        import torch
        self._vector_env.episode_length_buf = value
        raw = self._raw_env.episode_length_buf
        self.randomization_proof = dict(
            matches_raw_environment=torch.equal(raw, value),
            assigned_sha256=tensor_sha(value), actual_sha256=tensor_sha(raw),
            minimum=int(raw.min()), maximum=int(raw.max()), nonzero=int((raw != 0).sum()),
        )
        if not self.randomization_proof['matches_raw_environment']:
            raise ValueError('episode randomization was shadowed by proxy')


def training_arguments(forwarded, arm='no_cost', *, training_seed, ramp_steps=4000, condition="low"):
    if arm != 'no_cost' or type(training_seed) is not int or training_seed not in TRAIN_SEEDS or ramp_steps not in (32, 4000):
        raise ValueError('undeclared curriculum treatment')
    forwarded = list(forwarded)
    fixed = {
        'env.rewards.adaptive_posture.weight': '1.0',
        'env.rewards.contact_slip.weight': '1.0',
        'env.rewards.contact_slip.params.mode': arm,
        'env.rewards.contact_slip.params.ramp_steps': str(ramp_steps),
        'agent.device': 'cuda:1', 'agent.algorithm.learning_rate': str(learning_rate(condition)),
        'agent.policy.command_mode': 'conditioned', 'agent.policy.prior_mode': 'anchored',
        'agent.policy.input_mode': 'targets', 'agent.policy.class_name': 'CommandPriorActorCritic',
        'agent.algorithm.class_name': 'PriorPPO', 'agent.algorithm.teacher_coef': '0.02',
        'env.rewards.foothold_support.weight': '0.0', 'agent.experiment_name': EXPERIMENT,
        'agent.policy.require_config_match': 'true',
        'env.scene.terrain.terrain_generator.seed': '110',
    }
    allowed = {'--headless': False, '--device': True, '--num_envs': True, '--max_iterations': True,
               '--seed': True, '--resume': False, '--load_run': True, '--checkpoint': True,
               '--run_name': True, '--task': True}
    seen, overrides, index = {}, {}, 0
    while index < len(forwarded):
        arg = forwarded[index]
        if arg in allowed:
            if arg in seen or (allowed[arg] and index + 1 >= len(forwarded)):
                raise ValueError('duplicate/missing CLI value')
            seen[arg] = forwarded[index + 1] if allowed[arg] else True
            index += 2 if allowed[arg] else 1
        elif '=' in arg and not arg.startswith(('+', '-')):
            key, value = arg.split('=', 1)
            if key not in fixed or key in overrides or value != fixed[key]:
                raise ValueError('configuration substitution: ' + key)
            overrides[key] = value
            index += 1
        else:
            raise ValueError('unsupported training argument: ' + arg)
    if seen.get('--seed') != str(training_seed):
        raise ValueError('forwarded seed differs from requested training seed')
    if seen.get('--device') != 'cuda:1':
        raise ValueError('both simulator and agent must use cuda:1')
    if seen.get('--task', TRAIN_TASK) != TRAIN_TASK:
        raise ValueError('wrong opt-in task')
    if '--task' not in seen:
        forwarded += ['--task', TRAIN_TASK]
    for key, value in fixed.items():
        if key not in overrides:
            forwarded.append(f'{key}={value}')
    return forwarded


def validate_budget(forwarded, ramp_steps, training_seed, condition="low"):
    def value(flag):
        if forwarded.count(flag) != 1:
            raise ValueError('required unique flag: ' + flag)
        return forwarded[forwarded.index(flag) + 1]
    shape = (int(value('--num_envs')), int(value('--max_iterations')), ramp_steps)
    modes = [arg.split('=', 1)[1] for arg in forwarded if arg.startswith('env.rewards.contact_slip.params.mode=')]
    if modes != ['no_cost'] or type(training_seed) is not int or training_seed not in TRAIN_SEEDS:
        raise ValueError('seed study requires a declared seed and no-cost mode')
    if (shape not in ((256, 2, 32), (4096, 2, 4000), (4096, 250, 4000))
            or (shape[0] == 256 and training_seed != 51)
            or (shape[1] == 250 and condition == 'high' and training_seed != 51)):
        raise ValueError('undeclared env/iteration/ramp budget')
    if (value('--seed') != str(training_seed) or value('--load_run') != f'v24_{condition}_init'
            or value('--checkpoint') != 'model_0.pt' or forwarded.count('--resume') != 1):
        raise ValueError('initial checkpoint/seed substitution')


def seed_configuration(saved, live_seed, training_seed):
    if type(training_seed) is not int or training_seed not in TRAIN_SEEDS:
        raise ValueError('undeclared training seed')
    normalized = saved['normalized']
    if (normalized['agent']['seed'] != str(training_seed)
            or normalized['env']['seed'] != str(training_seed)
            or type(live_seed) is not int or live_seed != training_seed):
        raise ValueError('requested, saved agent/environment and live environment seeds differ')
    return dict(requested=training_seed, saved_agent=training_seed,
                saved_environment=training_seed, live_environment=live_seed)


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
        source.load_state_dict(torch.load(PARENT, map_location=device, weights_only=False)["model_state_dict"])
        source.eval()
        with torch.no_grad():
            for name, old, new in (
                ("actor", source.act_inference(selected), policy.act_inference(selected)),
                ("critic", source.evaluate(selected), policy.evaluate(selected)),
                ("teacher", source.teacher_mean(selected), policy.teacher_mean(selected)),
            ):
                if not torch.equal(old, new):
                    raise ValueError(f"initial same-device v16 parent {name} output differs")
    return {"actor": True, "critic": True, "teacher": True, "device": str(device)}


@contextmanager
def audited_runner(arm: str, output: Path, expected_path: Path | None = None, *, schedule_output: Path, training_seed: int, ramp_steps=4000, condition="low"):
    import torch
    import rsl_rl.runners as runners

    original = runners.OnPolicyRunner

    # Runtime runner replacement intentionally subclasses the installed RSL class.
    class AuditedRunner(original):  # type: ignore[valid-type,misc]
        def learn(self, *args, **kwargs):
            env = self.env.unwrapped
            with torch.no_grad():
                obs = self.env.get_observations()["policy"]
            robot = env.scene["robot"].data
            from week03_ant.tasks.lr_continuation_v24_cfg import validate_lr_config_parity
            if validate_lr_config_parity() is not True:
                raise ValueError("v16 config parity check did not pass")
            contact = contact_geometry(env)  # Required even when RewardManager skips weight-zero terms.
            parity = initial_policy_parity(self.alg.policy, obs)
            saved = training_parameters(self.log_dir, training_seed, ramp_steps, condition)
            actual_seed = seed_configuration(saved, env.cfg.seed, training_seed)
            data = {
                "arm": arm, "lr_condition": condition, "learning_rate": learning_rate(condition), "training_seed": training_seed, "actual_seed": actual_seed,
                "created_utc": utc(), "log_dir": str(Path(self.log_dir).relative_to(ROOT)),
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
                "iteration": self.current_learning_iteration, "common_step_counter": env.common_step_counter,
                "optimizer_state_empty": not bool(self.alg.optimizer.state_dict()["state"]),
            }
            save_json(output, data)
            if (data["contact_slip_weight"] != 1. or env.common_step_counter != 0 or data["iteration"] != 0
                    or not data["optimizer_state_empty"] or obs.shape != (env.num_envs, 91)):
                raise ValueError("actual starting training config differs from declared arm")
            validate_training_start(self.alg.policy.state_dict(), self.alg.optimizer, training_seed, condition)
            if expected_path is not None:
                check_initial(data, json.loads(Path(expected_path).read_text()))
            term_cfg = env.reward_manager.get_term_cfg("contact_slip")
            reward = term_cfg.func
            if term_cfg.weight != 1. or term_cfg.params != {"mode": arm, "ramp_steps": ramp_steps} or reward.history:
                raise ValueError("actual reward instance/config/history differs")
            lr_audit = LearningRateAudit(self.alg, saved["normalized"]["agent"]["algorithm"], condition)
            lr_audit.install()
            sampled_env, self.env = ContinuationSamplingEnv(self.env, env), None
            self.env = sampled_env
            try:
                result = super().learn(*args, **kwargs)
            finally:
                self.env = sampled_env._vector_env
                lr_proof = lr_audit.finish(kwargs.get("num_learning_iterations", args[0] if args else 0))
                save_json(output.with_name(output.stem + "_lr.json"), lr_proof)
                # Preserve actual reward-time evidence even when learning or a later
                # guard fails; exclusive writes prohibit silent replacement.
                schedule = reward.schedule(arm, ramp_steps)
                schedule.update(arm=arm, num_envs=env.num_envs,
                                episode_randomization=sampled_env.randomization_proof)
                save_json(schedule_output, schedule)
                try:
                    rollout = sampled_env.summary()
                except ValueError as error:
                    rollout = {"invalid_reason": str(error)}
                save_json(output.with_name(output.stem + "_contact_rollout.json"), {
                    "arm": arm, "created_utc": utc(), "source_initial_audit": str(output.relative_to(ROOT)),
                    "rollout": rollout, "passive_post_action": True,
                })
            if not lr_proof["passed"]:
                raise ValueError("incomplete PPO/Adam LR audit")
            steps = kwargs.get("num_learning_iterations", args[0] if args else 0) * 32
            audit_schedule(schedule, mode=arm, policy_steps=steps, ramp_steps=ramp_steps)
            if "invalid_reason" in rollout:
                raise ValueError(rollout["invalid_reason"])
            if not sampled_env.randomization_proof or not sampled_env.randomization_proof["matches_raw_environment"]:
                raise ValueError("random episode horizons did not reach actual environment")
            return result

    runners.OnPolicyRunner = AuditedRunner
    try:
        yield
    finally:
        runners.OnPolicyRunner = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('setup', 'train'))
    parser.add_argument('--lr-condition', choices=('high', 'low'), required=True)
    parser.add_argument('--arm', choices=('no_cost',))
    parser.add_argument('--training-seed', type=int, choices=TRAIN_SEEDS)
    parser.add_argument('--prepare-output', type=Path)
    parser.add_argument('--audit-output', type=Path)
    parser.add_argument('--schedule-output', type=Path)
    parser.add_argument('--expected-initial', type=Path)
    parser.add_argument('--ramp-steps', type=int, default=4000)
    parser.add_argument('--dry-run', action='store_true')
    args, forwarded = parser.parse_known_args(argv)
    if args.operation == 'setup':
        if args.prepare_output is None or forwarded or args.arm or args.training_seed is not None or args.audit_output or args.schedule_output:
            parser.error('setup requires --prepare-output only')
        if not args.dry_run:
            print(json.dumps(prepare_checkpoint(args.prepare_output, args.lr_condition), indent=2))
        return
    if not args.arm or args.training_seed is None or args.audit_output is None or args.schedule_output is None or args.prepare_output:
        parser.error('train requires --training-seed --arm no_cost --audit-output --schedule-output')
    for output in (args.audit_output, args.schedule_output,
                   args.audit_output.with_name(args.audit_output.stem + '_contact_rollout.json'),
                   args.audit_output.with_name(args.audit_output.stem + '_lr.json')):
        if output.exists():
            raise FileExistsError(output)
    if args.expected_initial and not args.expected_initial.is_file():
        raise FileNotFoundError(args.expected_initial)
    forwarded = training_arguments(forwarded, args.arm, training_seed=args.training_seed, ramp_steps=args.ramp_steps, condition=args.lr_condition)
    validate_budget(forwarded, args.ramp_steps, args.training_seed, args.lr_condition)
    script = ROOT / 'scripts/train.py'
    if args.dry_run:
        print(json.dumps({'arm': args.arm, 'backend': [str(script), *forwarded]}, indent=2))
        return
    importlib.import_module("week03_ant.tasks.lr_continuation_v24")
    from week03_ant.command_policy import register_command_components
    register_command_components()
    old_argv = sys.argv
    try:
        sys.argv = [str(script), *forwarded]
        with audited_runner(args.arm, args.audit_output, args.expected_initial,
                            schedule_output=args.schedule_output, training_seed=args.training_seed, ramp_steps=args.ramp_steps, condition=args.lr_condition):
            runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
