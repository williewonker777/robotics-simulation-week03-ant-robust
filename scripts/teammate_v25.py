"""Closed-world v25 training bridge; audited live objects, no SDK file changes."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import runpy
import sys
from week03_ant.teammate_study_v25 import (
    ROOT, ART, TRAIN_TASK, EXPERIMENT, SEEDS, ARMS, PARENT, entropy, init_path,
    save_json, state_hashes, tensor_sha, utc, training_parameters, validate_training_start,
)
from contact_v16 import ContactSamplingEnv, contact_geometry


def training_arguments(forwarded, arm, *, training_seed):
    entropy(arm)
    init_path(training_seed)
    forwarded = list(forwarded)
    fixed = {'env.rewards.adaptive_posture.weight':'1.0', 'env.rewards.contact_slip.weight':'1.0',
        'env.rewards.contact_slip.params.mode':'no_cost', 'env.rewards.contact_slip.params.ramp_steps':'4000',
        'env.rewards.teammate_recovery.weight':'1.0',
        'env.rewards.teammate_recovery.params.enabled':str(arm != 'control').lower(),
        'env.rewards.foothold_support.weight':'0.0', 'agent.device':'cuda:1',
        'agent.policy.command_mode':'conditioned', 'agent.policy.prior_mode':'anchored',
        'agent.policy.input_mode':'targets', 'agent.policy.class_name':'CommandPriorActorCritic',
        'agent.policy.require_config_match':'true', 'agent.algorithm.class_name':'PriorPPO',
        'agent.algorithm.teacher_coef':'0.02', 'agent.algorithm.entropy_coef':str(entropy(arm)),
        'agent.experiment_name':EXPERIMENT, 'env.scene.terrain.terrain_generator.seed':'130'}
    allowed = {'--headless':False, '--device':True, '--num_envs':True, '--max_iterations':True,
               '--seed':True, '--resume':False, '--load_run':True, '--checkpoint':True,
               '--run_name':True, '--task':True}
    seen, overrides, i = {}, {}, 0
    while i < len(forwarded):
        arg = forwarded[i]
        if arg in allowed:
            if arg in seen or (allowed[arg] and i+1 == len(forwarded)):
                raise ValueError('duplicate/missing CLI argument')
            seen[arg] = forwarded[i+1] if allowed[arg] else True
            i += 2 if allowed[arg] else 1
        elif '=' in arg and not arg.startswith(('+','-')):
            key,value = arg.split('=',1)
            if key not in fixed or key in overrides or value != fixed[key]:
                raise ValueError('configuration substitution: '+key)
            overrides[key] = value
            i += 1
        else:
            raise ValueError('unsupported argument: '+arg)
    required = {'--seed':str(training_seed), '--device':'cuda:1', '--load_run':f'v25_init_seed{training_seed}',
                '--checkpoint':'model_0.pt', '--resume':True, '--headless':True}
    if any(seen.get(key) != value for key,value in required.items()):
        raise ValueError('seed/device/initializer changed')
    shape = (int(seen.get('--num_envs',0)),int(seen.get('--max_iterations',0)))
    if shape not in ((256,2),(4096,2),(4096,250)) or (shape == (256,2) and training_seed != 61):
        raise ValueError('undeclared development/main budget')
    phase = 'preflight' if shape[0] == 256 else 'capacity' if shape[1] == 2 else 'main'
    if seen.get('--run_name') != f'v25_{phase}_{arm}{training_seed}' or seen.get('--task',TRAIN_TASK) != TRAIN_TASK:
        raise ValueError('task/run identity changed')
    if '--task' not in seen:
        forwarded += ['--task',TRAIN_TASK]
    return forwarded + [f'{key}={value}' for key,value in fixed.items() if key not in overrides]


class SamplingEnv(ContactSamplingEnv):
    randomization_proof = None

    @property
    def episode_length_buf(self):
        return self._vector_env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        import torch
        self._vector_env.episode_length_buf = value
        raw = self._raw_env.episode_length_buf
        self.randomization_proof = dict(matches_raw_environment=torch.equal(value,raw),
            assigned_sha256=tensor_sha(value), actual_sha256=tensor_sha(raw), minimum=int(raw.min()),
            maximum=int(raw.max()), nonzero=int((raw != 0).sum()))
        if not self.randomization_proof['matches_raw_environment']:
            raise ValueError('random horizons shadowed')


def initial_policy_parity(policy, observations):
    import torch
    from week03_ant.command_policy import CommandPriorActorCritic
    device, selected = observations.device, {'policy':observations}
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        source = CommandPriorActorCritic(selected, {'policy':['policy'],'critic':['policy']},8,
            input_mode='targets', prior_mode='anchored', command_mode='conditioned',require_config_match=True).to(device)
        source.load_state_dict(torch.load(PARENT,map_location=device,weights_only=False)['model_state_dict'])
        source.eval()
        with torch.no_grad():
            for name,method in (('actor','act_inference'),('critic','evaluate'),('teacher','teacher_mean')):
                if not torch.equal(getattr(source,method)(selected),getattr(policy,method)(selected)):
                    raise ValueError('initial parent '+name+' output differs')
    return dict(actor=True,critic=True,teacher=True,device=str(device))


def contact_schedule(reward, num_envs):
    """Attach the live vector count required by the immutable contact validator."""
    if type(num_envs) is not int or num_envs <= 0:
        raise ValueError('invalid live environment count')
    record = reward.schedule('no_cost', 4000)
    record['num_envs'] = num_envs
    return record


def compare_initial(actual, expected):
    keys = ('arm','training_seed','actual_seed','normalized_parameters','num_envs','dt',
            'observation_dimensions','initial_state_sha256','initial_prefix_sha256','policy_state_sha256','rng_sha256',
            'initial_policy_parity','optimizer_state_empty','iteration','common_step_counter')
    for key in keys:
        if actual[key] != expected[key]:
            raise ValueError('own capacity/main initial mismatch: '+key)


@contextmanager
def audited_runner(arm, seed, output, schedule_output, expected=None):
    import torch
    import rsl_rl.runners as runners
    original = runners.OnPolicyRunner
    class AuditedRunner(original):
        def learn(self, *args, **kwargs):
            env = self.env.unwrapped
            from week03_ant.tasks.teammate_v25_cfg import validate_config_parity
            validate_config_parity()
            with torch.no_grad():
                obs = self.env.get_observations()['policy']
            robot = env.scene['robot'].data
            saved = training_parameters(self.log_dir,arm,seed)
            if type(env.cfg.seed) is not int or env.cfg.seed != seed:
                raise ValueError('live environment seed changed')
            validate_training_start(self.alg.policy.state_dict(),self.alg.optimizer)
            data = dict(arm=arm,training_seed=seed,actual_seed=dict.fromkeys(
                ('requested','saved_agent','saved_environment','live_environment'),seed), created_utc=utc(),
                log_dir=str(Path(self.log_dir).relative_to(ROOT)), normalized_parameters=saved['normalized'],
                saved_parameter_sha256=saved['sha256'], num_envs=env.num_envs,dt=float(env.step_dt),
                observation_dimensions=list(obs.shape), initial_state_sha256={
                    'root_state':tensor_sha(robot.root_state_w),'joint_pos':tensor_sha(robot.joint_pos),
                    'joint_vel':tensor_sha(robot.joint_vel),'observations':tensor_sha(obs)},
                initial_prefix_sha256=tensor_sha(obs[:,:88]),initial_policy_parity=initial_policy_parity(self.alg.policy,obs),
                policy_state_sha256=state_hashes(self.alg.policy.state_dict()),
                rng_sha256={'cpu':tensor_sha(torch.get_rng_state()),'cuda':tensor_sha(torch.cuda.get_rng_state(env.device))},
                contact_preflight=contact_geometry(env),iteration=self.current_learning_iteration,
                common_step_counter=env.common_step_counter,optimizer_state_empty=not self.alg.optimizer.state_dict()['state'])
            save_json(output,data)
            if (data['iteration'] != 0 or env.common_step_counter != 0 or obs.shape != (env.num_envs,91)
                    or abs(env.step_dt-1/60)>1e-12):
                raise ValueError('actual initialization shape/iteration/dt changed')
            if expected is not None:
                compare_initial(data,json.loads(Path(expected).read_text()))
            recovery_cfg = env.reward_manager.get_term_cfg('teammate_recovery')
            contact_cfg = env.reward_manager.get_term_cfg('contact_slip')
            if (recovery_cfg.weight != 1. or recovery_cfg.params != {'enabled':arm != 'control'}
                    or contact_cfg.weight != 1. or contact_cfg.params != {'mode':'no_cost','ramp_steps':4000}
                    or recovery_cfg.func.history or contact_cfg.func.history):
                raise ValueError('live reward binding differs')
            counts = dict(ppo_updates=0,adam_steps=0)
            runtime_trace = []
            algorithm, optimizer = self.alg, self.alg.optimizer
            original_update,original_step = algorithm.update,optimizer.step
            def validate_runtime():
                if (algorithm.entropy_coef != entropy(arm) or algorithm.teacher_coef != .02
                        or algorithm.num_learning_epochs != 5 or algorithm.num_mini_batches != 4
                        or algorithm.schedule != 'fixed' or algorithm.desired_kl is not None or algorithm.learning_rate != 1e-4
                        or any(group['lr'] != 1e-4 for group in optimizer.param_groups)):
                    raise ValueError('runtime PPO coefficient/learning schedule changed')
                return dict(entropy_coef=float(algorithm.entropy_coef),teacher_coef=float(algorithm.teacher_coef),
                    learning_rate=float(algorithm.learning_rate),schedule=algorithm.schedule,desired_kl=algorithm.desired_kl,
                    optimizer_groups=[{key: value for key,value in group.items() if key != 'params'}
                                      for group in optimizer.param_groups])
            runtime_initial = validate_runtime()
            def counted_update(*a,**kw):
                before = validate_runtime()
                result = original_update(*a,**kw)
                if any(not torch.isfinite(torch.as_tensor(value)).all() for value in result.values()):
                    raise ValueError('nonfinite PPO scalar')
                counts['ppo_updates'] += 1
                runtime_trace.append(dict(update=counts['ppo_updates'],adam_steps=counts['adam_steps'],
                                          before=before,after=validate_runtime()))
                return result
            def counted_step(*a,**kw):
                validate_runtime()
                result = original_step(*a,**kw)
                counts['adam_steps'] += 1
                return result
            validate_runtime()
            algorithm.update,optimizer.step = counted_update,counted_step
            sampled = SamplingEnv(self.env,env)
            self.env = sampled
            try:
                result = super().learn(*args,**kwargs)
            finally:
                self.env = sampled._vector_env
                algorithm.update,optimizer.step = original_update,original_step
                schedule = dict(arm=arm,training_seed=seed,num_envs=env.num_envs, **counts,
                    entropy_coef=entropy(arm),teacher_coef=.02,learning_rate=1e-4,
                    policy_steps=len(recovery_cfg.func.history),recovery=recovery_cfg.func.history,
                    runtime_initial=runtime_initial,runtime_final=validate_runtime(),runtime_updates=runtime_trace,
                    contact=contact_schedule(contact_cfg.func,env.num_envs), episode_randomization=sampled.randomization_proof)
                save_json(schedule_output,schedule)
                save_json(output.with_name(output.stem+'_contact_rollout.json'),
                    dict(arm=arm,passive_post_action=True,rollout=sampled.summary()))
            iterations = kwargs.get('num_learning_iterations',args[0] if args else 0)
            if counts != dict(ppo_updates=iterations,adam_steps=iterations*20):
                raise ValueError('actual PPO/Adam budget differs')
            if schedule['policy_steps'] != iterations*32:
                raise ValueError('actual policy budget differs')
            return result
    runners.OnPolicyRunner = AuditedRunner
    try:
        yield
    finally:
        runners.OnPolicyRunner = original


def validate_expected_initial(forwarded, arm, seed, expected):
    """Reject missing/substituted main proof before simulator startup."""
    iterations = int(forwarded[forwarded.index('--max_iterations') + 1])
    if iterations != 250:
        if expected is not None:
            raise ValueError('development cannot substitute expected main initialization')
        return
    canonical = ART / 'expected_main' / f'{arm}{seed}_initial.json'
    if (expected is None or expected.absolute() != canonical.absolute()
            or expected.is_symlink()):
        raise ValueError('main requires its exact frozen expected initialization path')
    if not expected.is_file():
        raise FileNotFoundError(expected)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--training-seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--audit-output',type=Path,required=True)
    parser.add_argument('--schedule-output',type=Path,required=True)
    parser.add_argument('--expected-initial',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args,forwarded=parser.parse_known_args(argv)
    forwarded=training_arguments(forwarded,args.arm,training_seed=args.training_seed)
    validate_expected_initial(forwarded,args.arm,args.training_seed,args.expected_initial)
    for path in (args.audit_output,args.schedule_output,args.audit_output.with_name(args.audit_output.stem+'_contact_rollout.json')):
        if path.exists():
            raise FileExistsError(path)
    if args.dry_run:
        print(json.dumps(forwarded,indent=2)); return
    importlib.import_module('week03_ant.tasks.teammate_v25')
    from week03_ant.command_policy import register_command_components
    register_command_components()
    script=ROOT/'scripts/train.py'
    old=sys.argv
    try:
        sys.argv=[str(script),*forwarded]
        with audited_runner(args.arm,args.training_seed,args.audit_output,args.schedule_output,args.expected_initial):
            runpy.run_path(str(script),run_name='__main__')
    finally:
        sys.argv=old

if __name__=='__main__':
    main()
