"""Fixed global learning-rate continuation, initialization and study contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from . import seed_study_v22 as high
from . import paired_horizon_study_v23 as prior
from .posture_study import (ROOT as ROOT, START as START, V5_SHA as V5_SHA,
    save_json as save_json, sha as sha, state_hashes as state_hashes,
    tensor_sha as tensor_sha, utc as utc, validate_teacher as validate_teacher,
    verify_hashes as verify_hashes)
from .command_study import check_initial as check_initial

__all__ = ('START', 'V5_SHA', 'save_json', 'tensor_sha', 'utc', 'check_initial')

ART = ROOT / 'artifacts/terrain_demo/lr_continuation_v24'
WORK = ROOT / 'outputs/lr_continuation_v24_20260928'
PLAN = ROOT / 'docs/experiment_plans/lr_continuation_v24.md'
EXPERIMENT = 'week03_ant_lr_continuation_v24'
TRAIN_TASK = 'Week03-Ant-LR-Continuation-v24-Train-v0'
TASK = high.TASK
TRAIN_SEEDS = (51, 52, 53)
RUNS = ('low51', 'low52', 'low53')
HIGH_RUNS = high.RUNS
LR_VALUES = {'high': 1.e-4, 'low': 1.e-5}
TRAIN_SEED, TRAIN_GEOMETRY = 51, 110
ENVS, ITERATIONS, STEPS_PER_ENV, RAMP_STEPS = 4096, 250, 32, 4000
TRANSITIONS = 32768000
TOTAL_TRANSITIONS = 98304000
DEVELOPMENT_TRANSITIONS = 1605632
NEW_TRANSITIONS = 132677632
PARENT, PARENT_SHA, INIT_SOURCE, INIT_SHA = high.PARENT, high.PARENT_SHA, high.INIT_SOURCE, high.INIT_SHA
CONTROLLERS = ('parent', *HIGH_RUNS, *RUNS, 'history_parent',
               *('history_' + run for run in HIGH_RUNS), *('history_' + run for run in RUNS))
COMPARISONS = tuple((prefix + low, prefix + old) for prefix in ('', 'history_') for low, old in zip(RUNS, HIGH_RUNS)) + tuple((prefix + run, prefix + 'parent') for runs in (RUNS, HIGH_RUNS) for prefix in ('', 'history_') for run in runs)
HOLDOUTS = ((119, 79), (120, 80))
WINDOWS, PHYSICAL_SECONDS, EVAL_ENVS = (16, 64), 64, 175
PHYSICAL_ROLLOUTS, FIRST_EPISODES, WINDOW_OBSERVATIONS = 28, 4900, 9800
EXPECTED_GPU_COMMANDS = 52
REFERENCE_CONTROLLERS = ('parent', 'history_parent')
DEVELOPMENT_CONTROLLERS = ('parent', 'history_parent', *RUNS, *('history_' + name for name in RUNS))
SCHEMA = 'week03_ant_lr_continuation_v24_window_v1'
BUNDLE_SCHEMA = 'week03_ant_lr_continuation_v24_bundle_v1'
TRAINING_SOURCES = ('src/week03_ant/lr_study_v24.py', 'src/week03_ant/tasks/lr_continuation_v24.py',
    'src/week03_ant/tasks/lr_continuation_v24_cfg.py', 'scripts/lr_continuation_v24.py',
    'scripts/run_lr_training_v24.py', 'tests/test_lr_training_v24.py', 'tests/test_lr_harness_v24.py',
    'docs/experiment_plans/lr_continuation_v24.md')
EVALUATION_SOURCES = ('scripts/evaluate_lr_v24.py', 'scripts/run_lr_eval_v24.py',
                      'scripts/summarize_lr_v24.py', 'tests/test_lr_eval_v24.py')
SOURCE_FILES = (*TRAINING_SOURCES, *EVALUATION_SOURCES)


def read(path):
    return json.loads(Path(path).read_text())


def inside(name):
    path = Path(name)
    resolved = (ROOT / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError('v24 provenance path escapes repository')
    return resolved


def learning_rate(condition):
    if condition not in LR_VALUES:
        raise ValueError('undeclared LR condition')
    return LR_VALUES[condition]


def training_seed(run):
    if run not in (*RUNS, *HIGH_RUNS):
        raise ValueError('undeclared trained run')
    return int(run[-2:])


def model_key(controller):
    if controller not in CONTROLLERS:
        raise ValueError('undeclared controller')
    return controller.removeprefix('history_')


def lr_condition(controller):
    key = model_key(controller)
    return 'not_applicable' if key == 'parent' else 'low' if key in RUNS else 'high'


def model_learning_rate(controller):
    condition = lr_condition(controller)
    return None if condition == 'not_applicable' else learning_rate(condition)


def policy_mode(controller):
    model_key(controller)
    return 'hybrid' if controller.startswith('history_') else 'v10'


def command_mode(controller):
    model_key(controller)
    return 'conditioned'


def source_hashes(training=False):
    return {name: sha(ROOT / name) for name in (TRAINING_SOURCES if training else SOURCE_FILES)}


def legacy_hashes():
    frozen = prior.verify_frozen()
    post = read(prior.ART / 'static_validation.json')['post_experiment_verifier_sha256']
    result = {'sources': {**frozen['legacy']['sources'], **frozen['source_sha256'], **post},
              'models': frozen['legacy']['models']}
    verify_hashes({**result['sources'], **result['models']})
    if len(result['sources']) != 237 or len(result['models']) != 22:
        raise ValueError('v24 legacy inventory differs')
    return result


def init_path(condition='low'):
    learning_rate(condition)
    return ROOT / f'logs/rsl_rl/{EXPERIMENT}/v24_{condition}_init/model_0.pt'


def validate_init_difference(high_state, low_state):
    """Exact deserialized comparison with only empty Adam group LR projected."""
    import torch
    def exact(a, b):
        if torch.is_tensor(a):
            return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
        return a == b
    candidate = copy.deepcopy(low_state)
    for state, expected in ((high_state, 1.e-4), (candidate, 1.e-5)):
        optimizer = state['optimizer_state_dict']
        if state['iter'] != 0 or optimizer['state'] or not optimizer['param_groups']:
            raise ValueError('initial iteration/Adam state differs')
        if any(group['lr'] != expected for group in optimizer['param_groups']):
            raise ValueError('initial saved LR differs')
    for group in candidate['optimizer_state_dict']['param_groups']:
        group['lr'] = 1.e-4
    if not exact(high_state, candidate):
        raise ValueError('initializers differ outside Adam LR')
    return True


def prepare_checkpoint(destination, condition='low'):
    import torch
    rate = learning_rate(condition)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if sha(INIT_SOURCE) != INIT_SHA:
        raise ValueError('historical initializer changed')
    original = torch.load(INIT_SOURCE, map_location='cpu', weights_only=False)
    candidate = copy.deepcopy(original)
    for group in candidate['optimizer_state_dict']['param_groups']:
        group['lr'] = 1.e-5
    validate_init_difference(original, candidate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        if condition == 'high':
            stream.write(INIT_SOURCE.read_bytes())
        else:
            torch.save(candidate, stream)
    saved = torch.load(destination, map_location='cpu', weights_only=False)
    validate_init_difference(original, saved if condition == 'low' else candidate)
    return dict(checkpoint=str(destination.relative_to(ROOT)), sha256=sha(destination),
                source_checkpoint=str(INIT_SOURCE.relative_to(ROOT)), source_sha256=INIT_SHA,
                parent_sha256=PARENT_SHA, lr_condition=condition, learning_rate=rate,
                state_sha256=state_hashes(saved['model_state_dict']), optimizer_state_empty=True,
                historical_infos_preserved=True, only_optimizer_group_lr_changed=condition == 'low')


def validate_training_start(state, optimizer, seed=None, condition='low'):
    import torch
    if type(seed) is not int or seed not in TRAIN_SEEDS:
        raise ValueError('undeclared seed')
    original = torch.load(INIT_SOURCE, map_location='cpu', weights_only=False)
    if sha(INIT_SOURCE) != INIT_SHA or state_hashes(state) != state_hashes(original['model_state_dict']):
        raise ValueError('initial policy changed')
    saved = optimizer.state_dict()
    expected = copy.deepcopy(original['optimizer_state_dict'])
    for group in expected['param_groups']:
        group['lr'] = learning_rate(condition)
    if type(optimizer) is not torch.optim.Adam or saved != expected:
        raise ValueError('loaded initial Adam/LR differs; never coerce')
    validate_teacher(state)


def training_parameters(log_directory, seed, ramp_steps=RAMP_STEPS, condition='low'):
    from .posture_study import training_parameters as parameters
    result = parameters(log_directory, 'adaptive')
    env, agent = (result['normalized'][key] for key in ('env', 'agent'))
    if type(seed) is not int or seed not in TRAIN_SEEDS or any(int(v['seed']) != seed for v in (env, agent)):
        raise ValueError('saved seeds differ')
    term = env['rewards']['contact_slip']
    if (float(term['weight']) != 1. or term['params'] != {'mode': 'no_cost', 'ramp_steps': str(ramp_steps)}
            or term['func'] != 'week03_ant.tasks.contact_continuation_v21_cfg:ContinuationContactSlipReward'
            or int(env['scene']['terrain']['terrain_generator']['seed']) != TRAIN_GEOMETRY
            or agent['experiment_name'] != EXPERIMENT or agent['load_run'] != f'v24_{condition}_init'
            or agent['load_checkpoint'] != 'model_0.pt'
            or float(agent['algorithm']['learning_rate']) != learning_rate(condition)
            or agent['algorithm']['schedule'] != 'fixed' or agent['algorithm']['desired_kl'] != 'null'):
        raise ValueError('saved training configuration/LR differs')
    # Only enumerated opt-in identities normalize to the frozen common v22 format.
    term['func'] = 'week03_ant.tasks.contact_curriculum_v20_cfg:CurriculumContactSlipReward'
    term['params']['mode'] = '<paired-coefficient-trajectory>'
    agent['experiment_name'], agent['load_run'] = 'week03_ant_contact_curriculum_v20', 'v20_init'
    return result


def models():
    result = high.models()
    if set(result) != {'parent', *HIGH_RUNS}:
        raise ValueError('historical model inventory differs')
    if result['parent']['checkpoint'] != str(PARENT.relative_to(ROOT)) or result['parent']['sha256'] != PARENT_SHA:
        raise ValueError('parent binding differs')
    for key in HIGH_RUNS:
        if result[key]['checkpoint'] != str((high.ART / 'runs' / key / 'model_249.pt').relative_to(ROOT)):
            raise ValueError('historical high model path differs')
    path = ART / 'trained_models.json'
    if path.exists():
        trained = read(path)
        if set(trained['models']) != set(RUNS):
            raise ValueError('low model inventory differs')
        if (trained['total_training_transitions'] != TOTAL_TRANSITIONS
                or trained['new_training_transitions'] != NEW_TRANSITIONS
                or trained['training_freeze_sha256'] != sha(ART / 'training_frozen.json')
                or trained['training_validation_sha256'] != sha(ART / 'training_validation.json')
                or trained['training_commands_sha256'] != sha(ART / 'training_commands.jsonl')):
            raise ValueError('trained model provenance/budget differs')
        sources = [entry['source_checkpoint'] for entry in trained['models'].values()]
        if len(set(sources)) != 3 or any(not inside(name).is_relative_to(ROOT / 'logs/rsl_rl' / EXPERIMENT)
            or inside(name).name != 'model_249.pt' for name in sources):
            raise ValueError('fresh low checkpoint namespace/alias differs')
        result = {**result, **trained['models']}
    for key, entry in result.items():
        if sha(inside(entry['checkpoint'])) != entry['sha256']:
            raise ValueError('model bytes changed')
        if key in RUNS and (entry['training_seed'] != training_seed(key) or entry['learning_rate'] != 1.e-5
                            or entry['iteration'] != 249 or entry['transitions'] != TRANSITIONS
                            or entry.get('lr_condition') != 'low'
                            or entry['checkpoint'] != str((ART / 'runs' / key / 'model_249.pt').relative_to(ROOT))):
            raise ValueError('low model binding differs')
    return result


def validate_request(controller, geometry, seed, scenario, seconds, envs, phase, *, instrumented=True):
    model_key(controller)
    if (scenario != 'mixed' or type(seconds) is not int or seconds != 64 or not instrumented
            or any(type(value) is not int for value in (geometry, seed, envs))):
        raise ValueError('undeclared evaluation contract')
    if phase in ('prepare_development', 'smoke'):
        valid = (geometry, seed, envs) == (51, 24, 35) and controller in DEVELOPMENT_CONTROLLERS
    elif phase in ('prepare', 'holdout'):
        valid = (geometry, seed) in HOLDOUTS and envs == EVAL_ENVS
    else:
        valid = False
    if phase.startswith('prepare'):
        valid &= controller == 'parent'
    if not valid:
        raise ValueError('undeclared evaluation population')


def frozen_budget():
    return dict(controllers=list(CONTROLLERS), holdouts=[list(pair) for pair in HOLDOUTS],
                windows=list(WINDOWS), physical_seconds=64, new_training_transitions=NEW_TRANSITIONS,
                physical_rollouts=28, first_episodes=4900, window_observations=9800)


def verify_references():
    result = read(ART / 'historical_references.json')
    verify_hashes(result['sha256'])
    if (result['v23_frozen_sha256'] != sha(prior.ART / 'frozen.json')
            or result['evaluation_records'] != read(prior.ART / 'development.json')['records']):
        raise ValueError('historical evaluation reference identity differs')
    return result


def verify_frozen():
    frozen = read(ART / 'frozen.json')
    if (frozen['source_sha256'] != source_hashes() or frozen['legacy'] != legacy_hashes()
            or frozen['models'] != models() or set(models()) != {'parent', *HIGH_RUNS, *RUNS}
            or any(frozen.get(key) != value for key, value in frozen_budget().items())):
        raise ValueError('v24 evaluation freeze changed')
    for key, name in (('development_sha256', 'development.json'), ('training_frozen_sha256', 'training_frozen.json'),
                      ('trained_models_sha256', 'trained_models.json'), ('training_commands_sha256', 'training_commands.jsonl'),
                      ('historical_references_sha256', 'historical_references.json')):
        if frozen[key] != sha(ART / name):
            raise ValueError('v24 evidence changed: ' + name)
    verify_references()
    return frozen


def evaluation_inputs():
    from .lp_study_v17 import evaluation_inputs as inputs
    return inputs(ART)


class LearningRateAudit:
    """Observe actual PPO updates and Adam steps without changing optimizer state."""
    def __init__(self, algorithm, saved_algorithm, condition):
        import torch
        self.algorithm, self.saved = algorithm, saved_algorithm
        self.rate = learning_rate(condition)
        self.condition = condition
        self.parameters = tuple(algorithm.policy.named_parameters())
        self.groups = copy.deepcopy(algorithm.optimizer.state_dict()['param_groups'])
        self.teacher = {name: value.detach().clone() for name, value in algorithm.policy.state_dict().items()
                        if name.startswith('teacher.')}
        if type(algorithm.optimizer) is not torch.optim.Adam or not self.teacher:
            raise ValueError('expected Adam and frozen teacher')
        self.updates, self.steps = [], []
        self.initial = self.observe()
        self.initial_optimizer = copy.deepcopy(algorithm.optimizer.state_dict()['param_groups'])
        self.original_update, self.original_step = algorithm.update, algorithm.optimizer.step

    def observe(self):
        alg = self.algorithm
        groups = alg.optimizer.param_groups
        expected = [parameter for _, parameter in self.parameters]
        actual = [parameter for group in groups for parameter in group['params']]
        if (len(actual) != len(expected) or [id(p) for p in actual] != [id(p) for p in expected]
                or len(set(map(id, actual))) != len(actual)):
            raise ValueError('optimizer parameter coverage/order changed')
        if (float(self.saved['learning_rate']) != self.rate or self.saved['schedule'] != 'fixed'
                or self.saved['desired_kl'] != 'null' or alg.learning_rate != self.rate
                or alg.schedule != 'fixed' or alg.desired_kl is not None
                or any(group['lr'] != self.rate for group in groups)):
            raise ValueError('runtime/saved/Adam LR or schedule changed')
        current = alg.optimizer.state_dict()['param_groups']
        if current != self.groups:
            raise ValueError('Adam options or coverage changed')
        names = [name for name, parameter in self.parameters if parameter.requires_grad]
        if not all(any(name.startswith(prefix) for name in names) for prefix in ('actor.', 'critic.')) or 'std' not in names:
            raise ValueError('actor critic std trainable coverage missing')
        return dict(saved_learning_rate=float(self.saved['learning_rate']), algorithm_learning_rate=alg.learning_rate,
                    group_learning_rates=[group['lr'] for group in groups], schedule=alg.schedule,
                    desired_kl=alg.desired_kl, parameter_names=[name for name, _ in self.parameters],
                    trainable_parameter_names=names)

    def check_teacher(self):
        import torch
        state = self.algorithm.policy.state_dict()
        if any(not torch.equal(value, state[name]) for name, value in self.teacher.items()):
            raise ValueError('frozen teacher changed')

    def install(self):
        def step(*args, **kwargs):
            row = dict(index=len(self.steps), update_index=len(self.updates) - 1, before=self.observe())
            self.steps.append(row)
            result = self.original_step(*args, **kwargs)
            row['after'] = self.observe()
            return result
        def update(*args, **kwargs):
            self.check_teacher()
            row = dict(index=len(self.updates), before=self.observe(), step_start=len(self.steps))
            self.updates.append(row)
            result = self.original_update(*args, **kwargs)
            row.update(after=self.observe(), step_end=len(self.steps))
            self.check_teacher()
            if row['step_end'] - row['step_start'] != 20:
                raise ValueError('PPO update did not execute 20 Adam steps')
            return result
        self.algorithm.optimizer.step, self.algorithm.update = step, update

    def finish(self, expected_updates):
        self.algorithm.update, self.algorithm.optimizer.step = self.original_update, self.original_step
        failure = None
        try:
            final = self.observe()
            self.check_teacher()
        except ValueError as error:
            final, failure = None, str(error)
        result = dict(lr_condition=self.condition, learning_rate=self.rate, initial=self.initial, final=final,
                      initial_optimizer_groups=self.initial_optimizer,
                      final_optimizer_groups=copy.deepcopy(self.algorithm.optimizer.state_dict()['param_groups']),
                      updates=self.updates, adam_steps=self.steps, update_count=len(self.updates),
                      adam_step_count=len(self.steps), expected_updates=expected_updates,
                      teacher_unchanged=failure is None, failure_reason=failure)
        result['passed'] = (failure is None and len(self.updates) == expected_updates and len(self.steps) == 20 * expected_updates
                            and all('after' in row for row in (*self.updates, *self.steps)))
        return result


def verify_training_freeze():
    from run_lr_training_v24 import verify_training_freeze as verify
    return verify()


def verify_training_ledger():
    from run_lr_training_v24 import verify_training_ledger as verify
    return verify()
