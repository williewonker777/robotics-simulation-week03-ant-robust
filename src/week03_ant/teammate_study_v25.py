"""Closed v25 identities and provenance shared by training and evaluation."""
from pathlib import Path
import json
from .posture_study import ROOT, START, V5_SHA, save_json, sha, state_hashes, tensor_sha, utc, validate_teacher, verify_hashes
from .lp_study_v17 import V16_CONTROL as PARENT, V16_CONTROL_SHA as PARENT_SHA, validate_training_start

ART = ROOT / 'artifacts/terrain_demo/teammate_port_v25'
WORK = ROOT / 'outputs/teammate_port_v25_20261001'
PLAN = ROOT / 'docs/experiment_plans/teammate_port_v25.md'
EXPERIMENT = 'week03_ant_teammate_port_v25'
TRAIN_TASK = 'Week03-Ant-Teammate-v25-Train-v0'
TASK = 'Week03-Ant-Contact-v16-Eval-v0'
SEEDS = (61, 62, 63)
ARMS = ('control', 'recovery', 'combined')
RUNS = tuple(f'{arm}{seed}' for seed in SEEDS for arm in ARMS)
LEGACY_CONTROLLERS = ('v5', 'v16_control', 'history_control', 'high53', 'history_high53')
CONTROLLERS = (*LEGACY_CONTROLLERS, *RUNS, *('history_' + run for run in RUNS))
HOLDOUTS = ((131, 111), (132, 112))
ENVS, ITERATIONS, STEPS, EVAL_ENVS, SECONDS = 4096, 250, 32, 175, 64
WINDOWS = (16, 64)
TRAIN_GEOMETRY = 130
TRANSITIONS = ENVS * ITERATIONS * STEPS
TOTAL_TRANSITIONS = len(RUNS) * TRANSITIONS
DEVELOPMENT_TRANSITIONS = 3 * 256 * 32 * 2 + 9 * 4096 * 32 * 2
TRAINING_SOURCES = (
    'src/week03_ant/teammate_recovery_v25.py', 'src/week03_ant/teammate_study_v25.py',
    'src/week03_ant/tasks/teammate_v25_cfg.py', 'src/week03_ant/tasks/teammate_v25.py',
    'scripts/teammate_v25.py', 'scripts/run_teammate_training_v25.py',
    'tests/test_teammate_math_v25.py', 'tests/test_teammate_training_v25.py',
    'tests/test_teammate_harness_v25.py', 'docs/experiment_plans/teammate_port_v25.md',
)
SOURCE_FILES = (*TRAINING_SOURCES, 'scripts/evaluate_teammate_v25.py', 'scripts/run_teammate_eval_v25.py',
                'scripts/summarize_teammate_v25.py', 'tests/test_teammate_eval_v25.py',
                'tests/test_teammate_summary_v25.py', 'scripts/audit_teammate_v25_raw.py',
                'tests/test_teammate_audit_v25.py',
                'artifacts/terrain_demo/teammate_port_v25/research.md',
                'artifacts/terrain_demo/teammate_port_v25/research_provenance.json',
                'licenses/teammate_ant_BSD-3-Clause.txt')


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes(training=False):
    return {name: sha(ROOT / name) for name in (TRAINING_SOURCES if training else SOURCE_FILES)}


def run_identity(run):
    if run not in RUNS:
        raise ValueError('undeclared v25 run')
    return next((arm, seed) for seed in SEEDS for arm in ARMS if run == f'{arm}{seed}')


def entropy(arm):
    if arm not in ARMS:
        raise ValueError('undeclared v25 arm')
    return .005 if arm == 'combined' else .002


def init_path(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('undeclared initializer seed')
    return ROOT / f'logs/rsl_rl/{EXPERIMENT}/v25_init_seed{seed}/model_0.pt'


def prepare_checkpoint(seed):
    import torch
    from .command_policy import CommandPriorActorCritic
    destination = init_path(seed)
    if destination.exists():
        raise FileExistsError(destination)
    if sha(PARENT) != PARENT_SHA:
        raise ValueError('parent identity changed')
    torch.manual_seed(seed)
    policy = CommandPriorActorCritic({'policy': torch.zeros(1,91)},
        {'policy':['policy'], 'critic':['policy']}, 8, command_mode='conditioned',
        prior_mode='anchored', input_mode='targets', require_config_match=True)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    policy.load_state_dict(parent['model_state_dict'])
    with torch.no_grad():
        policy.std.fill_(.2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    validate_training_start(policy.state_dict(), optimizer)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(dict(model_state_dict=policy.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            iter=0, infos=dict(study=EXPERIMENT, seed=seed, starting_sha256=PARENT_SHA,
                              command_mode='conditioned', reset_std=.2)), stream)
    return dict(checkpoint=str(destination.relative_to(ROOT)), sha256=sha(destination), training_seed=seed,
                parent_sha256=PARENT_SHA, state_sha256=state_hashes(policy.state_dict()),
                all_non_std_tensors_identical=True, optimizer_state_empty=True, iteration=0)


def model_entry(run, phase='holdout'):
    _, seed = run_identity(run)
    if phase == 'development':
        if seed != 61:
            raise ValueError('development evaluates seed61 common initializer only')
        item = read(ART / 'initial_shared.json')['models'][str(seed)]
        if item['checkpoint'] != str(init_path(seed).relative_to(ROOT)) or item['iteration'] != 0:
            raise ValueError('development initializer substitution')
    elif phase == 'holdout':
        trained = read(ART / 'trained_models.json')
        if set(trained['models']) != set(RUNS):
            raise ValueError('all nine finals must freeze before scoring')
        item = trained['models'][run]
        if (item['iteration'] != 249 or item['training_seed'] != seed or item['transitions'] != TRANSITIONS
                or item['checkpoint'] != str((ART / 'runs' / run / 'model_249.pt').relative_to(ROOT))):
            raise ValueError('final checkpoint substitution')
        if trained['training_freeze_sha256'] != sha(ART / 'training_frozen.json'):
            raise ValueError('training freeze changed')
    else:
        raise ValueError('undeclared model phase')
    verify_hashes({item['checkpoint']: item['sha256']})
    return dict(item)


def training_parameters(log_directory, arm, seed):
    import yaml
    entropy(arm)
    init_path(seed)
    directory = Path(log_directory)
    paths = {key: directory / 'params' / f'{key}.yaml' for key in ('env','agent')}
    values = {key: yaml.load(path.read_text(), Loader=yaml.BaseLoader) for key,path in paths.items()}
    env, agent = values['env'], values['agent']
    if any(int(item['seed']) != seed for item in (env,agent)):
        raise ValueError('saved simulator/agent seed differs')
    term = env['rewards']['teammate_recovery']
    if (term['func'] != 'week03_ant.tasks.teammate_v25_cfg:TeammateRecoveryReward'
            or float(term['weight']) != 1. or term['params'] != {'enabled': str(arm != 'control').lower()}
            or float(agent['algorithm']['entropy_coef']) != entropy(arm)
            or int(env['scene']['terrain']['terrain_generator']['seed']) != TRAIN_GEOMETRY
            or agent['load_run'] != f'v25_init_seed{seed}' or agent['load_checkpoint'] != 'model_0.pt'
            or agent['experiment_name'] != EXPERIMENT):
        raise ValueError('saved treatment/geometry/initializer differs')
    for field in ('log_dir', 'io_descriptors_output_dir'):
        if Path(env.pop(field)).resolve() != directory.resolve():
            raise ValueError('saved output directory differs')
    agent.pop('run_name')
    return dict(normalized=values, sha256={key:sha(path) for key,path in paths.items()})
