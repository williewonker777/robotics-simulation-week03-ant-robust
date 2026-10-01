# SPDX-License-Identifier: BSD-3-Clause
"""v25 metadata adapter; the original numerical evaluator is immutable."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'scripts/evaluate_unseen_terrain.py'
BASE_SHA = 'cf512d2833d5d3a1d808735cf40a1b91cbbbc6d80db2928acc22982450bcd422'
SCHEMA = 'week03_ant_teammate_v25_eval_v1'
LEGACY = ('v5', 'v16_control', 'history_control', 'high53', 'history_high53')
RUNS = tuple(f'{arm}{seed}' for seed in (61, 62, 63) for arm in ('control', 'recovery', 'combined'))
CONTROLLERS = LEGACY + RUNS + tuple('history_' + r for r in RUNS)
DEVELOPMENT = ('v16_control', 'history_control') + RUNS[:3] + tuple('history_' + r for r in RUNS[:3])

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_base():
    if sha(BASE) != BASE_SHA:
        raise ValueError('immutable base evaluator changed')
    spec = importlib.util.spec_from_file_location('v25_immutable_evaluator', BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def controller_binding(controller, phase, base=None):
    if controller not in CONTROLLERS or phase not in ('development', 'holdout'):
        raise ValueError('unknown controller or phase')
    if phase == 'development' and controller not in DEVELOPMENT:
        raise ValueError('development endpoint not declared')
    base = base or load_base()
    if controller in LEGACY:
        return base.resolve_model(controller)
    from week03_ant.teammate_study_v25 import model_entry
    run = controller.removeprefix('history_')
    entry = model_entry(run, phase)
    checkpoint = ROOT / entry['checkpoint']
    if sha(checkpoint) != entry['sha256']:
        raise ValueError('model binding mismatch')
    return checkpoint, entry['sha256'], 'hybrid' if controller.startswith('history_') else 'v10', 'conditioned'

def child_command(phase, arguments):
    return [sys.executable, str(Path(__file__).resolve()), '--numerical-child',
            '--phase', phase, *arguments]


def run_numerical_child(command, raw_output):
    """Kit can terminate its process in app.close(); enrichment lives outside it."""
    if raw_output.exists():
        raise FileExistsError(raw_output)
    subprocess.run(command, check=True, cwd=ROOT)
    if not raw_output.is_file():
        raise RuntimeError('numerical child exited without original raw evidence')


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--numerical-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=('development', 'holdout'), required=True)
    wrapper, rest = parser.parse_known_args(argv)
    base = load_base()
    original_resolve = base.resolve_model
    # Exactly three allowed global substitutions; physics, tracker and inference unchanged.
    base.CONTROLLERS = DEVELOPMENT if wrapper.phase == 'development' else CONTROLLERS
    base.SCHEMA = SCHEMA
    class Legacy:
        resolve_model = staticmethod(original_resolve)
    base.resolve_model = lambda c: controller_binding(c, wrapper.phase, Legacy)
    args = base.parse_args(rest)
    pairs = ((51, 24),) if wrapper.phase == 'development' else ((131, 111), (132, 112))
    expected_n = 35 if wrapper.phase == 'development' else 175
    if (args.geometry, args.seed) not in pairs or args.num_envs != expected_n or args.seconds != 64 or args.scenario != 'mixed' or args.record:
        raise ValueError('not a preregistered evaluation cell')
    if wrapper.numerical_child:
        # app.close() may exit here without returning. No required work follows.
        base.launch(rest)
        return
    from week03_ant.teammate_study_v25 import source_hashes
    sources = source_hashes()
    checkpoint, digest, _, _ = base.resolve_model(args.controller)
    raw_output = args.output.parent / '_adapter_raw' / args.output.name
    launch_args = list(rest)
    for i, item in enumerate(launch_args):
        if item == '--output':
            launch_args[i + 1] = str(raw_output)
            break
        if item.startswith('--output='):
            launch_args[i] = '--output=' + str(raw_output)
            break
    else:
        raise ValueError('explicit --output required')
    command = child_command(wrapper.phase, launch_args)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.with_suffix('.child_command.json').open('x') as stream:
        json.dump(dict(command=command, cwd=str(ROOT)), stream, indent=2)
    run_numerical_child(command, raw_output)
    if sha(BASE) != BASE_SHA or source_hashes() != sources or sha(checkpoint) != digest:
        raise ValueError('adapter inputs changed during evaluation')
    result = json.loads(raw_output.read_text())
    result['v25'] = dict(phase=wrapper.phase, adapter_source_sha256=sources,
        adapter_source_sha256_after=source_hashes(), base_evaluator_sha256=BASE_SHA,
        substitutions=['CONTROLLERS', 'SCHEMA', 'resolve_model'],
        model_phase=wrapper.phase, scoring_training_transitions=0,
        numerical_child_command=command, numerical_child_returncode=0)
    # Preserve the original raw file as well; exclusively create enriched evidence.
    result['v25']['original_raw_sha256'] = sha(raw_output)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')

if __name__ == '__main__':
    main()
