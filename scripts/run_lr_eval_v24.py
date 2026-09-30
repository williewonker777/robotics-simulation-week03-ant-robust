"""Fixed v24 command ledger: serial GPU jobs, exclusive outputs, no scored retries."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
from week03_ant import lr_study_v24 as study
from week03_ant.posture_study import cache_snapshot
from summarize_lp_v17 import PARITY_FIELDS
from summarize_contact_v16 import _paired, _flat_identity
from summarize_lr_v24 import audit_bundle, summarize, markdown

ART, WORK, ROOT = study.ART, study.WORK, study.ROOT
CACHE = Path('/tmp/isaaclab/terrains')
PYTHON = str(ROOT.parent / 'run-python')
read = study.read


def eval_command(controller, geometry, reset, phase, output, device='cuda:1'):
    study.validate_request(controller, geometry, reset, 'mixed', 64, 35 if phase in ('prepare_development', 'smoke') else 175, phase)
    if device != 'cuda:1':
        raise ValueError('v24 is assigned to cuda:1')
    command = [PYTHON, 'scripts/evaluate_lr_v24.py', '--headless', '--device', device,
               '--controller', controller, '--geometry', str(geometry), '--seed', str(reset),
               '--scenario', 'mixed', '--seconds', '64', '--num_envs', str(35 if phase in ('prepare_development', 'reference', 'smoke') else 175),
               '--phase', phase, '--output', str(output)]
    return command


def run(command, label):
    log = WORK / f'{label}.log'
    output = Path(command[command.index('--output') + 1])
    if log.exists() or output.exists():
        raise FileExistsError(f'no implicit retries/overwrites: {label}')
    started, begin = study.utc(), time.monotonic()
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(label=label, command=command, started_utc=started, finished_utc=study.utc(),
                  seconds=time.monotonic() - begin, returncode=result.returncode,
                  log=str(log.relative_to(ROOT)), log_sha256=study.sha(log))
    if (ART / 'evaluation_inputs.json').exists():
        inputs = study.evaluation_inputs()
        record.update(evaluation_inputs_sha256=study.sha(ART / 'evaluation_inputs.json'),
                      terrain_cache_manifest_sha256=inputs['terrain_cache_manifest_sha256'])
    with (ART / 'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    if result.returncode or not output.is_file():
        raise RuntimeError(f'evaluator failed; preserve evidence and stop: {label}')
    return output


def record(path, controller=None):
    result = dict(path=str(path.relative_to(ROOT)), sha256=study.sha(path))
    if controller is not None:
        result['controller'] = controller
    return result


def parity(new, old):
    if len(PARITY_FIELDS) != 35 or any(k not in new or k not in old or new[k] != old[k] for k in PARITY_FIELDS):
        raise ValueError('exact 35-field development parity failed')


def evaluator_equivalence(candidate=None):
    """Compare entire executable AST, removing ONLY the preregistered additions."""
    original = ast.parse((ROOT / 'scripts/evaluate_paired_horizon_v23.py').read_text())
    current = ast.parse((ROOT / 'scripts/evaluate_lr_v24.py').read_text() if candidate is None else candidate)
    common = [node.value for node in ast.walk(current) if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'common' for t in node.targets)]
    if len(common) != 1 or not isinstance(common[0], ast.Dict):
        raise ValueError('common raw provenance dictionary changed')
    allowed_dictionary = common[0]
    class Normalize(ast.NodeTransformer):
        def visit_Module(self, node):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
            return self.generic_visit(node)
        def visit_ImportFrom(self, node):
            if node.module == 'week03_ant.lr_study_v24':
                node.module = 'week03_ant.paired_horizon_study_v23'
            return node
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                if node.value.startswith(('v24 ', '[v24] ')):
                    node.value = node.value.replace('v24', 'v23', 1)
            return node
        def visit_Dict(self, node):
            additions = [(k.value, v) for k, v in zip(node.keys, node.values)
                         if isinstance(k, ast.Constant) and k.value in ('lr_condition', 'model_learning_rate')]
            if additions:
                if node is not allowed_dictionary:
                    raise ValueError('LR additions outside common raw provenance')
                expected = {'lr_condition': '"not_applicable" if key == "parent" else ("low" if key.startswith("low") else "high")',
                            'model_learning_rate': 'None if key == "parent" else (1.e-5 if key.startswith("low") else 1.e-4)'}
                if (len(additions) != len(expected) or {k for k, _ in additions} != set(expected)
                        or any(ast.dump(value) != ast.dump(ast.parse(expected[key], mode='eval').body)
                               for key, value in additions)):
                    raise ValueError('LR provenance AST changed')
                self.addition_count = getattr(self, 'addition_count', 0) + 1
            kept = [(k, v) for k, v in zip(node.keys, node.values)
                    if not (isinstance(k, ast.Constant) and k.value in ('lr_condition', 'model_learning_rate'))]
            node.keys, node.values = [k for k, _ in kept], [v for _, v in kept]
            return self.generic_visit(node)
    normalizer = Normalize()
    current = normalizer.visit(current)
    if getattr(normalizer, 'addition_count', 0) != 1:
        raise ValueError('LR provenance additions missing/duplicated')
    if ast.dump(Normalize().visit(original), include_attributes=False) != ast.dump(current, include_attributes=False):
        raise ValueError('whole evaluator AST differs outside preregistered allowlist')
    return True


def verify_training_handoff():
    """Pin the separate 13-command training ledger before evaluation starts."""
    from run_lr_training_v24 import verify_training_ledger
    training = verify_training_ledger()
    trained = read(ART / 'trained_models.json')
    if len(training) != 13 or trained.get('training_commands_sha256') != study.sha(ART / 'training_commands.jsonl'):
        raise ValueError('training ledger completion/link differs')
    ledger = ART / 'commands.jsonl'
    if ledger.exists():
        evaluations = [json.loads(line) for line in ledger.read_text().splitlines()]
        if evaluations and not datetime.fromisoformat(training[-1]['finished_utc']) < datetime.fromisoformat(evaluations[0]['started_utc']):
            raise ValueError('evaluation started before training completion')
    return training


def evaluation_references():
    references = study.verify_references()
    return {r['controller']: read(study.inside(r['path'])) for r in references['evaluation_records']}


def verify_development(data=None):
    data = read(ART / 'development.json') if data is None else data
    evaluator_equivalence()
    verify_training_handoff()
    references = evaluation_references()
    if (data.get('passed') is not True or data.get('parity_fields') != list(PARITY_FIELDS)
            or data.get('source_sha256') != study.source_hashes() or data.get('legacy') != study.legacy_hashes()
            or data.get('models') != study.models() or data.get('terrain_cache') != cache_snapshot(CACHE, [51])
            or data.get('evaluator_ast_equivalent') is not True):
        raise ValueError('development source/model/cache/parity contract changed')
    if [r['controller'] for r in data['records']] != list(study.DEVELOPMENT_CONTROLLERS):
        raise ValueError('development controller order/count differs')
    for r in (*data['records'], data['preparation']):
        if study.sha(study.inside(r['path'])) != r['sha256']:
            raise ValueError('development raw evidence changed')
    if read(study.inside(data['preparation']['path'])).get('scored_episodes') != 0:
        raise ValueError('preparation scored episodes')
    initial, flats = {}, {}
    for r in data['records']:
        bundle = read(study.inside(r['path']))
        rows = audit_bundle(bundle)
        if not bundle['instrumented'] or bundle['windows']['16']['controller'] != r['controller']:
            raise ValueError('invalid paired development record')
        for window in ('16', '64'):
            raw = bundle['windows'][window]
            _paired(initial, (51, 24), raw)
            if r['controller'].startswith('history_'):
                _flat_identity(flats, (window, 51, 24), raw, rows[window])
            if r['controller'] in ('parent', 'history_parent'):
                parity(raw, references[r['controller']]['windows'][window])
    return data


def verify_ledger(*, complete=False):
    expected = [('parent', 51, 24, 'prepare_development', WORK / 'prepare_development.json', 'prepare_development')]
    for phase, controllers in (('smoke', study.DEVELOPMENT_CONTROLLERS),):
        expected.extend((c, 51, 24, phase, WORK / f'{phase}_{c}.json', f'{phase}_{c}') for c in controllers)
    for g, r in study.HOLDOUTS:
        label = f'prepare_geometry{g}'
        expected.append(('parent', g, r, 'prepare', WORK / f'{label}.json', label))
    if complete:
        for c in study.CONTROLLERS:
            for g, r in study.HOLDOUTS:
                label = f'{c}__geometry{g}_reset{r}'
                expected.append((c, g, r, 'holdout', ART / 'evaluations' / f'{label}.json', f'evaluations_{label}'))
    ledger = [json.loads(line) for line in (ART / 'commands.jsonl').read_text().splitlines()]
    if len(ledger) != len(expected):
        raise ValueError('fixed ledger budget differs')
    for entry, (c, g, r, phase, output, label) in zip(ledger, expected):
        if (entry.get('command') != eval_command(c, g, r, phase, output)
                or entry.get('label') != label or entry.get('returncode') != 0
                or entry.get('log') != str((WORK / f'{label}.log').relative_to(ROOT))
                or study.sha(study.inside(entry['log'])) != entry.get('log_sha256')):
            raise ValueError('fixed command order/log/retry contract differs')
    return ledger


def development(device):
    study.verify_training_freeze()
    verify_training_handoff()
    evaluator_equivalence()
    evaluation_references()
    before, legacy, models = study.source_hashes(), study.legacy_hashes(), study.models()
    prep = WORK / 'prepare_development.json'
    run(eval_command('parent', 51, 24, 'prepare_development', prep, device), 'prepare_development')
    if read(prep).get('scored_episodes') != 0:
        raise ValueError('development preparation scored episodes')
    cache = cache_snapshot(CACHE, [51])
    references = evaluation_references()
    initial, flats = {}, {}
    data = dict(passed=True, created_utc=study.utc(), source_sha256=before, legacy=legacy, models=models,
                terrain_cache=cache, preparation=record(prep), records=[], parity_fields=list(PARITY_FIELDS),
                evaluator_ast_equivalent=True)
    for controller in study.DEVELOPMENT_CONTROLLERS:
        for check in (0, 1):
            if (study.source_hashes() != before or study.legacy_hashes() != legacy or study.models() != models
                    or cache_snapshot(CACHE, [51]) != cache):
                raise ValueError('development source/model/cache changed')
            study.verify_references()
            if check == 0:
                output = WORK / f'smoke_{controller}.json'
                run(eval_command(controller, 51, 24, 'smoke', output, device), f'smoke_{controller}')
        bundle = read(output)
        rows = audit_bundle(bundle)
        for window in ('16', '64'):
            raw = bundle['windows'][window]
            if controller in ('parent', 'history_parent'):
                parity(raw, references[controller]['windows'][window])
            _paired(initial, (51, 24), raw)
            if controller.startswith('history_'):
                _flat_identity(flats, (window, 51, 24), raw, rows[window])
        data['records'].append(record(output, controller))
    verify_development(data)
    study.save_json(ART / 'development.json', data)


def freeze():
    verify_development()
    study.save_json(ART / 'frozen.json', dict(frozen_at=study.utc(), source_sha256=study.source_hashes(), legacy=study.legacy_hashes(),
        models=study.models(), **study.frozen_budget(), development_sha256=study.sha(ART / 'development.json'),
        historical_references_sha256=study.sha(ART / 'historical_references.json'),
        v23_frozen_sha256=study.verify_references()['v23_frozen_sha256'],
        training_frozen_sha256=study.sha(ART / 'training_frozen.json'), trained_models_sha256=study.sha(ART / 'trained_models.json'),
        training_commands_sha256=study.sha(ART / 'training_commands.jsonl')))
    study.verify_frozen()


def prepare(device):
    study.verify_frozen()
    records = []
    for geometry, reset in study.HOLDOUTS:
        study.verify_frozen()
        label = f'prepare_geometry{geometry}'
        output = WORK / f'{label}.json'
        run(eval_command('parent', geometry, reset, 'prepare', output, device), label)
        study.verify_frozen()
        if read(output).get('scored_episodes') != 0:
            raise ValueError('initialization-only preparation scored episodes')
        records.append(record(output))
    study.save_json(ART / 'terrain_cache.json', dict(created_utc=study.utc(), preparation=records,
        cache=cache_snapshot(CACHE, [g for g, _ in study.HOLDOUTS])))
    verify_ledger()
    ledger = ART / 'commands.jsonl'
    raw = ledger.read_bytes()
    if len(raw.splitlines()) != 11:
        raise ValueError('preholdout command budget/order differs')
    study.save_json(ART / 'preholdout_ledger.json', dict(created_utc=study.utc(), path=str(ledger.relative_to(ROOT)),
        lines=11, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    study.save_json(ART / 'evaluation_inputs.json', dict(created_utc=study.utc(), experiment_freeze_sha256=study.sha(ART / 'frozen.json'),
        preholdout_ledger_sha256=study.sha(ART / 'preholdout_ledger.json'), terrain_cache_manifest_sha256=study.sha(ART / 'terrain_cache.json')))
    study.evaluation_inputs()


def evaluate(device):
    cache = read(ART / 'terrain_cache.json')['cache']
    initial, flats = {}, {}
    for controller in study.CONTROLLERS:
        for geometry, reset in study.HOLDOUTS:
            for check in (0, 1):
                study.verify_frozen()
                study.evaluation_inputs()
                if cache_snapshot(CACHE, [g for g, _ in study.HOLDOUTS]) != cache:
                    raise ValueError('holdout cache changed')
                if check == 0:
                    label = f'{controller}__geometry{geometry}_reset{reset}'
                    output = ART / 'evaluations' / f'{label}.json'
                    run(eval_command(controller, geometry, reset, 'holdout', output, device), f'evaluations_{label}')
            bundle = read(output)
            rows = audit_bundle(bundle)
            for window in ('16', '64'):
                raw = bundle['windows'][window]
                _paired(initial, (geometry, reset), raw)
                if controller.startswith('history_'):
                    _flat_identity(flats, (window, geometry, reset), raw, rows[window])


def report():
    result = summarize(ART)
    study.save_json(ART / 'summary.json', result)
    with (ART / 'summary.md').open('x') as stream:
        stream.write(markdown(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('development', 'freeze', 'prepare', 'evaluate', 'report'))
    parser.add_argument('--device', choices=('cuda:1',), default='cuda:1')
    args = parser.parse_args()
    for path in (ART, WORK, ART / 'evaluations'):
        path.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'outputs/hybrid_v11_20260922/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'development': lambda: development(args.device), 'freeze': freeze, 'prepare': lambda: prepare(args.device),
         'evaluate': lambda: evaluate(args.device), 'report': report}[args.phase]()


if __name__ == '__main__':
    main()
