# SPDX-License-Identifier: BSD-3-Clause
"""Sequential exact-matrix v25 evaluation, with explicit freeze and no overwrite."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from evaluate_teammate_v25 import CONTROLLERS, DEVELOPMENT, ROOT, controller_binding, sha

BASE_SOURCES = ('scripts/evaluate_unseen_terrain.py', 'src/week03_ant/tasks/contact_v16_cfg.py',
    'src/week03_ant/tasks/rough_v5_cfg.py', 'src/week03_ant/tasks/lanes.py',
    'src/week03_ant/history_gate.py', 'src/week03_ant/paired_horizon_v23.py')

def matrix(phase):
    if phase not in ('development','holdout'):
        raise ValueError('undeclared evaluation phase')
    controllers, pairs, n = (DEVELOPMENT, ((51, 24),), 35) if phase == 'development' else (CONTROLLERS, ((131,111),(132,112)),175)
    return [(c,g,r,n) for g,r in pairs for c in controllers]

def freeze(phase, destination, cache_proof=None):
    from week03_ant.teammate_study_v25 import source_hashes
    models = {}
    for controller, _, _, _ in matrix(phase):
        if controller not in models:
            path, digest, mode, access = controller_binding(controller, phase)
            models[controller] = dict(checkpoint=str(path.relative_to(ROOT)), sha256=digest, mode=mode, command_mode=access)
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), v5_sha256='889836488fc220963b35acecbb59a1f9a5d371732cf8cfddf08dc700bbca605e', phase=phase, models=models, source_sha256=source_hashes(),
        base_source_sha256={p:sha(ROOT/p) for p in BASE_SOURCES}, matrix=matrix(phase))
    if cache_proof is not None:
        proof=json.loads(cache_proof.read_text())
        if proof['phase'] != phase or proof['scored_episodes'] != 0 or proof['source_sha256'] != record['source_sha256'] or proof['models'] != models:
            raise ValueError('cache preparation binding differs')
        record['cache_preparation']=dict(path=str(cache_proof.resolve().relative_to(ROOT)),sha256=sha(cache_proof))
        record['cache_snapshot']=proof['cache_snapshot']
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        json.dump(record, stream, indent=2)
    return record

def verify(record):
    from week03_ant.teammate_study_v25 import source_hashes
    if 'cache_preparation' in record:
        from week03_ant.posture_study import cache_snapshot
        proof=record['cache_preparation']
        prepared=json.loads((ROOT/proof['path']).read_text())
        linked={**prepared['evidence_sha256'],prepared['preparation_freeze']['path']:prepared['preparation_freeze']['sha256']}
        if any(sha(ROOT/path) != digest for path,digest in linked.items()):
            raise ValueError('cache preparation evidence changed')
        if sha(ROOT/proof['path']) != proof['sha256'] or cache_snapshot('/tmp/isaaclab/terrains', sorted({g for _,g,_,_ in matrix(record['phase'])})) != record['cache_snapshot']:
            raise ValueError('frozen evaluation cache changed')
    if {p:sha(ROOT/p) for p in BASE_SOURCES} != record['base_source_sha256']:
        raise ValueError('frozen original sources changed')
    if source_hashes() != record['source_sha256']:
        raise ValueError('frozen source changed')
    for controller, entry in record['models'].items():
        path, digest, mode, access = controller_binding(controller, record['phase'])
        if entry != dict(checkpoint=str(path.relative_to(ROOT)),sha256=digest,mode=mode,command_mode=access) or sha(path) != digest:
            raise ValueError('frozen model changed')

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('development','holdout'), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--cache-proof', type=Path)
    p.add_argument('--device', default='cuda:1')
    a = p.parse_args(argv)
    if a.output_dir.exists():
        p.error('output directory exists; refusing overwrite')
    if not a.prepare and a.cache_proof is None:
        p.error('scored evaluation requires --cache-proof from zero-step preparation')
    record = json.loads(a.freeze.read_text()) if a.freeze.exists() else freeze(a.phase,a.freeze,a.cache_proof)
    if not a.prepare and ('cache_preparation' not in record or record['cache_preparation']['sha256'] != sha(a.cache_proof)):
        raise ValueError('scored freeze lacks exact cache preparation')
    if record['phase'] != a.phase or record['matrix'] != [list(x) for x in matrix(a.phase)]:
        # Newly created Python tuples compare differently from JSON lists.
        if record['phase'] != a.phase or [tuple(x) for x in record['matrix']] != matrix(a.phase):
            raise ValueError('frozen matrix mismatch')
    verify(record)
    a.output_dir.mkdir(parents=True)
    jobs=matrix(a.phase)
    if a.prepare:
        jobs=[next(cell for cell in jobs if cell[1]==g) for g in sorted({cell[1] for cell in jobs})]
    for c,g,r,n in jobs:
        verify(record)
        command = [sys.executable,str(ROOT/'scripts/evaluate_teammate_v25.py'),'--phase',a.phase,'--controller',c,'--geometry',str(g),'--seed',str(r),'--num-envs',str(n),'--seconds','64','--headless','--device',a.device,'--output',str(a.output_dir/f'{c}_g{g}_r{r}.json')]
        if a.prepare:
            command.append('--prepare')
        name=f'{c}_g{g}_r{r}'
        logdir=a.output_dir/'_execution'
        logdir.mkdir(exist_ok=True)
        job=dict(command=command,cwd=str(ROOT),phase=a.phase,prepare=a.prepare,
            freeze_path=str(a.freeze.resolve()),freeze_sha256=sha(a.freeze),
            started_utc=datetime.now(timezone.utc).isoformat(),
            stdout=str(logdir/f'{name}.stdout.log'),stderr=str(logdir/f'{name}.stderr.log'))
        with (logdir/f'{name}.command.json').open('x') as stream:
            json.dump(job,stream,indent=2)
        with (logdir/f'{name}.stdout.log').open('x') as stdout, (logdir/f'{name}.stderr.log').open('x') as stderr:
            completed=subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        outcome=dict(returncode=completed.returncode,finished_utc=datetime.now(timezone.utc).isoformat(),
            stdout_sha256=sha(logdir/f'{name}.stdout.log'),stderr_sha256=sha(logdir/f'{name}.stderr.log'))
        with (logdir/f'{name}.exit.json').open('x') as stream:
            json.dump(outcome,stream,indent=2)
        completed.check_returncode()
        verify(record)
    if a.prepare:
        from week03_ant.posture_study import cache_snapshot
        outputs=[a.output_dir/f'{c}_g{g}_r{r}.json' for c,g,r,n in jobs]
        for output in outputs:
            raw=json.loads(output.read_text())
            original=output.parent/'_adapter_raw'/output.name
            enriched=dict(raw)
            del enriched['v25']
            if sha(original) != raw['v25']['original_raw_sha256'] or json.loads(original.read_text()) != enriched:
                raise ValueError('cache adapter changed original numerical evidence')
            if raw['scored_episodes'] != 0 or raw['physics_steps'] != 0 or raw['windows']:
                raise ValueError('cache preparation performed scored steps')
        proof=dict(schema='week03_ant_teammate_v25_cache_v1',phase=a.phase,scored_episodes=0,
            source_sha256=record['source_sha256'],models=record['models'],
            preparation_freeze=dict(path=str(a.freeze.resolve().relative_to(ROOT)),sha256=sha(a.freeze)),
            evidence_sha256={str(path.resolve().relative_to(ROOT)):sha(path) for path in [*outputs,*(a.output_dir/'_adapter_raw'/p.name for p in outputs)]},
            cache_snapshot=cache_snapshot('/tmp/isaaclab/terrains',sorted({g for _,g,_,_ in jobs})))
        with (a.output_dir/'_execution/cache_preparation.json').open('x') as stream:
            json.dump(proof,stream,indent=2)

if __name__ == '__main__':
    main()
