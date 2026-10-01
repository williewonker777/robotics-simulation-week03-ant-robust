# SPDX-License-Identifier: BSD-3-Clause
"""Strict paired, seed-preserving v25 summary; 64s never rescues 16s."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from evaluate_teammate_v25 import CONTROLLERS, DEVELOPMENT, RUNS, SCHEMA, BASE_SHA, ROOT, sha
from run_teammate_eval_v25 import matrix
from summarize_unseen_terrain import episode_rows, groups, aggregate, paired_comparison, require, PAIR_FIELDS


def validate(records, frozen, phase='holdout'):
    cells = matrix(phase)
    indexed = {}
    for b in records:
        key = (b['controller'], b['geometry_seed'], b['reset_seed'])
        require(key not in indexed, 'duplicate cell')
        indexed[key] = b
        n = 35 if phase == 'development' else 175
        require(b['schema'] == SCHEMA and b['task'] == 'Week03-Ant-Contact-v16-Eval-v0', 'wrong evaluator')
        require(b['num_envs'] == n and b['scored_episodes'] == n and b['seconds'] == 64 and b['scenario'] == 'mixed' and b['dt'] == 1/60, 'physical protocol mismatch')
        require(b['new_training_transitions'] == 0 and b['teacher_tensor_identity_verified'] is True, 'training/teacher mismatch')
        require(b['success_thresholds_m'] == {'one_tile':13.1,'all_tiles':53.1}, 'threshold mismatch')
        require(set(b['windows']) == {'16','64'}, 'windows mismatch')
        require(b['source_sha256'] == b['source_sha256_after'] and b['checkpoint_sha256'] == b['checkpoint_sha256_after'], 'posthash mismatch')
        meta = b['v25']
        require(meta['phase'] == phase and meta['model_phase'] == phase and meta['base_evaluator_sha256'] == BASE_SHA and meta['substitutions'] == ['CONTROLLERS','SCHEMA','resolve_model'], 'adapter mismatch')
        require(meta['adapter_source_sha256'] == meta['adapter_source_sha256_after'] == frozen['source_sha256'], 'adapter source mismatch')
        model = frozen['models'][b['controller']]
        require(b['checkpoint_sha256'] == model['sha256'] and b['mode'] == model['mode'] and b['command_mode'] == model['command_mode'], 'frozen model mismatch')
        require(len(b['family_indices']) == n and len(b['level_indices']) == n, 'assignment lengths')
        require(len(b['family_names']) == 7 and len(set(b['family_names'])) == 7 and 'flat' in b['family_names'], 'family inventory')
        require(all(type(x) is int and 0 <= x < 7 for x in b['family_indices']) and all(type(x) is int and 0 <= x < 5 for x in b['level_indices']), 'assignment values')
        for family in range(7):
            for level in range(5):
                require(sum(f==family and l==level for f,l in zip(b['family_indices'],b['level_indices'])) == n//35, 'cell denominator')
        for seconds in (16,64):
            require(b['windows'][str(seconds)]['window_seconds'] == seconds, 'window label')
            episode_rows(b,seconds)
        a,z = b['windows']['16'],b['windows']['64']
        for i,steps in enumerate(z['episode_steps']):
            require(a['episode_steps'][i] == min(960,steps), 'dependent lengths')
            if steps <= 960:
                require(all(a[k][i] == z[k][i] for k in ('forward_distance','episode_terminated','episode_out_of_lane','episode_world_exit')), 'terminal window divergence')
    require(set(indexed) == {(c,g,r) for c,g,r,_ in cells}, 'incomplete matrix')
    for g,r in sorted({(g,r) for _,g,r,_ in cells}):
        reference = indexed[(cells[0][0],g,r)]
        for c,gg,rr,_ in cells:
            if (gg,rr) == (g,r):
                require(all(indexed[(c,g,r)][k] == reference[k] for k in PAIR_FIELDS), 'initial/config/RNG pairing mismatch')
    if phase == 'development':
        for c in DEVELOPMENT[2:]:
            parent = 'history_control' if c.startswith('history_') else 'v16_control'
            require(indexed[(c,51,24)]['windows'] == indexed[(parent,51,24)]['windows'], 'development per-environment parity failed')
    return indexed


def flat_identity(candidate, v5):
    fields = ('distance_m','steps','one','six','fall','lane','world','survival','safe_survival','speed_m_s')
    reference = {(r['geometry'],r['reset'],r['env']):r for r in v5 if r['family']=='flat'}
    flat = [r for r in candidate if r['family']=='flat']
    return bool(flat) and all(r.get('flat_raw_identity', True) and r['duty']==0 and r['switches']==0 and all(r[k] == reference[(r['geometry'],r['reset'],r['env'])][k] for k in fields) for r in flat)


def gate(candidate, baseline, history=False, v5=None):
    a,b = groups(candidate),groups(baseline)
    rough,old = a['rough'],b['rough']
    checks = {**{f'rough_{k}_retained':rough[k]>=old[k] for k in ('one','six')},
        **{f'rough_{k}_retained':rough[k]<=old[k] for k in ('fall','lane')},
        'world_zero':a['rough']['world']==a['flat']['world']==0,
        'flat_falls_retained':a['flat']['fall']<=b['flat']['fall'],
        'flat_lanes_retained':a['flat']['lane']<=b['flat']['lane'],
        'strict_rough_improvement':rough['one']>old['one'] or rough['six']>old['six']}
    if history:
        checks['flat_v5_identity'] = flat_identity(candidate,v5) and flat_identity(baseline,v5)
    else:
        checks['flat_speed_retained'] = a['flat']['mean_speed_m_s'] >= b['flat']['mean_speed_m_s']
    return dict(verdict='PASS' if all(checks.values()) else 'FAIL',checks=checks,
                candidate=a,baseline=b,paired=paired_comparison(candidate,baseline))


def summarize(records, frozen, phase='holdout'):
    indexed = validate(records,frozen,phase)
    if phase == 'development':
        return dict(schema='week03_ant_teammate_v25_development_v1',verdict='PASS',physical_first_episodes=280,dependent_window_observations=560,exact_per_environment_parity=True)
    result = dict(schema='week03_ant_teammate_v25_summary_v1',physical_first_episodes=8050,
        dependent_window_observations=16100,paired_initial_conditions=350,statistical_independence_claim=False,
        primary_seconds=16,diagnostic_seconds=64,legacy_references_replicated=False,windows={})
    for seconds in (16,64):
        rows = [r for b in indexed.values() for r in episode_rows(b,seconds)]
        for row in rows:
            if row['family'] == 'flat':
                c,g,r,i = row['controller'],row['geometry'],row['reset'],row['env']
                window=indexed[(c,g,r)]['windows'][str(seconds)]
                reference=indexed[('v5',g,r)]['windows'][str(seconds)]
                physical=all(value[i] == reference[key][i] for key,value in window.items() if isinstance(value,list))
                routing=all(value[i] == reference['routing'][key][i] for key,value in window['routing'].items()
                    if key != 'episode_mean_action_disagreement_rms')
                row['flat_raw_identity']=physical and routing
        selected = {c:[r for r in rows if r['controller']==c] for c in CONTROLLERS}
        controllers = {}
        for c,rr in selected.items():
            cells=[]
            for g in (131,132):
                for family in sorted({r['family'] for r in rr}):
                    for level in range(5):
                        cell=[r for r in rr if r['geometry']==g and r['family']==family and r['level']==level]
                        cells.append(dict(geometry=g,family=family,level=level,**aggregate(cell)))
            controllers[c]=dict(total=groups(rr),
                by_map={str(g):groups([r for r in rr if r['geometry']==g]) for g in (131,132)},
                by_family={f:aggregate([r for r in rr if r['family']==f]) for f in sorted({r['family'] for r in rr})},
                by_level={str(l):groups([r for r in rr if r['level']==l]) for l in range(5)},
                by_map_family_level=cells)
        contrasts={}
        for history in (False,True):
            prefix='history_' if history else ''
            for arm,base in (('recovery','control'),('combined','recovery'),('combined','control')):
                seeds={}
                for seed in (61,62,63):
                    a,b=selected[f'{prefix}{arm}{seed}'],selected[f'{prefix}{base}{seed}']
                    tests={str(g):gate([r for r in a if r['geometry']==g],[r for r in b if r['geometry']==g],history,selected['v5']) for g in (131,132)}
                    tests['pooled']=gate(a,b,history,selected['v5'])
                    paired_cells=[]
                    for g in (131,132):
                        for level in range(5):
                            for family in sorted({r['family'] for r in a}):
                                aa=[r for r in a if r['geometry']==g and r['level']==level and r['family']==family]
                                bb=[r for r in b if r['geometry']==g and r['level']==level and r['family']==family]
                                paired_cells.append(dict(geometry=g,level=level,family=family,
                                    candidate=aggregate(aa),baseline=aggregate(bb),
                                    changes={k:dict(false_to_true=sum(x[k] and not y[k] for x,y in zip(aa,bb)),
                                        true_to_false=sum(y[k] and not x[k] for x,y in zip(aa,bb)))
                                        for k in ('one','six','fall','lane','world','survival')}))
                    seeds[str(seed)]=dict(verdict='PASS' if all(x['verdict']=='PASS' for x in tests.values()) else 'FAIL',scopes=tests,paired_map_family_level=paired_cells)
                contrasts[f'{prefix}{arm}_vs_{base}']=dict(verdict='PASS' if all(x['verdict']=='PASS' for x in seeds.values()) else 'FAIL',seed_passes=sum(x['verdict']=='PASS' for x in seeds.values()),seeds=seeds,
                    pooled_descriptive=dict(candidate=groups([r for seed in (61,62,63) for r in selected[f'{prefix}{arm}{seed}']]),
                        baseline=groups([r for seed in (61,62,63) for r in selected[f'{prefix}{base}{seed}']])))
        references={f'{c}_vs_{ref}':paired_comparison(selected[c],selected[ref]) for c in CONTROLLERS if c not in ('v5','v16_control','history_control','high53','history_high53') for ref in ('v5','v16_control','history_control','high53','history_high53')}
        result['windows'][str(seconds)]=dict(controllers=controllers,contrasts=contrasts,legacy_descriptive_comparisons=references,episode_rows=rows)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('development','holdout'),default='holdout')
    a=p.parse_args(argv)
    paths=sorted(a.input_dir.glob('*.json'))
    records=[json.loads(f.read_text()) for f in paths]
    frozen=json.loads(a.freeze.read_text())
    from audit_teammate_v25_raw import audit_evaluation_records, compare_summary
    audited=audit_evaluation_records(records, frozen)
    result=summarize(records,frozen,a.phase)
    if a.phase == 'holdout':
        compare_summary(result,audited)
    evidence_paths=[a.freeze,*paths,*(a.input_dir/'_adapter_raw'/f.name for f in paths)]
    if 'cache_preparation' in frozen:
        cache_path=ROOT/frozen['cache_preparation']['path']
        require(sha(cache_path)==frozen['cache_preparation']['sha256'],'cache proof changed')
        evidence_paths.append(cache_path)
        cache=json.loads(cache_path.read_text())
        evidence_paths.extend(ROOT/path for path in cache['evidence_sha256'])
        evidence_paths.append(ROOT/cache['preparation_freeze']['path'])
    result['evidence_sha256']={str(path.resolve().relative_to(ROOT)):sha(path) for path in evidence_paths}
    result['freeze_path']=str(a.freeze.resolve().relative_to(ROOT))
    result['freeze_sha256']=sha(a.freeze)
    result['independent_raw_audit']=audited
    for path,bundle in zip(paths,records):
        original=a.input_dir/'_adapter_raw'/path.name
        require(sha(original)==bundle['v25']['original_raw_sha256'], 'original raw digest mismatch')
        enriched=dict(bundle)
        del enriched['v25']
        require(json.loads(original.read_text())==enriched,'adapter altered original numerical evidence')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')

if __name__=='__main__':
    main()
