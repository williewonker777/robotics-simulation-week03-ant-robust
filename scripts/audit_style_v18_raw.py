"""Standalone post-score v18 first-episode audit; no project scoring imports."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
import struct

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/terrain_demo/terrain_style_v18'
SUMMARY = json.loads((ART / 'summary.json').read_text())
CONTROLLERS = ('v16_control', 'always', 'gated', 'history_original', 'history_always', 'history_gated')
MAPS = ((105, 66), (106, 67))
FLAT_FIELDS = (
    'episode_return', 'forward_distance', 'maximum_distance_m', 'episode_lengths',
    'episode_active_steps', 'episode_terminated', 'episode_out_of_lane', 'episode_world_exit',
    'episode_strict_one_tile_success', 'episode_strict_all_tiles_success',
    'episode_full_horizon_survival', 'first_hit_one_seconds', 'first_hit_six_seconds',
    'distance_at_snapshot_m', 'episode_v10_target_steps', 'episode_v10_duty',
    'episode_alpha_sum', 'episode_switch_count', 'episode_switch_steps',
    'episode_switch_to_v10', 'history_switch_events', 'episode_uncertain_steps',
    'episode_fall_within_switch_window', 'episode_max_action_jump_rms',
)

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def counts(rows):
    rough = [r for r in rows if r['family'] != 'flat']
    flat = [r for r in rows if r['family'] == 'flat']
    out = {'n': len(rough), 'flat_n': len(flat)}
    for k in ('one', 'six', 'falls', 'lane', 'world'):
        out[k] = sum(r[k] for r in rough)
    for k in ('falls', 'lane', 'world'):
        out['flat_' + k] = sum(r[k] for r in flat)
    out['flat_mean_episode_speed'] = mean(r['distance'] / (r['steps'] * r['dt']) for r in flat) if flat else None
    return out

def same_counts(actual, expected, label):
    for k,v in actual.items():
        if k == 'flat_lane':
            # Included in v18 groups as a separate flat integrity check.
            pass
        if v is None:
            assert expected.get(k) is None, (label, k, v, expected.get(k))
        elif isinstance(v, float):
            assert math.isclose(v, expected[k], rel_tol=1e-12, abs_tol=1e-12), (label, k, v, expected[k])
        else:
            assert v == expected[k], (label, k, v, expected[k])

inventory = []
sections = {}
for scenario, folder, envs, seconds in (('mixed','evaluations',175,16),('stones','horizon',10,64)):
    assert len(list((ART / folder).glob('*.json'))) == 12
    grouped = defaultdict(list)
    initial = {}
    flats = {}
    for controller in CONTROLLERS:
        for geometry, reset in MAPS:
            path = ART / folder / f'{controller}__geometry{geometry}_reset{reset}.json'
            data = json.loads(path.read_text())
            assert (data['controller'],data['scenario'],data['geometry_seed'],data['reset_seed'],data['num_envs'],data['condition']['seconds']) == (controller,scenario,geometry,reset,envs,seconds)
            assert len(data['family_indices']) == envs
            assert data['condition']['max_steps'] == seconds*60
            state = (data['initial_state_sha256'],data['initial_prefix_sha256'],data['initial_rng_sha256'])
            if (geometry,reset) in initial:
                assert state == initial[(geometry,reset)], ('initial', folder, controller, geometry)
            else:
                initial[(geometry,reset)] = state
            n = envs
            keys = ('forward_distance','episode_lengths','episode_active_steps','episode_terminated','episode_out_of_lane','episode_world_exit','episode_strict_one_tile_success','episode_strict_all_tiles_success','family_indices','level_indices')
            assert all(isinstance(data[k],list) and len(data[k])==n for k in keys)
            rows = []
            for i in range(n):
                family = data['family_names'][data['family_indices'][i]]
                level = data['level_indices'][i]
                distance = data['forward_distance'][i]
                steps = data['episode_lengths'][i]
                assert steps == data['episode_active_steps'][i] and 0 < steps <= seconds*60
                fall,lane,world = [data[k][i] for k in ('episode_terminated','episode_out_of_lane','episode_world_exit')]
                assert all(type(flag) is bool for flag in (fall,lane,world))
                safe = not (fall or lane or world)
                one = safe and distance >= f32(data['condition']['one_threshold'])
                six = safe and distance >= f32(data['condition']['six_threshold'])
                assert (one,six)==(data['episode_strict_one_tile_success'][i],data['episode_strict_all_tiles_success'][i])
                rows.append({'family':family,'level':level,'one':int(one),'six':int(six),'falls':int(fall),'lane':int(lane),'world':int(world),'distance':distance,'steps':steps,'dt':data['condition']['dt']})
            if scenario=='mixed':
                assert Counter((r['family'],r['level']) for r in rows)=={(family,level):5 for family in data['family_names'] for level in range(5)}
            else:
                assert Counter((r['family'],r['level']) for r in rows)=={('stepping_stones',4):10}
            if scenario=='mixed' and controller.startswith('history_'):
                indices = [i for i,r in enumerate(rows) if r['family']=='flat']
                flat_data = {'indices':indices, 'levels':[data['level_indices'][i] for i in indices]}
                for k in FLAT_FIELDS:
                    assert len(data[k]) == n
                    flat_data[k] = [data[k][i] for i in indices]
                key=(geometry,reset)
                if key in flats:
                    assert flats[key] == flat_data, ('flat identity',controller,geometry)
                else:
                    flats[key]=flat_data
            grouped[controller].extend(rows)
            inventory.append({'path':str(path.relative_to(ART)),'sha256':digest(path),'episodes':n})
    section = SUMMARY[folder]
    independent = {}
    for controller,rows in grouped.items():
        overall = counts(rows)
        same_counts(overall,section['groups'][controller],(folder,controller))
        for family in sorted(set(r['family'] for r in rows)):
            same_counts(counts([r for r in rows if r['family']==family]),section['family'][controller][family],(folder,controller,family))
        for family,level in sorted(set((r['family'],r['level']) for r in rows)):
            key=f'{family}/level{level}'
            same_counts(counts([r for r in rows if r['family']==family and r['level']==level]),section['family_level'][controller][key],(folder,controller,key))
        independent[controller] = {k:overall[k] for k in ('n','flat_n','one','six','falls','lane','world','flat_falls','flat_lane','flat_world','flat_mean_episode_speed')}
    assert len(flats)==(2 if scenario=='mixed' else 0)
    assert section['actor_gated_vs_always']['passed'] is False
    assert section['hybrid_gated_vs_always']['passed'] is False
    sections[folder]={'groups':independent,'exact_initial_state_prefix_rng_pairs':len(initial),'history_flat_raw_identity_maps':len(flats)}
assert len(inventory)==24 and sum(x['episodes'] for x in inventory)==2220
assert {(x['path'],x['sha256'],x['episodes']) for x in inventory} == {(x['path'],x['sha256'],x['episodes']) for x in SUMMARY['files']}
audit={'passed':True,'independent_method':'raw first-episode arrays; f32 strict thresholds; no project scorer imports','summary_sha256':digest(ART/'summary.json'),'files':inventory,'total_first_episodes':2220,'sections':sections,'primary_actor_promotion':False,'primary_hybrid_promotion':False,'secondary_actor_promotion':False,'secondary_hybrid_promotion':False,'limitations':'Read-only numeric/provenance cross-check, not simulator replay; one training seed and two fresh maps.'}
evidence = ART / 'independent_raw_audit.json'
if evidence.exists():
    assert json.loads(evidence.read_text()) == audit, 'saved independent audit differs'
else:
    with evidence.open('x') as fp:
        json.dump(audit,fp,ensure_ascii=False,indent=2)
        fp.write('\n')
print('PASS: 24 files / 2220 first episodes; all controller, family, level counts, flat speeds, paired initials, hybrid flat identity, raw SHA match summary; four promotions FAIL')
