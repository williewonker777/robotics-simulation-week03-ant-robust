from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_teammate_v25 import gate, flat_identity

def rows(six=False,speed=1,fall=False):
    return [dict(controller='x',geometry=131,reset=111,env=i,seconds=16,family=f,level=0,distance_m=54 if six else 14,steps=960,one=True,six=six,fall=fall,lane=False,world=False,survival=True,safe_survival=True,speed_m_s=speed,duty=0,switches=0) for i,f in enumerate(('rough','flat'))]

class SummaryTests(unittest.TestCase):
    def test_strict_gain_needed(self):
        self.assertEqual(gate(rows(),rows())['verdict'],'FAIL')
        self.assertEqual(gate(rows(True),rows())['verdict'],'PASS')
    def test_flat_speed_retention(self):
        self.assertEqual(gate(rows(True,.9),rows())['verdict'],'FAIL')
    def test_rough_falls_retention(self):
        self.assertEqual(gate(rows(True,fall=True),rows())['verdict'],'FAIL')
    def test_history_exact_flat(self):
        a,b=rows(True),rows()
        a[1]=b[1].copy()
        self.assertTrue(flat_identity(a,b))
        self.assertEqual(gate(a,b,True,b)['verdict'],'PASS')
        a[1]['duty']=.1
        self.assertEqual(gate(a,b,True,b)['verdict'],'FAIL')
    def test_world_fails(self):
        a=rows(True);a[0]['world']=True
        self.assertEqual(gate(a,rows())['verdict'],'FAIL')

if __name__=='__main__': unittest.main()


def bundles(phase='holdout'):
    from summarize_teammate_v25 import matrix, SCHEMA, BASE_SHA
    records=[]
    frozen={'source_sha256':{'source':'hash'},'models':{}}
    for c,g,r,n in matrix(phase):
        mode='hybrid' if c.startswith('history_') else 'v5' if c=='v5' else 'v10'
        model={'sha256':c.removeprefix('history_'),'mode':mode,'command_mode':None if c=='v5' else 'conditioned'}
        frozen['models'][c]=model
        b=dict(schema=SCHEMA,task='Week03-Ant-Contact-v16-Eval-v0',controller=c,geometry_seed=g,reset_seed=r,num_envs=n,scored_episodes=n,seconds=64,scenario='mixed',dt=1/60,new_training_transitions=0,teacher_tensor_identity_verified=True,success_thresholds_m={'one_tile':13.1,'all_tiles':53.1},source_sha256={'old':'hash'},source_sha256_after={'old':'hash'},checkpoint_sha256=model['sha256'],checkpoint_sha256_after=model['sha256'],mode=mode,command_mode=model['command_mode'],initial_state_sha256={'root_state':'x','joint_pos':'x','joint_vel':'x','observations':'x'},initial_prefix_sha256='x',initial_rng_sha256={'cpu':'x','cuda':'x'},terrain_config_sha256=str(g),family_names=['a','b','c','d','e','f','flat'],difficulties=[0,.25,.5,.75,1],family_indices=[i//5%7 for i in range(n)],level_indices=[i%5 for i in range(n)],windows={},v25={'phase':phase,'model_phase':phase,'base_evaluator_sha256':BASE_SHA,'substitutions':['CONTROLLERS','SCHEMA','resolve_model'],'adapter_source_sha256':frozen['source_sha256'],'adapter_source_sha256_after':frozen['source_sha256']})
        for seconds in (16,64):
            b['windows'][str(seconds)]=dict(window_seconds=seconds,forward_distance=[14.]*n,episode_terminated=[False]*n,episode_out_of_lane=[False]*n,episode_world_exit=[False]*n,episode_steps=[seconds*60]*n,routing={'episode_v10_duty':[0. if c.startswith('history_') or c=='v5' else 1.]*n,'episode_switch_count':[0]*n})
        records.append(b)
    return records,frozen

class MatrixSummaryTests(unittest.TestCase):
    def test_totals_and_seed_gate(self):
        from summarize_teammate_v25 import summarize
        records,frozen=bundles()
        # Seed61/62 improve both maps; seed63 improves one map but not the other.
        for b in records:
            if b['controller'] in ('combined61','combined62','combined63') and not (b['controller']=='combined63' and b['geometry_seed']==132):
                for w in b['windows'].values(): w['forward_distance'][0]=54.
        result=summarize(records,frozen)
        self.assertEqual(result['physical_first_episodes'],8050)
        verdict=result['windows']['16']['contrasts']['combined_vs_control']
        self.assertEqual(verdict['seed_passes'],2)
        self.assertEqual(verdict['verdict'],'FAIL')
        self.assertEqual(verdict['seeds']['63']['scopes']['pooled']['verdict'],'PASS')
        self.assertEqual(result['windows']['64']['controllers']['v5']['total']['rough']['episodes'],300)
    def test_development_parity_and_pairing(self):
        from summarize_teammate_v25 import summarize
        records,frozen=bundles('development')
        self.assertEqual(summarize(records,frozen,'development')['verdict'],'PASS')
        records[2]['windows']['16']['forward_distance'][0]+=1
        with self.assertRaisesRegex(ValueError,'parity'): summarize(records,frozen,'development')
    def test_missing_and_rng_substitution(self):
        from summarize_teammate_v25 import summarize
        records,frozen=bundles()
        with self.assertRaisesRegex(ValueError,'matrix'): summarize(records[:-1],frozen)
        records[0]['initial_rng_sha256']['cpu']='bad'
        with self.assertRaisesRegex(ValueError,'pairing'): summarize(records,frozen)
