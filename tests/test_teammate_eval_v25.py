import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
import json
import subprocess
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import evaluate_teammate_v25 as ev
import run_teammate_eval_v25 as driver

class EvalTests(unittest.TestCase):
    def test_exact_matrix(self):
        self.assertEqual(len(driver.matrix('holdout')),46)
        self.assertEqual(sum(n for *_,n in driver.matrix('holdout')),8050)
        self.assertEqual(sum(n for *_,n in driver.matrix('development')),280)
        self.assertEqual(len(ev.CONTROLLERS),23)
        self.assertEqual(len(set(ev.CONTROLLERS)),23)
    def test_base_pin(self):
        self.assertEqual(ev.sha(ev.BASE),ev.BASE_SHA)
        base=ev.load_base()
        self.assertEqual(base.CONTROLLERS,ev.LEGACY)
        self.assertEqual(base.SCHEMA,'week03_ant_unseen_layout_v1')
    def test_legacy_delegation(self):
        class Fake:
            @staticmethod
            def resolve_model(c): return ('p','digest',c,None)
        self.assertEqual(ev.controller_binding('high53','holdout',Fake),('p','digest','high53',None))
        with self.assertRaises(ValueError): ev.controller_binding('high53','development',Fake)
        with self.assertRaises(ValueError): ev.controller_binding('best','holdout',Fake)
    def test_prepare_zero_no_output_overwrite(self):
        base=ev.load_base()
        a=base.parse_args(['--controller','v5','--geometry','131','--seed','111','--output','/tmp/nonexistent-v25-test-output-238920.json','--prepare'])
        self.assertTrue(a.prepare)

    def test_new_model_binding_and_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model_249.pt'
            path.write_bytes(b'frozen model')
            calls=[]
            def entry(run,phase):
                calls.append((run,phase))
                return {'checkpoint':str(path),'sha256':ev.sha(path)}
            with patch.dict(sys.modules,{'week03_ant.teammate_study_v25':SimpleNamespace(model_entry=entry)}):
                result=ev.controller_binding('history_combined61','development')
                self.assertEqual(result[2:],('hybrid','conditioned'))
                self.assertEqual(calls,[('combined61','development')])
                with self.assertRaises(ValueError): ev.controller_binding('combined62','development')
            with patch.dict(sys.modules,{'week03_ant.teammate_study_v25':SimpleNamespace(model_entry=lambda *a:{'checkpoint':str(path),'sha256':'wrong'})}):
                with self.assertRaisesRegex(ValueError,'binding'): ev.controller_binding('combined61','holdout')
    def test_prepare_matrix_first_endpoint(self):
        for phase, count in (('development',1),('holdout',2)):
            jobs=driver.matrix(phase)
            prep=[next(cell for cell in jobs if cell[1]==g) for g in sorted({cell[1] for cell in jobs})]
            self.assertEqual(len(prep),count)
            self.assertTrue(all(cell[0]==jobs[0][0] for cell in prep))

    def test_real_driver_freeze_auditor_schema(self):
        import importlib.util
        from datetime import datetime, timedelta
        try:
            import pytest
        except ImportError:
            self.skipTest('full-array test fixture needs repository pytest environment')
        spec=importlib.util.spec_from_file_location('v25_fullraw_fixture',SCRIPTS.parent/'tests/test_teammate_audit_v25.py')
        fixture=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        import summarize_teammate_v25 as summary
        records,_=fixture.matrix()
        sources={'new.py':fixture.H}
        def binding(c,phase):
            return ev.ROOT/'synthetic'/c,fixture.audit.LEGACY_SHA.get(c,fixture.H),'hybrid' if c.startswith('history_') else 'v5' if c=='v5' else 'v10',None if c=='v5' else 'conditioned'
        with tempfile.TemporaryDirectory() as directory, patch.object(driver,'controller_binding',side_effect=binding), patch.dict(sys.modules,{'week03_ant.teammate_study_v25':SimpleNamespace(source_hashes=lambda:sources)}):
            frozen=driver.freeze('holdout',Path(directory)/'freeze.json')
        for b in records:
            b['source_sha256']={k:v for k,v in frozen['base_source_sha256'].items() if k!='scripts/evaluate_unseen_terrain.py'}
            b['source_sha256_after']=dict(b['source_sha256'])
            b['v25']['base_evaluator_sha256']=ev.BASE_SHA
            start=datetime.fromisoformat(frozen['created_utc'])+timedelta(seconds=1)
            b['started_utc']=start.isoformat()
            b['finished_utc']=(start+timedelta(seconds=1)).isoformat()
        canonical=fixture.audit.audit_evaluation_records(records,frozen)
        result=summary.summarize(records,frozen)
        fixture.audit.compare_summary(result,canonical)
        self.assertEqual(result['physical_first_episodes'],8050)

    def test_actual_child_process_exit_still_enriches(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'enriched.json'
            raw=output.parent/'_adapter_raw'/output.name
            model=Path(directory)/'model.pt'
            model.write_bytes(b'pinned')
            payload={'schema':ev.SCHEMA,'windows':{},'scored_episodes':0,'physics_steps':0}
            code=f"import os,pathlib;pathlib.Path({str(raw)!r}).write_text({json.dumps(payload)!r});os._exit(0)"
            command=[sys.executable,'-c',code]
            arguments=['--phase','development','--controller','v16_control','--geometry','51','--seed','24','--num-envs','35','--seconds','64','--prepare','--output',str(output)]
            with patch.dict(sys.modules,{'week03_ant.teammate_study_v25':SimpleNamespace(source_hashes=lambda:{'f':'hash'})}), patch.object(ev,'controller_binding',return_value=(model,ev.sha(model),'v10','conditioned')), patch.object(ev,'child_command',return_value=command):
                ev.main(arguments)
                original_bytes=raw.read_bytes()
                enriched=json.loads(output.read_text())
                metadata=enriched.pop('v25')
                self.assertEqual(enriched,payload)
                self.assertEqual(metadata['original_raw_sha256'],ev.sha(raw))
                self.assertEqual(metadata['numerical_child_command'],command)
                self.assertEqual(metadata['numerical_child_returncode'],0)
                self.assertEqual(json.loads(raw.with_suffix('.child_command.json').read_text())['command'],command)
                with self.assertRaises(SystemExit): ev.main(arguments)
                self.assertEqual(raw.read_bytes(),original_bytes)
    def test_actual_child_zero_exit_missing_raw_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'without original raw'):
                ev.run_numerical_child([sys.executable,'-c','import os;os._exit(0)'],Path(directory)/'missing.json')
    def test_actual_child_nonzero_exit_rejected_even_with_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            raw=Path(directory)/'raw.json'
            code=f"import os,pathlib;pathlib.Path({str(raw)!r}).write_text('{{}}');os._exit(7)"
            with self.assertRaises(subprocess.CalledProcessError):
                ev.run_numerical_child([sys.executable,'-c',code],raw)
    def test_child_command_explicit_mode(self):
        command=ev.child_command('development',['--prepare'])
        self.assertEqual(command,[sys.executable,str(ev.Path(ev.__file__).resolve()),'--numerical-child','--phase','development','--prepare'])

    def test_actual_child_model_mutation_blocks_enrichment(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'enriched.json'
            raw=output.parent/'_adapter_raw'/output.name
            model=Path(directory)/'model.pt'
            model.write_bytes(b'pinned')
            digest=ev.sha(model)
            code=f"import os,pathlib;pathlib.Path({str(raw)!r}).write_text('{{}}');pathlib.Path({str(model)!r}).write_bytes(b'changed');os._exit(0)"
            with patch.dict(sys.modules,{'week03_ant.teammate_study_v25':SimpleNamespace(source_hashes=lambda:{'f':'hash'})}), patch.object(ev,'controller_binding',return_value=(model,digest,'v10','conditioned')), patch.object(ev,'child_command',return_value=[sys.executable,'-c',code]):
                with self.assertRaisesRegex(ValueError,'inputs changed'):
                    ev.main(['--phase','development','--controller','v16_control','--geometry','51','--seed','24','--num-envs','35','--seconds','64','--prepare','--output',str(output)])
            self.assertFalse(output.exists())
            self.assertTrue(raw.exists())

if __name__=='__main__': unittest.main()
