# SPDX-License-Identifier: BSD-3-Clause
"""Presentation plumbing tests; synthetic fixtures are not experiment evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_teammate_publication_v25.py'
spec = importlib.util.spec_from_file_location('public_verify', SCRIPT)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def fixture(root):
    """Stub the separately tested frozen auditor; exercise all public hash links."""
    entries = {}
    def put(name, value, original=None, projected=False):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value, str) else value)
        digest = v.sha(path)
        entries[name] = dict(path=name, original_sha256=original or digest,
                             published_sha256=digest, metadata_strings_only=projected)
        return digest
    controllers = ['c' + str(i) for i in range(23)]
    auditor = "CONTROLLERS=" + repr(controllers) + "\nMAPS=((131,111),(132,112))\n"
    auditor += "def audit_evaluation_records(records, frozen):\n assert sum([1e16,1.,-1e16])==0.0\n assert sum.__name__=='historical_sum'\n assert len(records)==46\n return {'physical_first_episodes':8050,'dependent_window_observations':16100}\n"
    auditor += "def compare_summary(summary, canonical):\n assert summary['physical_first_episodes']==canonical['physical_first_episodes']\n"
    sources = {'scripts/audit_teammate_v25_raw.py': put('scripts/audit_teammate_v25_raw.py', auditor)}
    sources.update({f'sources/{i}.py': put(f'sources/{i}.py', '# immutable') for i in range(19)})
    base = {f'base/{i}.py': put(f'base/{i}.py', '# immutable') for i in range(6)}
    model = 'models/final.pt'
    model_sha = put(model, 'not unpickled')
    frozen = dict(phase='holdout', source_sha256=sources, base_source_sha256=base,
                  models={c:dict(checkpoint=model, sha256=model_sha) for c in controllers})
    freeze_name = v.ART + '/frozen.json'
    original_freeze = 'a' * 64
    put(freeze_name, frozen, original_freeze, True)
    for c in controllers:
        for g,r in ((131,111),(132,112)):
            name = v.ART + f'/evaluations/{c}_g{g}_r{r}.json'
            raw_name = v.ART + f'/evaluations/_adapter_raw/{c}_g{g}_r{r}.json'
            numerical = dict(controller=c, geometry_seed=g, reset_seed=r, checkpoint=model, windows={'16': [0.0,False,None]})
            original_raw = 'b' * 64
            put(raw_name, numerical, original_raw, True)
            enriched = dict(numerical, v25={'original_raw_sha256': original_raw})
            put(name, enriched, 'c' * 64, True)
    summary = dict(freeze_path=freeze_name,freeze_sha256=original_freeze,physical_first_episodes=8050,
                   evidence_sha256={name:item['original_sha256'] for name,item in entries.items()
                                     if name == freeze_name or '/evaluations/' in name})
    put(v.ART + '/summary.json', summary, 'd' * 64, True)
    manifest = dict(schema=v.SCHEMA,science_hash_semantics=v.SEMANTICS,
                    files=list(entries.values()),excludes=['outputs/run.stdout.log'])
    (root / v.ART / 'publication_metadata.json').write_text(json.dumps(manifest))
    return manifest


class PublicationTests(unittest.TestCase):
    def test_stdlib_cli_and_original_hash_domains(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture(root)
            result=subprocess.run([sys.executable,'-S',str(SCRIPT),'--root',str(root)],capture_output=True,text=True,check=True)
            output=json.loads(result.stdout)
            self.assertEqual(output['verdict'],'PASS')
            self.assertEqual(output['arithmetic_policy'],v.ARITHMETIC_POLICY)
            self.assertEqual(output['python_version'],sys.version.split()[0])
            self.assertEqual((output['controllers'],output['evaluation_records']), (23,46))
            self.assertEqual((output['physical_first_episodes'],output['dependent_window_observations']),(8050,16100))
            self.assertTrue(output['summary_verified'])
            self.assertIn('attestation',output['limitation'])
    def test_malicious_and_noncanonical_paths(self):
        for path in ('../secret','/tmp/secret','a/../b','a//b','a/./b','a\\b','C:secret','.','a\nsecret'):
            with self.subTest(path=path), self.assertRaises(ValueError): v.relative(path)
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'actual').write_text('x');(root/'alias').symlink_to(root/'actual')
            with self.assertRaisesRegex(ValueError,'symlink'):v.inside(root,'alias')
    def test_duplicate_missing_hash_and_semantics_rejected(self):
        for mutation in ('duplicate','missing','hash','semantics','false_projection','exclusion'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);m=fixture(root)
                if mutation=='duplicate':m['files'].append(copy.deepcopy(m['files'][0]))
                if mutation=='missing':(root/m['files'][0]['path']).unlink()
                if mutation=='hash':m['files'][0]['published_sha256']='0'*64
                if mutation=='semantics':m['science_hash_semantics']='rewritten'
                if mutation=='false_projection':m['files'][0]['original_sha256']='0'*64
                if mutation=='exclusion':m['excludes']=['models/final.pt']
                with self.assertRaises(ValueError):v.manifest_files(root,m)
    def test_historical_claim_not_published_sha(self):
        entry={'path':'raw.json','original_sha256':'a'*64,'published_sha256':'b'*64,'metadata_strings_only':True}
        entries={'raw.json':entry}
        v.original_claim(entries,'raw.json','a'*64)
        with self.assertRaisesRegex(ValueError,'historical'):v.original_claim(entries,'raw.json','b'*64)
        with self.assertRaisesRegex(ValueError,'projected'):v.original_claim(entries,'raw.json','a'*64,immutable=True)
    def test_numeric_pair_preserves_types_and_signed_zero(self):
        self.assertFalse(v.exact([0.0],[False]))
        self.assertFalse(v.exact([0.0],[-0.0]))
        self.assertFalse(v.exact([1],[1.0]))
        self.assertTrue(v.exact({'a':[1.0,None,True]}, {'a':[1.0,None,True]}))
    def test_original_raw_hash_and_public_pair_tampering(self):
        for mutation in ('original_link','numeric_pair','source_projection','model_projection','summary_freeze'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);m=fixture(root)
                def update(name, change):
                    path=root/name;value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value))
                    next(e for e in m['files'] if e['path']==name)['published_sha256']=v.sha(path)
                if mutation=='original_link':
                    update(v.ART+'/evaluations/c0_g131_r111.json',lambda x:x['v25'].update(original_raw_sha256='f'*64))
                if mutation=='numeric_pair':
                    update(v.ART+'/evaluations/c0_g131_r111.json',lambda x:x['windows'].update({'16':[-0.0,False,None]}))
                if mutation in ('source_projection','model_projection'):
                    name='sources/0.py' if mutation=='source_projection' else 'models/final.pt'
                    next(e for e in m['files'] if e['path']==name)['metadata_strings_only']=True
                if mutation=='summary_freeze':
                    update(v.ART+'/summary.json',lambda x:x.update(freeze_sha256='e'*64))
                (root/v.ART/'publication_metadata.json').write_text(json.dumps(m))
                with self.assertRaises(ValueError):v.verify(root)
    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'x.json';path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(ValueError,'duplicate JSON'):v.read(path)


    def test_summary_evidence_fail_closed(self):
        mutations=('missing','empty','not_mapping','omitted_required','missing_target','unsafe','wrong_original')
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);m=fixture(root)
                name=v.ART+'/summary.json';path=root/name;summary=json.loads(path.read_text())
                if mutation=='missing': del summary['evidence_sha256']
                elif mutation=='empty': summary['evidence_sha256']={}
                elif mutation=='not_mapping': summary['evidence_sha256']=[]
                elif mutation=='omitted_required': del summary['evidence_sha256'][v.ART+'/evaluations/c0_g131_r111.json']
                elif mutation=='missing_target': summary['evidence_sha256']['missing-required-proof.json']='a'*64
                elif mutation=='unsafe': summary['evidence_sha256']['../../outside.json']='a'*64
                else: summary['evidence_sha256'][v.ART+'/frozen.json']='e'*64
                path.write_text(json.dumps(summary))
                next(e for e in m['files'] if e['path']==name)['published_sha256']=v.sha(path)
                (root/v.ART/'publication_metadata.json').write_text(json.dumps(m))
                with self.assertRaises(ValueError): v.verify(root)
    def test_filename_record_binding(self):
        for mutation in ('swap_maps','float_geometry','bool_reset','wrong_controller'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);m=fixture(root)
                first=v.ART+'/evaluations/c0_g131_r111.json'
                second=v.ART+'/evaluations/c0_g132_r112.json'
                raw_first=v.ART+'/evaluations/_adapter_raw/c0_g131_r111.json'
                raw_second=v.ART+'/evaluations/_adapter_raw/c0_g132_r112.json'
                if mutation=='swap_maps':
                    for a,b in ((first,second),(raw_first,raw_second)):
                        left,right=(root/a).read_bytes(),(root/b).read_bytes()
                        (root/a).write_bytes(right);(root/b).write_bytes(left)
                else:
                    for name in (first,raw_first):
                        data=json.loads((root/name).read_text())
                        if mutation=='float_geometry':data['geometry_seed']=131.0
                        elif mutation=='bool_reset':data['reset_seed']=True
                        else:data['controller']='c1'
                        (root/name).write_text(json.dumps(data))
                for entry in m['files']:
                    entry['published_sha256']=v.sha(root/entry['path'])
                (root/v.ART/'publication_metadata.json').write_text(json.dumps(m))
                with self.assertRaisesRegex(ValueError,'filename/record identity'): v.verify(root)

    def test_historical_arithmetic_numeric_contract(self):
        from decimal import Decimal
        self.assertEqual(v.historical_sum([1e16,1.,-1e16]),0.)
        self.assertEqual(v.historical_sum([]),0)
        self.assertIs(type(v.historical_sum([])),int)
        self.assertEqual(v.historical_sum([1,2,3]),6)
        self.assertEqual(v.historical_sum([True,False,True]),2)
        self.assertIs(type(v.historical_sum([True,False])),int)
        self.assertEqual(v.historical_sum([1,True,.5]),2.5)
        self.assertEqual(v.historical_sum(iter([.25,.5])),.75)
        self.assertEqual(v.historical_sum([-0.0]).hex(),'0x0.0p+0')
        self.assertEqual(v.historical_sum([-0.0,-0.0]).hex(),'0x0.0p+0')
        for value in ('1',None,1+0j,Decimal('1'),[],{}):
            with self.subTest(value=value),self.assertRaises(TypeError):v.historical_sum([value])
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(ValueError):v.historical_sum([value])
        class Number(int):pass
        with self.assertRaises(TypeError):v.historical_sum([Number(1)])
    def test_actual_public_v5_speed_vector_historical_bits(self):
        # Recorded public vector fixture; expected bits captured with CPython 3.11.
        root=SCRIPT.parents[1]/v.ART/'evaluations'
        paths=[root/f'v5_g{g}_r{r}.json' for g,r in ((131,111),(132,112))]
        if not all(path.exists() for path in paths):
            self.skipTest('public evaluation fixtures not present')
        for seconds,expected in ((16,'0x1.5d87160cb92d2p+10'),(64,'0x1.6f25f8367f976p+10')):
            values=[]
            for path in paths:
                b=json.loads(path.read_text());w=b['windows'][str(seconds)]
                values.extend(distance/(steps*b['dt']) for distance,steps in zip(w['forward_distance'],w['episode_steps']))
            self.assertEqual(len(values),350)
            self.assertEqual(v.historical_sum(values).hex(),expected)

if __name__=='__main__':unittest.main()
