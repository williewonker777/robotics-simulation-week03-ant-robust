# SPDX-License-Identifier: BSD-3-Clause
"""Post-experiment public-byte and recorded-array verification (stdlib only).

This presentation tool does not replay training, load checkpoints, or verify private
runtime/cache evidence. Original-to-public metadata equivalence is a publisher
attestation when the original bytes are not public. Historical hashes stay historical.
Use a trusted reviewed checkout: the auditor is imported Python code, and this
manifest establishes consistency, not an independent authenticity trust anchor.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import struct
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = 'artifacts/terrain_demo/teammate_port_v25'
SEMANTICS = 'original execution bytes; not rewritten for publication'
SCHEMA = 'week03_ant_teammate_v25_publication_v1'
ARITHMETIC_POLICY = 'historical_numeric_left_fold_v1'


def historical_sum(values):
    """Original execution's ordered numeric addition, independent of Python sum.

    CPython 3.12 changed float sum accuracy. Recorded means use the 3.11
    left-fold order, not a compensated total or an approximate comparison.
    Only exact built-in numeric types present in the frozen auditor are accepted.
    """
    total = 0
    for value in values:
        if type(value) not in (bool, int, float):
            raise TypeError('historical_sum requires built-in bool/int/float values')
        if type(value) is float and not math.isfinite(value):
            raise ValueError('historical_sum requires finite inputs')
        total += value
    return total


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None, 'invalid SHA256')
    return value


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
            and all(ord(c) >= 32 for c in name), 'unsafe manifest path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and str(path) == name
            and name != '.', 'noncanonical manifest path')
    return name


def inside(root, name):
    relative(name)
    path = root / name
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        require(not current.is_symlink(), 'symlink publication path: ' + name)
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), 'missing public file: ' + name)
    return path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=invalid)


def exact(left, right):
    """Public pair identity preserves types and floating-point signed zero."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return struct.pack('!d', left) == struct.pack('!d', right)
    return left == right


def manifest_files(root, manifest):
    require(manifest['schema'] == SCHEMA and manifest['science_hash_semantics'] == SEMANTICS,
            'publication schema/hash semantics mismatch')
    require(isinstance(manifest['files'], list) and manifest['files'], 'empty public inventory')
    entries = {}
    for item in manifest['files']:
        name = relative(item['path'])
        require(name not in entries, 'duplicate publication path')
        require(type(item['metadata_strings_only']) is bool, 'metadata flag must be boolean')
        original, published = digest(item['original_sha256']), digest(item['published_sha256'])
        require(item['metadata_strings_only'] or original == published, 'unchanged file has different hash domains')
        require(sha(inside(root, name)) == published, 'published hash mismatch: ' + name)
        entries[name] = item
    excludes = manifest.get('excludes', [])
    require(isinstance(excludes, list), 'excludes must be a list')
    require(len(excludes) == len(set(excludes)), 'duplicate exclusion')
    for name in excludes:
        relative(name)
        require(name.endswith(('.log', '.jsonl')) and name not in entries, 'only unlisted logs may be excluded')
    return entries


def original_claim(entries, name, expected, immutable=False):
    relative(name)
    require(name in entries, 'missing manifest reference: ' + name)
    item = entries[name]
    require(item['original_sha256'] == digest(expected), 'historical hash mapping mismatch: ' + name)
    if immutable:
        require(not item['metadata_strings_only'] and item['published_sha256'] == expected,
                'scientific source/model was projected: ' + name)


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = read(inside(root, ART + '/publication_metadata.json'))
    entries = manifest_files(root, manifest)
    freeze_name, summary_name = ART + '/frozen.json', ART + '/summary.json'
    require(freeze_name in entries and summary_name in entries, 'freeze/summary absent from inventory')
    frozen = read(inside(root, freeze_name))
    require(frozen['phase'] == 'holdout', 'public replay requires complete holdout')
    require(len(frozen['source_sha256']) == 20 and len(frozen['base_source_sha256']) == 6,
            'frozen source inventory size differs')
    for group in ('source_sha256', 'base_source_sha256'):
        for name, expected in frozen[group].items():
            original_claim(entries, name, expected, immutable=True)
    require(len(frozen['models']) == 23, 'model controller inventory differs')
    for item in frozen['models'].values():
        original_claim(entries, item['checkpoint'], item['sha256'], immutable=True)
    # Import the immutable auditor only after checking its actual published bytes.
    auditor_name = 'scripts/audit_teammate_v25_raw.py'
    require(auditor_name in frozen['source_sha256'], 'auditor not frozen')
    spec = importlib.util.spec_from_file_location('v25_public_frozen_auditor', inside(root, auditor_name))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    # Deliberate module-local arithmetic binding; no builtins/global patch.
    audit.sum = historical_sum
    records = []
    required_evidence = {freeze_name}
    for controller in audit.CONTROLLERS:
        for geometry, reset in audit.MAPS:
            filename = f'{controller}_g{geometry}_r{reset}.json'
            name = ART + '/evaluations/' + filename
            raw_name = ART + '/evaluations/_adapter_raw/' + filename
            require(name in entries and raw_name in entries, 'missing declared evaluation pair')
            required_evidence.update((name, raw_name))
            bundle, raw = read(inside(root, name)), read(inside(root, raw_name))
            require(bundle['controller'] == controller
                    and type(bundle['geometry_seed']) is int and bundle['geometry_seed'] == geometry
                    and type(bundle['reset_seed']) is int and bundle['reset_seed'] == reset,
                    'filename/record identity differs: ' + filename)
            original_claim(entries, raw_name, bundle['v25']['original_raw_sha256'])
            numerical = dict(bundle)
            del numerical['v25']
            require(exact(numerical, raw), 'public raw/enriched numerical identity differs: ' + filename)
            model = frozen['models'][controller]
            require(bundle['checkpoint'] == model['checkpoint'], 'published checkpoint path binding differs')
            records.append(bundle)
    summary = read(inside(root, summary_name))
    require(summary['freeze_path'] == freeze_name, 'summary freeze path binding differs')
    original_claim(entries, freeze_name, summary['freeze_sha256'])
    # Every historical link must have a public representation in the original-byte
    # domain. Checking those records does not re-execute private runtime/cache work.
    require('evidence_sha256' in summary, 'missing summary evidence mapping')
    evidence = summary['evidence_sha256']
    require(isinstance(evidence, dict) and bool(evidence), 'summary evidence must be a nonempty mapping')
    require(required_evidence <= evidence.keys(), 'required summary evidence link omitted')
    for name, expected in evidence.items():
        original_claim(entries, name, expected)
    canonical = audit.audit_evaluation_records(records, frozen)
    audit.compare_summary(summary, canonical)
    return dict(verdict='PASS', arithmetic_policy=ARITHMETIC_POLICY, python_version=sys.version.split()[0],
        public_files_verified=len(entries), controllers=23, evaluation_records=46,
        physical_first_episodes=canonical['physical_first_episodes'],
        dependent_window_observations=canonical['dependent_window_observations'], summary_verified=True,
        scope='Published bytes and recorded raw-array/summary consistency; no simulation or model unpickling.',
        limitation='Training/runtime/cache evidence is a local verification record, not replayed here. '
            'Original-to-published metadata equivalence is publisher attestation when originals are private.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root), sort_keys=True))


if __name__ == '__main__':
    main()
