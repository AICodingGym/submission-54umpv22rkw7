"""Sequential matched end-to-end regression / auxiliary ordinal experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_small1024_pooling import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/small1024_dualhead_v1')
    parser.add_argument('--resume-audit-failure', action='store_true',
                        help='Resume a failed ordinal audit without rerunning completed training')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = ['deberta_baseline.py', 'run_small1024_dualhead.py', 'audit_deberta_ordinal.py',
             'verify_deberta_run.py', 'score_deberta.py']
    fingerprints = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    if args.resume_audit_failure:
        status = json.loads((output / 'pipeline_status.json').read_text())
        assert status['state'] == 'failed' and status['stage'].endswith('_ordinal_audit')
        previous_hashes = json.loads((output / 'source_hashes.json').read_text())
        for name in ['deberta_baseline.py', 'verify_deberta_run.py', 'score_deberta.py']:
            assert previous_hashes[name] == fingerprints[name], f'Training or inference changed: {name}'
        # Only one recovery launch is allowed, preserving the original failure.
        with (output / 'pipeline_resume_claim.json').open('x') as stream:
            json.dump({'pid': os.getpid(), 'resumed_at': time.time(), 'previous_status': status,
                       'previous_hashes': previous_hashes, 'resumed_hashes': fingerprints}, stream, indent=2)
        status = {**status, 'pid': os.getpid(), 'state': 'running', 'resumed_at': time.time()}
        status.pop('error', None)
    else:
        with (output / 'pipeline_claim.json').open('x') as stream:
            json.dump({'pid': os.getpid(), 'started_at': time.time()}, stream)
        status = {'pid': os.getpid(), 'state': 'running', 'completed': [], 'started_at': time.time()}
        write_json(output / 'source_hashes.json', fingerprints)

    def command(stage, argv):
        if stage in status['completed']:
            return
        for name, digest in fingerprints.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Source changed: {name}'
        status.update(stage=stage, command=argv, updated_at=time.time())
        write_json(output / 'pipeline_status.json', status)
        log_path = output / f'{stage}{"_resumed" if args.resume_audit_failure else ""}.log'
        with log_path.open('x') as log:
            subprocess.run([sys.executable, '-u', *argv], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        status['completed'].append(stage)
        status['updated_at'] = time.time()
        write_json(output / 'pipeline_status.json', status)

    common = ['deberta_baseline.py', '--model-size', 'small', '--max-length', '1024',
              '--pooling', 'mean', '--epochs', '10', '--seed', '42', '--batch-size', '4', '--eval-batch-size', '4']
    comparison = {'seed': 42, 'epochs': 10, 'max_length': 1024,
                  'primary_metric': 'selection regression B0 QWK', 'final_validation_evaluated': False,
                  'runs': {}, 'interpretation': 'Single-seed end-to-end screen; no fusion. Replicate improvements before scaling.'}
    try:
        reference = None
        for name, weight in [('regression', 0.), ('dual_01', .1), ('dual_03', .3)]:
            directory = output / name
            command(f'{name}_train', [*common, '--ordinal-weight', str(weight), '--output-dir', str(directory)])
            initial = json.loads((directory / 'initialization.json').read_text())
            if reference is None:
                reference = initial
            for key in ['encoder_sha256', 'regression_head_sha256', 'cpu_rng_sha256']:
                assert initial[key] == reference[key], f'Unmatched initialization: {name} {key}'
            if weight == .3:
                other = json.loads((output / 'dual_01/initialization.json').read_text())
                assert initial['ordinal_head_sha256'] == other['ordinal_head_sha256']
            command(f'{name}_audit', ['verify_deberta_run.py', '--run-dir', str(directory)])
            if weight:
                command(f'{name}_ordinal_audit', ['audit_deberta_ordinal.py', '--run-dir', str(directory)])
            report = json.loads((directory / 'report.json').read_text())
            comparison['runs'][name] = {'ordinal_weight': weight, 'selected_epoch': report['selected_epoch'],
                                       'selection': report['splits']['selection'], 'initialization_matched': True}
            if weight:
                comparison['runs'][name]['B0_delta_vs_regression'] = (report['splits']['selection']['B0']['qwk']
                    - comparison['runs']['regression']['selection']['B0']['qwk'])
            write_json(output / 'comparison_partial.json', comparison)
        write_json(output / 'comparison.json', comparison)
        status.update(state='completed', updated_at=time.time())
    except Exception as error:
        status.update(state='failed', error=repr(error), updated_at=time.time())
        raise
    finally:
        write_json(output / 'pipeline_status.json', status)


if __name__ == '__main__':
    main()
