"""Run the baseline and matched frozen-encoder pooling screen, sequentially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/small1024_pooling_v1')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive claim prevents simultaneous runners or accidental overwrite.
    with (output / 'pipeline_claim.json').open('x') as stream:
        json.dump({'pid': os.getpid(), 'started_at': time.time()}, stream)
    status = {'pid': os.getpid(), 'state': 'running', 'completed': [], 'started_at': time.time()}

    def command(stage, argv):
        status.update(stage=stage, command=argv, updated_at=time.time())
        write_json(output / 'pipeline_status.json', status)
        with (output / f'{stage}.log').open('x') as log:
            subprocess.run([sys.executable, '-u', *argv], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        status['completed'].append(stage)
        write_json(output / 'pipeline_status.json', status)

    common = ['deberta_baseline.py', '--model-size', 'small', '--max-length', '1024',
              '--epochs', '10', '--seed', '42', '--batch-size', '4', '--eval-batch-size', '4']
    baseline = output / 'baseline'
    try:
        command('baseline_train', [*common, '--output-dir', str(baseline)])
        command('baseline_audit', ['verify_deberta_run.py', '--run-dir', str(baseline)])
        for pooling in ['mean', 'attention']:
            directory = output / f'frozen_{pooling}'
            command(f'frozen_{pooling}_train', [*common, '--pooling', pooling,
                    '--encoder-checkpoint', str(baseline), '--freeze-encoder', '--output-dir', str(directory)])
            command(f'frozen_{pooling}_audit', ['verify_deberta_run.py', '--run-dir', str(directory)])
        # Audit every frozen tensor against the shared supervised encoder.
        import torch
        torch.set_num_threads(2)
        source = torch.load(baseline / 'model/model.pt', map_location='cpu', weights_only=True)
        comparison = {'seed': 42, 'epochs_per_run': 10, 'max_length': 1024,
                      'final_validation_evaluated': False, 'runs': {}}
        for name in ['baseline', 'frozen_mean', 'frozen_attention']:
            directory = output / name
            report = json.loads((directory / 'report.json').read_text())
            record = {'selected_epoch': report['selected_epoch'], 'selection': report['splits']['selection']}
            if name.startswith('frozen_'):
                state = torch.load(directory / 'model/model.pt', map_location='cpu', weights_only=True)
                encoder_keys = [k for k in source if k.startswith('encoder.')]
                assert all(torch.equal(source[k], state[k]) for k in encoder_keys), 'Frozen encoder changed'
                record['encoder_tensors_unchanged'] = len(encoder_keys)
                if name == 'frozen_attention':
                    assert torch.count_nonzero(state['attention.weight']) > 0, 'Attention did not learn'
                del state
            comparison['runs'][name] = record
        gap = (comparison['runs']['frozen_attention']['selection']['B0']['qwk']
               - comparison['runs']['frozen_mean']['selection']['B0']['qwk'])
        comparison['attention_minus_mean_selection_B0_qwk'] = gap
        comparison['interpretation'] = 'Single-seed frozen-encoder screen; joint tuning and seed replication required before backbone scaling.'
        write_json(output / 'comparison.json', comparison)
        status.update(state='completed', updated_at=time.time())
    except Exception as error:
        status.update(state='failed', error=repr(error), updated_at=time.time())
        raise
    finally:
        write_json(output / 'pipeline_status.json', status)


if __name__ == '__main__':
    main()
