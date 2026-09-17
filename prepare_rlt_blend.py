"""Prespecified 50/50 blend of a frozen baseline and RLT; not a single-model result."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from affine_calibration import fit_affine
from compare_final_candidate import freeze_entry
from deberta_baseline import FIXED, fit_thresholds, integer_scores, metrics, write_json
from verify_deberta_run import check_metrics, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--rlt-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite a blend experiment')
    entries = [freeze_entry(args.baseline_run, 'B0', 'baseline'),
               freeze_entry(args.rlt_run, 'B0', 'bottleneck')]
    if any(entries[0][key] != entries[1][key] for key in ['split_sha256', 'data_sha256']):
        raise ValueError('Blend components use different data or splits')
    for entry in entries:
        entry['weight'] = 0.5
    manifest = {'kind': 'baseline_rlt_equal_blend', 'single_model': False,
                'components': entries, 'fit_weights': False,
                'preparation_code_sha256': sha256(Path(__file__)),
                'final_validation_evaluated': False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'components.json', manifest)

    def blend(split):
        frames = []
        for entry in entries:
            directory = Path(entry['directory'])
            frame = pd.read_csv(directory / f'{split}_predictions.csv', dtype={'essay_id': str})
            if not frame.essay_id.is_unique:
                raise ValueError('Duplicate essay IDs')
            report = json.loads((directory / 'report.json').read_text())
            check_metrics(report['splits'][split]['B0'], frame.score.to_numpy(),
                          frame.raw_prediction.to_numpy(), frame.B0.to_numpy())
            frames.append(frame.set_index('essay_id'))
        if set(frames[0].index) != set(frames[1].index):
            raise ValueError('Different split essay IDs')
        other = frames[1].loc[frames[0].index]
        if not np.array_equal(frames[0].score, other.score):
            raise ValueError('Labels differ between components')
        raw = (frames[0].raw_prediction.to_numpy(dtype=np.float64)
               + other.raw_prediction.to_numpy(dtype=np.float64)) / 2
        return frames[0][['score']].assign(raw_prediction=raw).reset_index()

    calibration = blend('calibration')
    y, raw = calibration.score.to_numpy(), calibration.raw_prediction.to_numpy()
    affine = fit_affine(y, raw)
    thresholds = {'B0': FIXED.tolist(), 'B1': fit_thresholds(y, raw).tolist(),
                  'affine': affine['thresholds']}
    # All calibration choices are fixed before selection predictions are accessed.
    write_json(output / 'thresholds.json', {'versions': thresholds, 'affine': affine,
               'fit_split': 'calibration', 'weights_fitted': False})
    report = {'kind': manifest['kind'], 'single_model': False, 'splits': {},
              'final_validation_evaluated': False}
    for split in ['calibration', 'selection']:
        frame = calibration if split == 'calibration' else blend(split)
        y, raw = frame.score.to_numpy(), frame.raw_prediction.to_numpy()
        report['splits'][split] = {}
        for version, bounds in thresholds.items():
            frame[version] = integer_scores(raw, bounds)
            report['splits'][split][version] = metrics(y, raw, bounds)
        frame.to_csv(output / f'{split}_predictions.csv', index=False)
    write_json(output / 'report.json', report)
    print(json.dumps({key: {k: value[k] for k in ['qwk', 'mae']}
                      for key, value in report['splits']['selection'].items()}))


if __name__ == '__main__':
    main()
