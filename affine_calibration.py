"""Fit a fixed two-parameter calibration grid on calibration, then score selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deberta_baseline import FIXED, integer_scores, metrics, write_json
from verify_deberta_run import check_metrics, sha256


def fit_affine(y, raw):
    # Prespecified grid; no selection examples enter this function.
    slopes, offsets = np.meshgrid(np.linspace(0.7, 1.4, 141), np.linspace(-1, 1, 401))
    slopes, offsets = slopes.ravel(), offsets.ravel()
    bounds = (FIXED[None] - offsets[:, None]) / slopes[:, None]
    best = None
    for start in range(0, len(bounds), 256):
        thresholds = bounds[start:start + 256]
        scores = (raw[None, :, None] >= thresholds[:, None, :]).sum(-1) + 1
        mean = scores.mean(1)
        denominator = y.var() + scores.var(1) + (y.mean() - mean) ** 2
        qwk = 1 - ((scores - y[None]) ** 2).mean(1) / denominator
        displacement = ((thresholds - FIXED[None]) ** 2).mean(1)
        index = int(np.lexsort((displacement, -qwk))[0])
        key = (-float(qwk[index]), float(displacement[index]))
        if best is None or key < best[0]:
            best = (key, start + index)
    index = best[1]
    return {'slope': float(slopes[index]), 'offset': float(offsets[index]),
            'thresholds': bounds[index].tolist(), 'calibration_qwk': -best[0][0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if len({path.name for path in args.model_dir}) != len(args.model_dir):
        parser.error('Model directory names must be distinct')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fitted = {}
    for directory in args.model_dir:
        audit = json.loads((directory / 'verification.json').read_text())
        report = json.loads((directory / 'report.json').read_text())
        if sha256(directory / 'model/model.pt') != audit['model_sha256']:
            raise ValueError('Checkpoint differs from the audited model')
        frame = pd.read_csv(directory / 'calibration_predictions.csv', dtype={'essay_id': str})
        y, raw = frame.score.to_numpy(), frame.raw_prediction.to_numpy()
        check_metrics(report['splits']['calibration']['B0'], y, raw, integer_scores(raw))
        calibration = fit_affine(y, raw)
        measured = metrics(y, raw, calibration['thresholds'])
        if not np.isclose(measured['qwk'], calibration['calibration_qwk'], atol=1e-12, rtol=0):
            raise ValueError('Vectorized QWK disagrees with the reference metric')
        fitted[directory.name] = {'source_dir': str(directory.resolve()),
            'model_sha256': audit['model_sha256'], 'fit_split': 'calibration',
            'source_calibration_sha256': sha256(directory / 'calibration_predictions.csv'),
            'calibration': calibration, 'calibration_metrics': measured}
    # All candidate calibration parameters are fixed before any selection predictions are read.
    write_json(args.output_dir / 'fitted_calibrations.json', {
        'method': 'Affine score calibration; slope 0.7..1.4 step 0.005, offset -1..1 step 0.005',
        'tie_break': 'Smallest mean squared threshold displacement from half integers',
        'code_sha256': sha256(Path(__file__)), 'models': fitted})
    for directory in args.model_dir:
        frame = pd.read_csv(directory / 'selection_predictions.csv', dtype={'essay_id': str})
        entry = fitted[directory.name]
        bounds = entry['calibration']['thresholds']
        entry['selection_metrics'] = metrics(frame.score.to_numpy(), frame.raw_prediction.to_numpy(), bounds)
        frame[['essay_id', 'score', 'raw_prediction']].assign(
            affine_score=integer_scores(frame.raw_prediction.to_numpy(), bounds)).to_csv(
                args.output_dir / f'{directory.name}_selection_predictions.csv', index=False)
        print(directory.name, json.dumps({k: entry['selection_metrics'][k] for k in ['qwk', 'mae']}), flush=True)
    write_json(args.output_dir / 'report.json', {'models': fitted, 'final_validation_evaluated': False})


if __name__ == '__main__':
    main()
