"""Package an audited RLT model with already fitted affine thresholds as B1."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from affine_calibration import fit_affine
from deberta_baseline import integer_scores, metrics, write_json
from verify_deberta_run import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source, calibration_dir, output = (p.resolve() for p in
                                      [args.model_dir, args.calibration_dir, args.output_dir])
    if output.exists():
        raise ValueError('Refusing to overwrite a candidate')
    audit = json.loads((source / 'verification.json').read_text())
    fitted_path = calibration_dir / 'fitted_calibrations.json'
    fitted = json.loads(fitted_path.read_text())
    entry = fitted['models'][source.name]
    if (sha256(source / 'model/model.pt') != audit['model_sha256']
            or audit['model_sha256'] != entry['model_sha256']
            or entry['source_dir'] != str(source) or entry['fit_split'] != 'calibration'
            or sha256(source / 'calibration_predictions.csv') != entry['source_calibration_sha256']
            or audit['max_reload_raw_difference'] != 0
            or not audit['all_report_metrics_recomputed']
            or fitted['code_sha256'] != sha256(Path(__file__).with_name('affine_calibration.py'))):
        raise ValueError('Calibration provenance does not match the audited model')
    calibration = pd.read_csv(source / 'calibration_predictions.csv', dtype={'essay_id': str})
    # Refit only calibration to verify the frozen thresholds before reading selection.
    if fit_affine(calibration.score.to_numpy(), calibration.raw_prediction.to_numpy()) != entry['calibration']:
        raise ValueError('Frozen calibration parameters are not reproducible')
    bounds = np.asarray(entry['calibration']['thresholds'])
    config = json.loads((source / 'config.json').read_text())
    report = json.loads((source / 'report.json').read_text())
    if config['smoke'] or report['final_validation_evaluated']:
        raise ValueError('Expected a full run without final-validation evaluation')
    provenance = {'method': fitted['method'], 'source_run': str(source),
                  'source_model_sha256': audit['model_sha256'],
                  'source_calibration_sha256': entry['source_calibration_sha256'],
                  'fitted_parameters_sha256': sha256(fitted_path),
                  'packaging_code_sha256': sha256(Path(__file__)),
                  'parameters': entry['calibration'], 'fit_split': 'calibration',
                  'weights_retrained': False, 'deployment_version': 'B1'}
    config['calibration_override'] = provenance
    report['calibration_override'] = provenance
    output.mkdir(parents=True, exist_ok=False)
    (output / 'model').symlink_to(source / 'model', target_is_directory=True)
    shutil.copy2(source / 'history.json', output / 'history.json')
    for name in ['calibration', 'selection']:
        frame = calibration if name == 'calibration' else pd.read_csv(
            source / 'selection_predictions.csv', dtype={'essay_id': str})
        frame['B1'] = integer_scores(frame.raw_prediction.to_numpy(), bounds)
        report['splits'][name]['B1'] = metrics(frame.score.to_numpy(), frame.raw_prediction.to_numpy(), bounds)
        frame.to_csv(output / f'{name}_predictions.csv', index=False)
    write_json(output / 'config.json', config)
    write_json(output / 'report.json', report)
    write_json(output / 'thresholds.json', {'thresholds': bounds.tolist(), 'fit_split': 'calibration',
               'selected_epoch': report['selected_epoch'], 'calibration_override': provenance})
    print(json.dumps(report['splits']['selection']['B1']))


if __name__ == '__main__':
    main()
