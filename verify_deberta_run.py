"""Audit saved DeBERTa predictions and independently reload the selected model."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error

from deberta_baseline import ROOT, write_json
from score_deberta import DebertaScorer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_metrics(expected, y, raw, scores):
    assert expected['rows'] == len(y)
    if not len(y):
        return
    qwk = cohen_kappa_score(y, scores, labels=list(range(1, 7)), weights='quadratic')
    if expected['qwk'] is None:
        assert not np.isfinite(qwk)
    else:
        assert np.isclose(expected['qwk'], qwk, atol=1e-12, rtol=0)
    assert np.isclose(expected['mae'], mean_absolute_error(y, scores), atol=1e-12, rtol=0)
    assert np.isclose(expected['continuous_mae'], mean_absolute_error(y, raw), atol=1e-6, rtol=0)
    assert np.array_equal(expected['confusion_matrix'], confusion_matrix(y, scores, labels=list(range(1, 7))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    config = json.loads((directory / 'config.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    calibration = json.loads((directory / 'thresholds.json').read_text())
    history = json.loads((directory / 'history.json').read_text())
    assert not config['smoke'], 'This audit requires a complete run'
    assert len(history) == config['epochs']
    best = max(history, key=lambda h: h['selection']['qwk'])
    assert report['selected_epoch'] == calibration['selected_epoch'] == best['epoch']
    assert report['splits']['selection']['B0'] == best['selection']
    assert calibration['fit_split'] == 'calibration'
    thresholds = np.asarray(calibration['thresholds'])
    assert thresholds.shape == (5,) and np.isfinite(thresholds).all() and (np.diff(thresholds) > 0).all()
    assert report['final_validation_evaluated'] is False
    split_path = ROOT / 'splits/deberta_seed42.csv'
    assert sha256(split_path) == config['split_sha256']
    metadata = json.loads(split_path.with_suffix('.json').read_text())
    assert sha256(ROOT / 'train.csv') == metadata['data_sha256']
    split = pd.read_csv(split_path, dtype={'essay_id': str})
    assert split.groupby('group').split.nunique().max() == 1
    # Only the three development partitions enter the audit or inference.
    source = pd.read_csv(ROOT / 'train.csv', dtype={'essay_id': str})
    source = source[source.essay_id.isin(split.loc[split.split != 'final_validation', 'essay_id'])].set_index('essay_id')
    audited = 0
    for name in ['train', 'selection', 'calibration']:
        pred = pd.read_csv(directory / f'{name}_predictions.csv', dtype={'essay_id': str})
        assert pred.essay_id.is_unique
        assert set(pred.essay_id) == set(split.loc[split.split == name, 'essay_id'])
        assert np.array_equal(pred.score, source.loc[pred.essay_id, 'score'])
        raw = pred.raw_prediction.to_numpy()
        assert np.isfinite(raw).all()
        assert np.array_equal(pred.truncated, pred.token_length > config['max_length'])
        for version, boundaries in [('B0', np.arange(1.5, 6, 1)), ('B1', thresholds)]:
            scores = np.searchsorted(boundaries, raw, side='right') + 1
            assert np.array_equal(pred[version], scores)
            check_metrics(report['splits'][name][version], pred.score, raw, scores)
            for group, mask in [('truncated', pred.truncated.to_numpy()), ('untruncated', ~pred.truncated.to_numpy())]:
                check_metrics(report['splits'][name][f'{group}_{version}'], pred.score.to_numpy()[mask], raw[mask], scores[mask])
        audited += len(pred)
    torch.set_num_threads(8)
    scorer = DebertaScorer(directory, device='cuda', version='B1')
    selection = pd.read_csv(directory / 'selection_predictions.csv', dtype={'essay_id': str})
    raw, scores = scorer.predict(source.loc[selection.essay_id, 'full_text'].tolist(), batch_size=config['eval_batch_size'])
    difference = float(np.max(np.abs(raw - selection.raw_prediction.to_numpy(dtype=np.float32))))
    assert difference == 0, f'Reloaded raw scores differ: {difference}'
    assert np.array_equal(scores, selection.B1)
    assert np.array_equal(np.searchsorted(np.arange(1.5, 6, 1), raw, side='right') + 1, selection.B0)
    assert not (directory / 'final_validation_predictions.csv').exists()
    verification = {'prediction_rows_audited': audited, 'selection_reload_rows': len(selection),
                    'max_reload_raw_difference': difference, 'all_integer_predictions_match': True,
                    'all_report_metrics_recomputed': True, 'reserved_final_rows': int((split.split == 'final_validation').sum()),
                    'final_validation_evaluated': False, 'model_sha256': sha256(directory / 'model/model.pt')}
    suffix = '' if config.get('model_size', 'small') == 'small' else f"_{config['model_size']}"
    snapshot = Path((ROOT / f'models/deberta{suffix}_snapshot.txt').read_text().strip())
    snapshot = snapshot if snapshot.is_absolute() else ROOT / snapshot
    assert snapshot.name == config['revision']
    environment = {'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy', 'pandas', 'scikit-learn', 'sentencepiece', 'protobuf', 'huggingface-hub', 'tokenizers']},
                   'base_weights_sha256': sha256(snapshot / 'pytorch_model.bin'), 'base_model_revision': config['revision'],
                   'training_script_sha256': sha256(ROOT / 'deberta_baseline.py'),
                   'encoder_layers': scorer.model.encoder.config.num_hidden_layers,
                   'model_parameters': sum(p.numel() for p in scorer.model.parameters())}
    write_json(directory / 'verification.json', verification)
    write_json(directory / 'environment.json', environment)
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
