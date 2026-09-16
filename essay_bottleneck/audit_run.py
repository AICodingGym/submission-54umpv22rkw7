"""Audit a completed bottleneck run and reload its complete selection predictions."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from transformers import AutoModel

from deberta_baseline import ROOT, write_json
from verify_deberta_run import check_metrics, sha256
from .score import BottleneckScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    config = json.loads((directory / 'config.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    calibration = json.loads((directory / 'thresholds.json').read_text())
    history = json.loads((directory / 'history.json').read_text())
    assert not config['smoke'] and not report['smoke']
    assert not config['final_validation_evaluated'] and not report['final_validation_evaluated']
    assert len(history) == config['epochs']
    best = max(history, key=lambda h: h['selection']['qwk'])
    assert best['epoch'] == report['selected_epoch'] == calibration['selected_epoch']
    assert best['selection'] == report['splits']['selection']['B0']
    assert calibration['fit_split'] == 'calibration'
    thresholds = np.asarray(calibration['thresholds'])
    assert thresholds.shape == (5,) and np.isfinite(thresholds).all() and (np.diff(thresholds) > 0).all()
    split_path = ROOT / 'splits/deberta_seed42.csv'
    assert sha256(split_path) == config['split_sha256']
    assert sha256(ROOT / 'train.csv') == config['data_sha256']
    for filename, digest in config['code_sha256'].items():
        assert sha256(ROOT / filename) == digest, f'Run code changed: {filename}'
    split = pd.read_csv(split_path, dtype={'essay_id': str})
    assert split.essay_id.is_unique and split.groupby('group').split.nunique().max() == 1
    source = pd.read_csv(ROOT / 'train.csv', dtype={'essay_id': str})
    source = source[source.essay_id.isin(split.loc[split.split != 'final_validation', 'essay_id'])].set_index('essay_id')
    torch.set_num_threads(4)
    scorer = BottleneckScorer(directory, device='cuda', version='B1')
    assert scorer.model.teacher is None
    frozen_encoder_unchanged = None
    if not config['finetune_encoder']:
        original = AutoModel.from_pretrained(config['source'], local_files_only=True)
        reference = original.state_dict()
        for name, value in scorer.model.encoder.state_dict().items():
            assert torch.equal(value.cpu(), reference[name]), f'Frozen encoder changed: {name}'
        frozen_encoder_unchanged = True
        del reference, original
    covered, predictions = {}, {}
    for name in ['train', 'selection', 'calibration']:
        ids = split.loc[split.split == name, 'essay_id']
        lengths = pd.Series([len(ids) for ids in scorer.tokenizer(
            source.loc[ids, 'full_text'].tolist(), truncation=False)['input_ids']], index=ids)
        cap = config['max_length']
        truncated = lengths > cap if cap is not None else lengths < 0
        assert len(lengths) == config['lengths'][name]['rows']
        assert int(lengths.max()) == config['lengths'][name]['max_tokens']
        assert int(truncated.sum()) == config['lengths'][name]['truncated']
        covered[name] = {'rows': len(lengths), 'truncated': int(truncated.sum())}
        if name == 'train':
            continue
        pred = pd.read_csv(directory / f'{name}_predictions.csv', dtype={'essay_id': str})
        assert pred.essay_id.is_unique and set(pred.essay_id) == set(ids)
        assert np.array_equal(pred.score, source.loc[pred.essay_id, 'score'])
        assert np.array_equal(pred.token_length, lengths.loc[pred.essay_id])
        assert np.array_equal(pred.truncated, truncated.loc[pred.essay_id])
        raw = pred.raw_prediction.to_numpy()
        assert np.isfinite(raw).all()
        for version, bounds in [('B0', np.arange(1.5, 6, 1)), ('B1', thresholds)]:
            scores = np.searchsorted(bounds, raw, side='right') + 1
            assert np.array_equal(scores, pred[version])
            check_metrics(report['splits'][name][version], pred.score, raw, scores)
        predictions[name] = pred
    selection = predictions['selection']
    raw, scores = scorer.predict(source.loc[selection.essay_id, 'full_text'].tolist(),
                                 batch_size=config['eval_batch_size'])
    difference = float(np.abs(raw - selection.raw_prediction.to_numpy(dtype=np.float32)).max())
    assert difference == 0 and np.array_equal(scores, selection.B1)
    assert np.array_equal(np.searchsorted(np.arange(1.5, 6, 1), raw, side='right') + 1, selection.B0)
    assert not (directory / 'final_validation_predictions.csv').exists()
    verification = {'prediction_rows_audited': sum(len(p) for p in predictions.values()),
                    'selection_reload_rows': len(selection), 'max_reload_raw_difference': difference,
                    'all_integer_predictions_match': True, 'all_report_metrics_recomputed': True,
                    'all_token_lengths_recomputed': True, 'length_coverage': covered,
                    'frozen_encoder_unchanged': frozen_encoder_unchanged,
                    'final_validation_evaluated': False, 'model_sha256': sha256(directory / 'model/model.pt')}
    write_json(directory / 'verification.json', verification)
    environment = {'packages': {p: importlib.metadata.version(p) for p in
                   ['torch', 'transformers', 'numpy', 'pandas', 'scikit-learn', 'sentencepiece', 'tokenizers']},
                   'gpu': torch.cuda.get_device_name(), 'source_revision': config['revision'],
                   'model_parameters': sum(p.numel() for p in scorer.model.parameters())}
    write_json(directory / 'environment.json', environment)
    if args.baseline_dir:
        base = pd.read_csv(args.baseline_dir / 'selection_predictions.csv', dtype={'essay_id': str})
        assert base.essay_id.is_unique and set(base.essay_id) == set(selection.essay_id)
        base = base.set_index('essay_id').loc[selection.essay_id].reset_index()
        assert np.array_equal(base.score, selection.score)
        assert np.array_equal(base.token_length, selection.token_length)
        comparison = {'baseline_dir': str(args.baseline_dir), 'candidate_dir': str(directory), 'groups': {}}
        for group, mask in [('all', np.ones(len(selection), dtype=bool)),
                            ('over_512', selection.token_length.to_numpy() > 512),
                            ('at_most_512', selection.token_length.to_numpy() <= 512)]:
            result = {'rows': int(mask.sum())}
            for version in ['B0', 'B1']:
                for model_name, frame in [('baseline', base), ('candidate', selection)]:
                    y, scores = frame.score.to_numpy()[mask], frame[version].to_numpy()[mask]
                    result[f'{model_name}_{version}'] = {
                        'qwk': float(cohen_kappa_score(y, scores, labels=list(range(1, 7)), weights='quadratic')),
                        'mae': float(mean_absolute_error(y, scores))}
            comparison['groups'][group] = result
        write_json(directory / 'baseline_comparison.json', comparison)
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
