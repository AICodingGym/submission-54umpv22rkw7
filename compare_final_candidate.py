"""Freeze an audited RLT candidate before evaluating the reserved validation split.

The fixed output directory allows retries for the same frozen comparison only.
Running this command opens final_validation; use it only after model selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deberta_baseline import FIXED, ROOT, integer_scores, metrics, prepare_split, write_json
from essay_bottleneck.score import BottleneckScorer
from score_deberta import DebertaScorer
from verify_deberta_run import sha256


def freeze_entry(directory, version, kind):
    directory = directory.resolve()
    config = json.loads((directory / 'config.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    audit = json.loads((directory / 'verification.json').read_text())
    digest = sha256(directory / 'model/model.pt')
    if (audit['model_sha256'] != digest or audit['max_reload_raw_difference'] != 0
            or not audit['all_integer_predictions_match']
            or not audit['all_report_metrics_recomputed']
            or not audit['all_token_lengths_recomputed']):
        raise ValueError(f'An intact, audited checkpoint is required: {directory}')
    bounds = (json.loads((directory / 'thresholds.json').read_text())['thresholds']
              if version == 'B1' else FIXED.tolist())
    return {'directory': str(directory), 'kind': kind, 'version': version,
            'model_sha256': digest, 'thresholds': bounds,
            'selected_epoch': report['selected_epoch'],
            'selection_qwk': report['splits']['selection'][version]['qwk'],
            'split_sha256': config['split_sha256'],
            'data_sha256': config['data_sha256'],
            'config_sha256': sha256(directory / 'config.json'),
            'model_files_sha256': {p.name: sha256(p) for p in sorted((directory / 'model').iterdir())
                                   if p.is_file() and p.name != 'model.pt'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--version', choices=['B0', 'B1'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    benchmarks = json.loads((ROOT / 'reports/strong_baseline.json').read_text())
    entries = {'candidate': freeze_entry(args.candidate_run, args.version, 'bottleneck')}
    if entries['candidate']['selection_qwk'] <= benchmarks['rlt_target_selection_qwk']:
        raise ValueError('Candidate must first surpass the fixed selection benchmark')
    for name in ['selection_benchmark', 'platform_benchmark']:
        reference = benchmarks[name]
        entry = freeze_entry(ROOT / reference['model_dir'], reference['version'], 'baseline')
        if entry['model_sha256'] != reference['model_sha256']:
            raise ValueError(f'Benchmark checkpoint changed: {name}')
        entries[name] = entry
    split_digest = sha256(ROOT / 'splits/deberta_seed42.csv')
    data_digest = sha256(ROOT / 'train.csv')
    for entry in entries.values():
        if entry['split_sha256'] != split_digest or entry['data_sha256'] != data_digest:
            raise ValueError('All models must use the same unchanged data and split')
    manifest = {'models': entries, 'purpose': 'One frozen final-validation comparison; no fitting'}
    output = ROOT / 'runs/rlt_final_comparison'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'frozen_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Final comparison already frozen with different models')
    else:
        with manifest_path.open('x') as handle:
            json.dump(manifest, handle, indent=2)
            handle.write('\n')
    # Model versions and thresholds are frozen on disk before loading holdout labels.
    frame = prepare_split(pd.read_csv(ROOT / 'train.csv', dtype={'essay_id': str}),
                          ROOT / 'splits/deberta_seed42.csv')
    frame = frame.loc[frame.split == 'final_validation'].reset_index(drop=True)
    results = {}
    for name, entry in entries.items():
        prediction_path = output / f'{name}_predictions.csv'
        if prediction_path.exists():
            predictions = pd.read_csv(prediction_path, dtype={'essay_id': str})
            if (not np.array_equal(predictions.essay_id, frame.essay_id)
                    or not np.array_equal(predictions.score, frame.score)):
                raise ValueError('Stored final predictions have mismatched rows')
            raw = predictions.raw_prediction.to_numpy(dtype=np.float32)
            scores = predictions.prediction.to_numpy()
        else:
            scorer_class = BottleneckScorer if entry['kind'] == 'bottleneck' else DebertaScorer
            scorer = scorer_class(entry['directory'], args.device, entry['version'])
            raw, scores = scorer.predict(frame.full_text.tolist(), batch_size=4)
            frame[['essay_id', 'score']].assign(raw_prediction=raw, prediction=scores).to_csv(
                prediction_path, index=False)
            del scorer
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        if not np.isfinite(raw).all() or not np.array_equal(scores, integer_scores(raw, entry['thresholds'])):
            raise ValueError('Invalid final predictions')
        results[name] = metrics(frame.score.to_numpy(), raw, entry['thresholds'])
        print(name, json.dumps(results[name]), flush=True)
    write_json(output / 'report.json', {'split': 'final_validation', 'rows': len(frame),
               'manifest_sha256': sha256(manifest_path), 'models': results,
               'note': 'No model fitting, threshold fitting, or model selection on this split.'})


if __name__ == '__main__':
    main()
