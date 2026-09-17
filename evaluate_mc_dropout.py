"""Three seeded dropout passes of one existing checkpoint; average raw scores."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deberta_baseline import ROOT, FIXED, fit_thresholds, integer_scores, metrics, write_json
from score_deberta import DebertaScorer
from verify_deberta_run import sha256


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    torch.set_num_threads(8)
    seeds = [42, 17, 2026]
    source = ROOT / 'reports/small1024_dual01_submission'
    metadata = json.loads((source / 'submission.json').read_text())
    run = ROOT / metadata['run_dir']
    output = ROOT / 'reports/three_seed_inference/mc_dropout'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        raise ValueError('This experiment is complete; refusing to overwrite it')
    provenance = {'model_sha256': metadata['model_sha256'], 'seeds': seeds,
                  'batch_size': 4, 'dropout_p': .1, 'source': metadata['run_dir']}
    if (output / 'config.json').exists():
        assert json.loads((output / 'config.json').read_text()) == provenance
    write_json(output / 'config.json', provenance)
    assert sha256(run / 'model/model.pt') == metadata['model_sha256']
    scorer = DebertaScorer(run, device='cuda', version='B1')
    old_thresholds = scorer.thresholds.copy()
    frame = pd.read_csv(ROOT / 'train.csv', dtype={'essay_id': str}).set_index('essay_id')
    split_ids = pd.read_csv(ROOT / 'splits/deberta_seed42.csv', dtype={'essay_id': str})
    datasets, baseline = {}, {}
    for name in ['selection', 'calibration']:
        pred = pd.read_csv(source / f'{name}_predictions.csv', dtype={'essay_id': str}, float_precision='round_trip')
        assert pred.essay_id.is_unique
        assert set(pred.essay_id) == set(split_ids.loc[split_ids.split == name, 'essay_id'])
        part = frame.loc[pred.essay_id].reset_index()
        assert np.array_equal(part.score, pred.score)
        datasets[name], baseline[name] = part, pred.raw_prediction.to_numpy()
    sample = pd.read_csv(ROOT / 'sample_submission.csv', dtype={'essay_id': str})
    test = pd.read_csv(ROOT / 'test.csv', dtype={'essay_id': str}).set_index('essay_id')
    assert test.index.is_unique and sample.essay_id.is_unique and set(test.index) == set(sample.essay_id)
    datasets['test'] = test.loc[sample.essay_id].reset_index()
    original_test = pd.read_csv(source / 'test_predictions.csv', dtype={'essay_id': str}, float_precision='round_trip')
    assert original_test.essay_id.equals(sample.essay_id)
    baseline['test'] = original_test.raw_prediction.to_numpy()

    # Enable only existing dropout modules, keeping the overall model in eval.
    scorer.model.eval()
    dropouts = []
    for name, module in scorer.model.named_modules():
        if isinstance(module, torch.nn.Dropout) and module.p > 0:
            module.train()
            dropouts.append({'name': name, 'p': module.p})
    assert dropouts and not scorer.model.training
    sample_texts = datasets['selection'].full_text.iloc[:32].tolist()
    seed_all(seeds[0]); first, _ = scorer.predict(sample_texts, batch_size=4)
    seed_all(seeds[0]); repeated, _ = scorer.predict(sample_texts, batch_size=4)
    assert np.array_equal(first, repeated), 'Seed replay is not reproducible'
    seed_all(seeds[1]); different, _ = scorer.predict(sample_texts, batch_size=4)
    assert np.any(first != different), 'Dropout did not produce stochastic predictions'
    results, saved_frames = {}, {}
    for name, part in datasets.items():
        passes = []
        cached = output / f'{name}_predictions.csv'
        if cached.exists():
            previous = pd.read_csv(cached, dtype={'essay_id': str}, float_precision='round_trip')
            assert previous.essay_id.equals(part.essay_id)
            assert np.array_equal(previous.eval_raw.to_numpy(), baseline[name])
            passes = [previous[f'raw_seed_{seed}'].to_numpy(dtype=np.float32) for seed in seeds]
            print(json.dumps({'stage': 'reuse_saved_passes', 'split': name}), flush=True)
        else:
            for seed in seeds:
                seed_all(seed)
                raw, _ = scorer.predict(part.full_text.tolist(), batch_size=4)
                passes.append(raw)
                progress = {'stage': 'inference', 'split': name, 'seed': seed, 'rows': len(raw)}
                write_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        stacked = np.stack(passes)
        assert np.isfinite(stacked).all()
        mean = stacked.mean(axis=0, dtype=np.float64)
        std = stacked.std(axis=0, dtype=np.float64)
        columns = ['essay_id', 'score'] if name != 'test' else ['essay_id']
        saved = part[columns].copy()
        saved['eval_raw'] = baseline[name]
        for index, seed in enumerate(seeds):
            saved[f'raw_seed_{seed}'] = stacked[index]
        saved['mean_raw'] = mean
        saved['seed_std'] = std
        saved['B0'] = integer_scores(mean)
        saved['B1_original_thresholds'] = integer_scores(mean, old_thresholds)
        saved.to_csv(output / f'{name}_predictions.csv', index=False)
        results[name], saved_frames[name] = (stacked, mean, std), saved
    thresholds = fit_thresholds(datasets['calibration'].score.to_numpy(), results['calibration'][1])
    write_json(output / 'thresholds.json', {'thresholds': thresholds.tolist(), 'fit_split': 'calibration',
                                          'prediction': 'mean of three seeded dropout regression outputs'})
    report = {'model': metadata['model'], 'source_run': metadata['run_dir'],
              'source_model_sha256': metadata['model_sha256'], 'selected_epoch': metadata['selected_epoch'],
              'seeds': seeds, 'batch_size': 4, 'aggregation': 'arithmetic mean of raw regression scores, then thresholding',
              'inference': 'MC dropout; same checkpoint, no training or weight changes',
              'dropout_modules': dropouts, 'seed_replay_max_difference': float(np.max(np.abs(first - repeated))),
              'different_seed_probe_max_difference': float(np.max(np.abs(first - different))),
              'original_thresholds': old_thresholds.tolist(), 'mc_thresholds': thresholds.tolist(),
              'final_validation_evaluated': False, 'platform_submitted': False, 'splits': {}}
    for name, (stacked, mean, std) in results.items():
        saved = saved_frames[name]
        saved['B1_recalibrated'] = integer_scores(mean, thresholds)
        saved.to_csv(output / f'{name}_predictions.csv', index=False)
        reloaded = pd.read_csv(output / f'{name}_predictions.csv', float_precision='round_trip')
        reconstructed = reloaded[[f'raw_seed_{s}' for s in seeds]].to_numpy(dtype=np.float32).mean(1, dtype=np.float64)
        assert np.allclose(reconstructed, reloaded.mean_raw, atol=1e-12, rtol=0)
        assert np.array_equal(integer_scores(reloaded.mean_raw, thresholds), reloaded.B1_recalibrated)
        record = {'rows': len(mean), 'mean_seed_std': float(std.mean()), 'p95_seed_std': float(np.quantile(std, .95)),
                  'mean_raw_shift_vs_eval': float(np.mean(mean - baseline[name])),
                  'mean_absolute_raw_shift_vs_eval': float(np.mean(np.abs(mean - baseline[name])))}
        if name != 'test':
            y = datasets[name].score.to_numpy()
            record.update(eval_B0=metrics(y, baseline[name], FIXED), eval_B1=metrics(y, baseline[name], old_thresholds),
                          mc_mean_B0=metrics(y, mean, FIXED), mc_mean_B1_original=metrics(y, mean, old_thresholds),
                          mc_mean_B1_recalibrated=metrics(y, mean, thresholds),
                          individual_seeds={str(seed): {'B0': metrics(y, stacked[i], FIXED),
                                                       'B1_original': metrics(y, stacked[i], old_thresholds)}
                                            for i, seed in enumerate(seeds)})
        report['splits'][name] = record
    for version in ['B1_original_thresholds', 'B1_recalibrated']:
        submission = saved_frames['test'][['essay_id', version]].rename(columns={version: 'score'})
        assert list(submission.columns) == list(sample.columns) and submission.essay_id.equals(sample.essay_id)
        assert submission.score.isin(range(1, 7)).all()
        submission.to_csv(output / f'submission_{version}.csv', index=False)
    assert sha256(run / 'model/model.pt') == metadata['model_sha256']
    report['checkpoint_file_unchanged'] = True
    write_json(output / 'report.json', report)
    write_json(output / 'progress.json', {'stage': 'completed'})
    print(json.dumps({'selection': report['splits']['selection']}), flush=True)


if __name__ == '__main__':
    main()
