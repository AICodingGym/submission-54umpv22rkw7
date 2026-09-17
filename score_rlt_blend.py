"""Score or audit an explicitly labelled two-component baseline/RLT blend."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deberta_baseline import ROOT, integer_scores, metrics, write_json
from essay_bottleneck.score import BottleneckScorer
from score_deberta import DebertaScorer
from verify_deberta_run import check_metrics, sha256


class BlendScorer:
    def __init__(self, directory, device='cpu', version='B0'):
        self.directory = Path(directory)
        self.device = device
        self.manifest = json.loads((self.directory / 'components.json').read_text())
        self.entries = self.manifest['components']
        if (self.manifest['kind'] != 'baseline_rlt_equal_blend' or len(self.entries) != 2
                or [entry['kind'] for entry in self.entries] != ['baseline', 'bottleneck']
                or any(entry['weight'] != 0.5 for entry in self.entries)):
            raise ValueError('Expected the prespecified 50/50 baseline/RLT blend')
        bounds = json.loads((self.directory / 'thresholds.json').read_text())['versions']
        self.thresholds = np.asarray(bounds[version], dtype=np.float64)
        if (self.thresholds.shape != (5,) or not np.isfinite(self.thresholds).all()
                or not (np.diff(self.thresholds) > 0).all()):
            raise ValueError('Invalid score thresholds')
        for entry in self.entries:
            source = Path(entry['directory'])
            if (sha256(source / 'model/model.pt') != entry['model_sha256']
                    or sha256(source / 'config.json') != entry['config_sha256']):
                raise ValueError('A frozen blend component changed')
            for filename, digest in entry['model_files_sha256'].items():
                if sha256(source / 'model' / filename) != digest:
                    raise ValueError(f'Component model metadata changed: {filename}')

    def predict(self, texts, batch_size=4):
        self.component_predictions = []
        # Load components sequentially to keep inference memory bounded.
        for entry in self.entries:
            scorer_class = DebertaScorer if entry['kind'] == 'baseline' else BottleneckScorer
            scorer = scorer_class(entry['directory'], self.device, 'B0')
            raw, _ = scorer.predict(texts, batch_size=batch_size)
            self.component_predictions.append(raw.astype(np.float64))
            del scorer
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        raw = (self.component_predictions[0] + self.component_predictions[1]) / 2
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite blend predictions')
        return raw, integer_scores(raw, self.thresholds)


def audit(directory, device):
    scorer = BlendScorer(directory, device, 'B0')
    for entry in scorer.entries:
        if (entry['data_sha256'] != sha256(ROOT / 'train.csv')
                or entry['split_sha256'] != sha256(ROOT / 'splits/deberta_seed42.csv')):
            raise ValueError('Data or split differs from the frozen components')
    split = pd.read_csv(ROOT / 'splits/deberta_seed42.csv', dtype={'essay_id': str})
    source = pd.read_csv(ROOT / 'train.csv', dtype={'essay_id': str}).set_index('essay_id')
    report = json.loads((directory / 'report.json').read_text())
    thresholds = json.loads((directory / 'thresholds.json').read_text())['versions']
    frames = {}
    for name in ['calibration', 'selection']:
        frame = pd.read_csv(directory / f'{name}_predictions.csv', dtype={'essay_id': str})
        if (not frame.essay_id.is_unique
                or set(frame.essay_id) != set(split.loc[split.split == name, 'essay_id'])
                or not np.array_equal(frame.score, source.loc[frame.essay_id, 'score'])):
            raise ValueError('Invalid blend split rows or labels')
        parts = [pd.read_csv(Path(e['directory']) / f'{name}_predictions.csv', dtype={'essay_id': str})
                 .set_index('essay_id').loc[frame.essay_id].raw_prediction.to_numpy(dtype=np.float64)
                 for e in scorer.entries]
        if not np.array_equal(frame.raw_prediction, (parts[0] + parts[1]) / 2):
            raise ValueError('Stored predictions differ from the fixed blend')
        for version, bounds in thresholds.items():
            if not np.array_equal(frame[version], integer_scores(frame.raw_prediction.to_numpy(), bounds)):
                raise ValueError('Stored integer scores differ from frozen thresholds')
            check_metrics(report['splits'][name][version], frame.score.to_numpy(),
                          frame.raw_prediction.to_numpy(), frame[version].to_numpy())
        frames[name] = frame
    frame = frames['selection']
    raw, scores = scorer.predict(source.loc[frame.essay_id, 'full_text'].tolist())
    differences = []
    for entry, actual in zip(scorer.entries, scorer.component_predictions):
        expected = pd.read_csv(Path(entry['directory']) / 'selection_predictions.csv',
                               dtype={'essay_id': str}).set_index('essay_id').loc[frame.essay_id]
        difference = float(np.abs(actual - expected.raw_prediction.to_numpy()).max())
        if difference != 0:
            raise ValueError('Reloaded component predictions changed')
        differences.append(difference)
    if not np.array_equal(raw, frame.raw_prediction) or not np.array_equal(scores, frame.B0):
        raise ValueError('Reloaded blend predictions changed')
    code = [ROOT / f for f in ['score_rlt_blend.py', 'score_deberta.py', 'deberta_baseline.py']]
    code += sorted((ROOT / 'essay_bottleneck').glob('*.py'))
    verification = {'kind': 'baseline_rlt_equal_blend', 'single_model': False,
                    'components_sha256': sha256(directory / 'components.json'),
                    'thresholds_sha256': sha256(directory / 'thresholds.json'),
                    'all_report_metrics_recomputed': True, 'all_integer_predictions_match': True,
                    'selection_reload_rows': len(frame), 'component_max_reload_differences': differences,
                    'max_reload_raw_difference': 0., 'final_validation_evaluated': False,
                    'inference_code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in code}}
    write_json(directory / 'verification.json', verification)
    print(json.dumps(verification))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--version', choices=['B0', 'B1', 'affine'], default='B0')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit', action='store_true')
    mode.add_argument('--csv', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.audit:
        audit(args.model_dir, args.device)
        return
    frame = pd.read_csv(args.csv, dtype={'essay_id': str})
    if not frame.essay_id.is_unique or frame.full_text.isna().any():
        raise ValueError('Expected unique IDs and valid essay text')
    scorer = BlendScorer(args.model_dir, args.device, args.version)
    raw, scores = scorer.predict(frame.full_text.tolist())
    print(frame[['essay_id']].assign(raw_prediction=raw, score=scores).to_csv(index=False), end='')


if __name__ == '__main__':
    main()
