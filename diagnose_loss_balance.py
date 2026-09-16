"""Compare score/reconstruction readout gradients on 18 frozen-encoder train essays."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from deberta_baseline import ROOT, write_json
from essay_bottleneck.model import BottleneckRegressor
from verify_deberta_run import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = args.run_dir.resolve()
    digest = sha256(run / 'model/model.pt')
    model = BottleneckRegressor.load(run / 'model', 'cpu')
    if model.config.finetune_encoder or not model.config.reconstruction_weight:
        raise ValueError('This diagnostic requires a frozen encoder and a reconstruction loss')
    if digest != sha256(run / 'model/model.pt'):
        raise ValueError('Checkpoint changed while loading')
    config = json.loads((run / 'config.json').read_text())
    split = pd.read_csv(ROOT / 'splits/deberta_seed42.csv', dtype={'essay_id': str})
    frame = pd.read_csv(ROOT / 'train.csv', dtype={'essay_id': str})
    frame = frame[frame.essay_id.isin(split.loc[split.split == 'train', 'essay_id'])]
    frame = frame.groupby('score', group_keys=False).head(3).reset_index(drop=True)
    assert len(frame) == 18
    tokenizer = AutoTokenizer.from_pretrained(run / 'model', local_files_only=True)
    shared = list(model.readout.parameters())
    rows = []
    for start in range(0, len(frame), 2):
        part = frame.iloc[start:start + 2]
        options = ({'truncation': False} if config['max_length'] is None else
                   {'truncation': True, 'max_length': config['max_length']})
        inputs = tokenizer(part.full_text.tolist(), padding=True, return_tensors='pt', **options)
        result = model(**inputs, labels=torch.tensor(part.score.to_numpy(), dtype=torch.float32))
        score = torch.cat([value.flatten() for value in torch.autograd.grad(
            result['score_loss'], shared, retain_graph=True)])
        reconstruction = torch.cat([value.flatten() for value in torch.autograd.grad(
            result['reconstruction_loss'], shared)]) * model.config.reconstruction_weight
        cosine = torch.nn.functional.cosine_similarity(score[None], reconstruction[None]).item()
        rows.append({'essay_ids': part.essay_id.tolist(), 'score_loss': result['score_loss'].item(),
                     'reconstruction_loss': result['reconstruction_loss'].item(),
                     'gradient_cosine': cosine, 'score_gradient_norm': score.norm().item(),
                     'weighted_reconstruction_gradient_norm': reconstruction.norm().item(),
                     'weighted_gradient_norm_ratio': (reconstruction.norm() / score.norm().clamp_min(1e-12)).item()})
    report = {'run': str(run), 'model_sha256': digest, 'split': 'train', 'precision': 'CPU FP32',
              'mode': 'eval, dropout disabled', 'batches': rows,
              'mean_gradient_cosine': float(np.mean([r['gradient_cosine'] for r in rows])),
              'mean_weighted_gradient_norm_ratio': float(np.mean([r['weighted_gradient_norm_ratio'] for r in rows])),
              'note': 'Small local gradient diagnostic, not evidence that removing reconstruction improves generalization.'}
    write_json(ROOT / 'reports' / run.name / 'loss_balance_diagnostic.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'batches'}), flush=True)


if __name__ == '__main__':
    main()
