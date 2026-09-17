"""Audit auxiliary ordinal outputs; regression is audited by verify_deberta_run.py."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deberta_baseline import ROOT, write_json
from score_deberta import DebertaScorer
from verify_deberta_run import check_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    config = json.loads((directory / 'config.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    history = json.loads((directory / 'history.json').read_text())
    assert config['ordinal_weight'] > 0 and not config['freeze_encoder']
    assert config['prediction_head'] == 'regression'
    assert not report['final_validation_evaluated']
    assert not (directory / 'final_validation_predictions.csv').exists()
    torch.set_num_threads(8)
    scorer = DebertaScorer(directory, device='cuda', version='B0')
    cutpoints = scorer.model.ordinal_head.cutpoints().detach().cpu().numpy()
    assert len(cutpoints) == 5 and np.isfinite(cutpoints).all() and (np.diff(cutpoints) > 0).all()
    for epoch in history:
        assert np.isclose(epoch['train_loss'], epoch['train_mse']
                          + config['ordinal_weight'] * epoch['train_ordinal_bce'], rtol=1e-6, atol=1e-6)
        assert np.all(np.diff(epoch['selection_ordinal']['cutpoints']) > 0)
    initial = json.loads((directory / 'initialization.json').read_text())
    updated = {}
    for prefix, key in [('encoder.', 'encoder_sha256'), ('head.', 'regression_head_sha256'),
                        ('ordinal_head.', 'ordinal_head_sha256')]:
        digest = hashlib.sha256()
        for name, value in scorer.model.named_parameters():
            if name.startswith(prefix):
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        updated[prefix] = digest.hexdigest() != initial[key]
        assert updated[prefix], f'Parameters never changed: {prefix}'
    audited = 0
    columns = [f'ordinal_logit_gt_{k}' for k in range(1, 6)]
    for split in ['train', 'selection', 'calibration']:
        pred = pd.read_csv(directory / f'{split}_predictions.csv', dtype={'essay_id': str}, float_precision='round_trip')
        # Match row-major logits from inference; reduction order otherwise
        # differs for pandas' column-major matrix at float32 precision.
        logits = np.ascontiguousarray(pred[columns].to_numpy(dtype=np.float32))
        assert np.isfinite(logits).all() and (np.diff(logits, axis=1) <= 0).all()
        expected = (1 + torch.from_numpy(logits).sigmoid().sum(1)).numpy()
        assert np.array_equal(expected, pred.ordinal_expected.to_numpy(dtype=np.float32))
        assert ((expected >= 1) & (expected <= 6)).all()
        targets = (pred.score.to_numpy()[:, None] > np.arange(1, 6)).astype(np.float32)
        bce = float(torch.nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(logits), torch.from_numpy(targets)))
        saved = report['splits'][split]['ordinal']
        assert np.isclose(saved['bce'], bce, atol=1e-7, rtol=0)
        assert np.array_equal(cutpoints, np.asarray(saved['cutpoints'], dtype=np.float32))
        integer = np.searchsorted(np.arange(1.5, 6, 1), expected, side='right') + 1
        # ordinal_diagnostics computes metrics with float32 labels. Match its
        # arithmetic dtype, keeping the existing strict metric tolerances.
        check_metrics(saved['B0'], pred.score.to_numpy(dtype=np.float32), expected, integer)
        audited += len(pred)
    # Independently reload both heads on precisely the saved selection IDs.
    selection = pd.read_csv(directory / 'selection_predictions.csv', dtype={'essay_id': str}, float_precision='round_trip')
    split_ids = pd.read_csv(ROOT / 'splits/deberta_seed42.csv', dtype={'essay_id': str})
    allowed = set(split_ids.loc[split_ids.split == 'selection', 'essay_id'])
    assert set(selection.essay_id) <= allowed
    if not config['smoke']:
        assert set(selection.essay_id) == allowed
    frame = pd.read_csv(ROOT / 'train.csv', dtype={'essay_id': str}).set_index('essay_id')
    tokens = scorer.tokenizer(frame.loc[selection.essay_id, 'full_text'].tolist(),
                              truncation=True, max_length=config['max_length'])['input_ids']
    order = np.argsort([len(ids) for ids in tokens])
    raw = np.empty(len(order), dtype=np.float32)
    logits = np.empty((len(order), 5), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), config['eval_batch_size']):
            indices = order[start:start + config['eval_batch_size']]
            inputs = scorer.tokenizer.pad({'input_ids': [tokens[i] for i in indices]},
                                          padding=True, return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=scorer.bf16):
                regression, ordinal = scorer.model(return_ordinal=True, **inputs)
            raw[indices], logits[indices] = regression.cpu().numpy(), ordinal.cpu().numpy()
    regression_difference = float(np.max(np.abs(raw - selection.raw_prediction.to_numpy(dtype=np.float32))))
    ordinal_difference = float(np.max(np.abs(logits - selection[columns].to_numpy(dtype=np.float32))))
    assert regression_difference == ordinal_difference == 0
    verification = {'prediction_rows_audited': audited, 'selection_reload_rows': len(selection),
                    'regression_reload_max_difference': regression_difference,
                    'ordinal_logits_reload_max_difference': ordinal_difference,
                    'loss_components_reconciled': True, 'ordered_cutpoints': cutpoints.tolist(),
                    'updated_parameter_groups': updated, 'final_validation_evaluated': False}
    write_json(directory / 'ordinal_verification.json', verification)
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
