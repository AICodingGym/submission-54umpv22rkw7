"""Offline supervised bottleneck experiments on the existing fixed split."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from deberta_baseline import (FIXED, ROOT, fit_thresholds, integer_scores, metrics,
                              prepare_split, write_json)
from .model import BottleneckConfig, BottleneckRegressor
from .batching import encode_texts, group_within_effective_batches, pad_batch
from .features import cache_encoder_features, pad_cached_features
from .averaging import ParameterEMA


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-size', choices=['small', 'base'], default='base')
    parser.add_argument('--source', type=Path, help='Local pretrained encoder/tokenizer override')
    parser.add_argument('--init-run', type=Path,
                        help='Audited bottleneck weights for a new joint-training run; fresh optimizer')
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not already exist')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--effective-batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=None,
                        help='Optional explicit truncation cap; default keeps full essays')
    parser.add_argument('--pooling', choices=['latent', 'mean'], default='latent')
    parser.add_argument('--num-latents', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--reconstruction-weight', type=float, default=0.1)
    parser.add_argument('--score-objective', choices=['mse', 'ordinal_bce'], default='mse')
    parser.add_argument('--finetune-encoder', action='store_true')
    parser.add_argument('--normalize-queries', action='store_true')
    parser.add_argument('--group-microbatches', action='store_true',
                        help='Sort lengths within each effective batch; keep its sample membership')
    parser.add_argument('--cache-frozen-features', action='store_true',
                        help='Cache train-only frozen encoder features in CPU RAM')
    parser.add_argument('--class-weight-power', type=float, default=0.,
                        help='Inverse train class-frequency power for scoring loss; 0 disables weighting')
    parser.add_argument('--ema-decay', type=float, default=0.,
                        help='Trainable-parameter EMA for evaluation and export; 0 disables it')
    parser.add_argument('--encoder-lr', type=float, default=2e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--smoke', action='store_true', help='One epoch, at most two essays per grade/split')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.effective_batch_size) < 1:
        parser.error('epoch and batch counts must be positive')
    if ((args.max_length is not None and args.max_length < 2)
            or args.effective_batch_size % args.batch_size):
        parser.error('explicit max length must be >= 2 and effective batch must be a multiple of batch size')
    if any(not math.isfinite(lr) or lr <= 0 for lr in [args.encoder_lr, args.head_lr]):
        parser.error('learning rates must be finite and positive')
    if args.init_run and not args.finetune_encoder:
        parser.error('--init-run currently requires --finetune-encoder')
    if args.cache_frozen_features and args.finetune_encoder:
        parser.error('--cache-frozen-features requires a frozen encoder')
    if not math.isfinite(args.class_weight_power) or not 0 <= args.class_weight_power <= 1:
        parser.error('--class-weight-power must be finite and in [0, 1]')
    if not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        parser.error('--ema-decay must be finite and in [0, 1)')
    if args.ema_decay and args.finetune_encoder:
        parser.error('--ema-decay currently requires a frozen encoder')
    return args


def make_score_weights(labels, power):
    labels = np.asarray(labels)
    if not len(labels) or not np.isin(labels, np.arange(1, 7)).all():
        raise ValueError('Expected train grades in 1..6')
    counts = np.bincount(labels.astype(int), minlength=7)[1:]
    weights = np.ones(6, dtype=np.float64)
    if power:
        if (counts == 0).any():
            raise ValueError('Class weighting requires train examples in every grade')
        weights = (len(labels) / (6 * counts)) ** power
        weights /= np.dot(counts, weights) / counts.sum()
    return weights.astype(np.float32), counts


def load_initial_state(directory, architecture, source, split_path):
    """Validate a completed parent run; thresholds and optimizer are never inherited."""
    directory = directory.resolve()
    config = json.loads((directory / 'config.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    audit = json.loads((directory / 'verification.json').read_text())
    previous = BottleneckConfig(**json.loads((directory / 'model/bottleneck.json').read_text()))
    for name, value in asdict(architecture).items():
        if name != 'finetune_encoder' and getattr(previous, name) != value:
            raise ValueError(f'Warm-start architecture differs: {name}')
    if (config['smoke'] or report['smoke'] or report['final_validation_evaluated']
            or Path(config['source']).resolve() != source.resolve()
            or config['split_sha256'] != hashlib.sha256(split_path.read_bytes()).hexdigest()
            or config['data_sha256'] != hashlib.sha256((ROOT / 'train.csv').read_bytes()).hexdigest()):
        raise ValueError('Warm start requires a full parent run with matching source, data and split')
    checkpoint = directory / 'model/model.pt'
    digest = hashlib.sha256()
    with checkpoint.open('rb') as weights:
        for block in iter(lambda: weights.read(1024 * 1024), b''):
            digest.update(block)
    if (digest.hexdigest() != audit['model_sha256'] or audit['max_reload_raw_difference'] != 0
            or not audit['all_integer_predictions_match'] or not audit['all_report_metrics_recomputed']):
        raise ValueError('Parent checkpoint does not match a successful audit')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if any(name.startswith('teacher.') for name in state):
        raise ValueError('Expected a deployment checkpoint without teacher weights')
    provenance = {'run': str(directory), 'model_sha256': digest.hexdigest(),
                  'selected_epoch': report['selected_epoch'], 'optimizer_restored': False,
                  'thresholds_inherited': False, 'teacher': 'frozen copy of initial warm-start encoder'}
    return state, provenance


def main():
    args = parse_args()
    architecture = BottleneckConfig(**{k: getattr(args, k) for k in BottleneckConfig.__dataclass_fields__})
    architecture.validate()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a small local check')
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing run: {output}')
    source = args.source
    if source is None:
        suffix = '' if args.model_size == 'small' else '_base'
        source = Path((ROOT / f'models/deberta{suffix}_snapshot.txt').read_text().strip())
        if not source.is_absolute():
            source = ROOT / source
    source = source.resolve()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if tokenizer.padding_side != 'right':
        raise ValueError('This position-based bottleneck requires right padding')
    split_path = ROOT / 'splits/deberta_seed42.csv'
    if not split_path.exists():
        raise ValueError('Create the baseline fixed split first; this experiment never creates it')
    frame = pd.read_csv(ROOT / 'train.csv', dtype={'essay_id': str})
    if (not frame.essay_id.is_unique or frame.full_text.isna().any()
            or frame.full_text.str.strip().eq('').any() or not frame.score.isin(range(1, 7)).all()):
        raise ValueError('Invalid essay data')
    frame = prepare_split(frame, split_path)
    data, tokens, lengths, truncated = {}, {}, {}, {}
    for name in ['train', 'selection', 'calibration']:
        part = frame.loc[frame.split == name].copy()
        if args.smoke:
            part = part.groupby('score', group_keys=False).head(2)
        part = part.reset_index(drop=True)
        data[name] = part
        tokens[name], lengths[name], truncated[name] = encode_texts(
            tokenizer, part.full_text.tolist(), args.max_length)
    class_weights, class_counts = make_score_weights(
        frame.loc[frame.split == 'train', 'score'].to_numpy(), args.class_weight_power)
    bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    initial_state, initial_provenance = (load_initial_state(args.init_run, architecture, source, split_path)
                                         if args.init_run else (None, None))
    encoder = AutoModel.from_pretrained(source, local_files_only=True)
    if initial_state is not None:
        encoder.load_state_dict({key.removeprefix('encoder.'): value
                                 for key, value in initial_state.items() if key.startswith('encoder.')})
    # Construct the fixed teacher after the student's initial encoder is restored.
    model = BottleneckRegressor(encoder, architecture)
    if initial_state is not None:
        restored = model.load_state_dict(initial_state, strict=False)
        teacher_keys = {key for key in model.state_dict() if key.startswith('teacher.')}
        if set(restored.missing_keys) != teacher_keys or restored.unexpected_keys:
            raise ValueError(f'Incomplete student warm start: {restored}')
        del initial_state
    model = model.to(device)
    if args.finetune_encoder:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    task_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith('encoder.')]
    groups = [{'params': task_params, 'lr': args.head_lr}]
    if encoder_params:
        groups.append({'params': encoder_params, 'lr': args.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    epochs = 1 if args.smoke else args.epochs
    batches = math.ceil(len(data['train']) / args.batch_size)
    accumulation = args.effective_batch_size // args.batch_size
    total_steps = epochs * math.ceil(batches / accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(0.1 * total_steps), total_steps)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(architecture=asdict(architecture), source=str(source), revision=source.name,
                  epochs=epochs, precision='bf16' if bf16 else 'fp32',
                  split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                  data_sha256=hashlib.sha256((ROOT / 'train.csv').read_bytes()).hexdigest(),
                  torch=torch.__version__, final_validation_evaluated=False,
                  truncation=tokenizer.truncation_side if args.max_length is not None else 'none',
                  padding='batch_longest',
                  teacher='frozen initial encoder' if architecture.reconstruction_weight else None,
                  class_weights=class_weights.tolist(), train_class_counts=class_counts.tolist(),
                  evaluation_weights='ema' if args.ema_decay else 'optimizer',
                  lengths={name: {'rows': len(v), 'truncated': int(truncated[name].sum()),
                                  'max_tokens': int(v.max())} for name, v in lengths.items()})
    if initial_provenance is not None:
        config['initial_bottleneck'] = initial_provenance
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    config['git_commit'] = commit.stdout.strip() if commit.returncode == 0 else None
    config['code_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path(__file__).parent.glob('*.py'))}
    config['code_sha256']['deberta_baseline.py'] = hashlib.sha256((ROOT / 'deberta_baseline.py').read_bytes()).hexdigest()
    provenance_path = source / 'supervised_source.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        for key in ['split_sha256', 'data_sha256']:
            if provenance[key] != config[key]:
                raise ValueError('Supervised encoder source uses different data or split')
        digest = hashlib.sha256()
        with (source / 'model.safetensors').open('rb') as weights:
            for block in iter(lambda: weights.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != provenance['encoder_weights_sha256']:
            raise ValueError('Supervised encoder weights differ from export provenance')
        config['supervised_source'] = provenance
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    print(json.dumps(config), flush=True)

    def inputs(name, indices):
        return pad_batch(tokenizer, tokens[name], indices, device)

    def predict(name):
        model.eval()
        raw = np.empty(len(data[name]), dtype=np.float32)
        order = np.argsort([len(ids) for ids in tokens[name]])
        with torch.inference_mode():
            for offset in range(0, len(order), args.eval_batch_size):
                indices = order[offset:offset + args.eval_batch_size]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                    raw[indices] = model(**inputs(name, indices))['scores'].cpu().numpy()
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite predictions')
        return raw

    cached_features = None
    if args.cache_frozen_features:
        cached_features, config['feature_cache'] = cache_encoder_features(
            model.encoder, tokenizer, tokens['train'], device, bf16, args.eval_batch_size,
            output / 'cache_progress.json')
        write_json(output / 'config.json', config)
        print(json.dumps({'feature_cache': config['feature_cache']}), flush=True)

    score_weight_table = (torch.tensor(class_weights, dtype=torch.float32, device=device)
                          if args.class_weight_power else None)
    ema = ParameterEMA(model, args.ema_decay) if args.ema_decay else None
    best, best_epoch, history = -np.inf, None, []
    for epoch in range(epochs):
        model.train()
        start = time.monotonic()
        order = np.random.default_rng(args.seed + epoch).permutation(len(data['train']))
        if args.group_microbatches:
            order = group_within_effective_batches(order, [len(ids) for ids in tokens['train']],
                                                  args.effective_batch_size)
        sums = dict(loss=0., score_loss=0., reconstruction_loss=0.)
        optimizer.zero_grad(set_to_none=True)
        for batch in range(batches):
            indices = order[batch * args.batch_size:(batch + 1) * args.batch_size]
            labels = torch.tensor(data['train'].score.to_numpy()[indices], dtype=torch.float32, device=device)
            encoded = (pad_cached_features(cached_features, indices, device)
                       if cached_features is not None else None)
            batch_weights = (score_weight_table[labels.long() - 1]
                             if score_weight_table is not None else None)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                result = model(**inputs('train', indices), labels=labels, encoded_hidden=encoded,
                               score_weights=batch_weights)
            if not torch.isfinite(result['loss']):
                raise ValueError('Nonfinite training loss')
            window_start = batch // accumulation * args.effective_batch_size
            window_samples = min(args.effective_batch_size, len(order) - window_start)
            (result['loss'] * len(indices) / window_samples).backward()
            for key in sums:
                sums[key] += result[key].detach().item() * len(indices)
            if (batch + 1) % accumulation == 0 or batch + 1 == batches:
                torch.nn.utils.clip_grad_norm_(task_params + encoder_params, 1., error_if_nonfinite=True)
                optimizer.step()
                if ema is not None:
                    ema.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch % 50 == 0 or batch + 1 == batches:
                progress = {'epoch': epoch + 1, 'batch': batch + 1, 'batches': batches,
                            'seconds': time.monotonic() - start,
                            **{key: result[key].detach().item() for key in sums}}
                write_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        with ema.apply() if ema is not None else nullcontext():
            raw = predict('selection')
            selection = metrics(data['selection'].score.to_numpy(), raw, FIXED)
            if selection['qwk'] is not None and selection['qwk'] > best:
                best, best_epoch = selection['qwk'], epoch + 1
                model.save(output / 'model')
                tokenizer.save_pretrained(output / 'model')
        history.append({'epoch': epoch + 1, 'seconds': time.monotonic() - start,
                        **{key: value / len(order) for key, value in sums.items()}, 'selection': selection})
        if ema is not None:
            history[-1]['ema_updates'] = ema.updates
        write_json(output / 'history.json', history)
        print(json.dumps(history[-1]), flush=True)
    if best_epoch is None:
        raise ValueError('No valid selection QWK; no checkpoint selected')
    # Drop training-only teacher/optimizer before restoring the selected student.
    del result, optimizer, scheduler, encoder_params, task_params, groups, encoder, model, cached_features, encoded, ema
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model = BottleneckRegressor.load(output / 'model', device)
    calibration = predict('calibration')
    # Smoke only exercises the path; thresholds on 12 rows are not meaningful.
    thresholds = FIXED.copy() if args.smoke else fit_thresholds(data['calibration'].score.to_numpy(), calibration)
    write_json(output / 'thresholds.json', {'thresholds': thresholds.tolist(),
               'fit_split': None if args.smoke else 'calibration', 'selected_epoch': best_epoch})
    report = {'selected_epoch': best_epoch, 'smoke': args.smoke,
              'final_validation_evaluated': False, 'splits': {}}
    for name in ['selection', 'calibration']:
        raw = calibration if name == 'calibration' else predict(name)
        y = data[name].score.to_numpy()
        report['splits'][name] = {'B0': metrics(y, raw, FIXED), 'B1': metrics(y, raw, thresholds)}
        data[name][['essay_id', 'score']].assign(raw_prediction=raw, B0=integer_scores(raw),
            B1=integer_scores(raw, thresholds), token_length=lengths[name],
            truncated=truncated[name]).to_csv(output / f'{name}_predictions.csv', index=False)
    write_json(output / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
