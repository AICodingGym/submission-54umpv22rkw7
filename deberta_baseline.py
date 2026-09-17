"""Single-split DeBERTa regression; final_validation is never evaluated here."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
FIXED = np.arange(1.5, 6, 1)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def integer_scores(raw, thresholds=FIXED):
    return np.searchsorted(thresholds, raw, side='right') + 1


def metrics(y, raw, thresholds):
    if not len(y):
        return {'rows': 0}
    scores = integer_scores(raw, thresholds)
    qwk = float(cohen_kappa_score(y, scores, labels=list(range(1, 7)), weights='quadratic'))
    return {'rows': len(y), 'qwk': qwk if np.isfinite(qwk) else None,
            'mae': float(mean_absolute_error(y, scores)),
            'continuous_mae': float(mean_absolute_error(y, raw)),
            'confusion_matrix': confusion_matrix(y, scores, labels=list(range(1, 7))).tolist()}


def prepare_split(frame, path):
    fingerprint = hashlib.sha256((ROOT / 'train.csv').read_bytes()).hexdigest()
    if path.exists():
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata['data_sha256'] != fingerprint:
            raise ValueError('Dataset changed; refusing to reuse split')
        split = pd.read_csv(path, dtype={'essay_id': str})
    else:
        # Group connected components of normalized word-trigram cosine >= .95.
        # Normalization applies to duplicate detection only, never model inputs.
        texts = frame.full_text.str.lower().str.replace(r'\W+', ' ', regex=True).str.strip()
        parent = np.arange(len(frame))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            parent[find(j)] = find(i)

        seen = {}
        for i, text in enumerate(texts):
            if text in seen:
                union(i, seen[text])
            seen[text] = i
        matrix = TfidfVectorizer(ngram_range=(3, 3), binary=True, use_idf=False,
                                 token_pattern=r'(?u)\b\w+\b', dtype=np.float32).fit_transform(texts)
        for start in range(0, len(frame), 128):
            similarity = (matrix[start:start + 128] @ matrix.T).tocoo()
            for a, b in zip(similarity.row[similarity.data >= .95], similarity.col[similarity.data >= .95]):
                if start + a < b:
                    union(start + a, b)
        groups = np.array([find(i) for i in range(len(frame))])
        fold = np.empty(len(frame), dtype=int)
        for k, (_, indices) in enumerate(StratifiedGroupKFold(10, shuffle=True, random_state=42).split(frame, frame.score, groups)):
            fold[indices] = k
        names = np.array(['train'] * 7 + ['selection', 'calibration', 'final_validation'])
        split = pd.DataFrame({'essay_id': frame.essay_id, 'group': groups, 'split': names[fold]})
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
        write_json(path.with_suffix('.json'), {'data_sha256': fingerprint, 'seed': 42,
                   'duplicate_rule': 'connected components: normalized exact match or binary word-trigram cosine >= 0.95',
                   'groups': len(np.unique(groups)), 'counts': split.split.value_counts().to_dict(),
                   'score_counts': pd.crosstab(split.split, frame.score).to_dict(),
                   'historical_exposure': 'Earlier repository experiments used all labels; reserved only for this new experiment.'})
    if split.essay_id.duplicated().any() or set(split.essay_id) != set(frame.essay_id):
        raise ValueError('Split IDs do not match dataset')
    if set(split.split) != {'train', 'selection', 'calibration', 'final_validation'}:
        raise ValueError('Invalid split names')
    if split.groupby('group').split.nunique().max() != 1:
        raise ValueError('Duplicate group leakage')
    return frame.merge(split, on='essay_id', validate='one_to_one')


class OrdinalHead(torch.nn.Module):
    """Proportional-odds head with strictly increasing learned cutpoints."""
    def __init__(self, hidden_size):
        super().__init__()
        self.score = torch.nn.Linear(hidden_size, 1, bias=False)
        self.first_cutpoint = torch.nn.Parameter(torch.tensor(-2.))
        self.raw_gaps = torch.nn.Parameter(torch.full((4,), math.log(math.expm1(1. - 1e-4))))

    def cutpoints(self):
        gaps = torch.nn.functional.softplus(self.raw_gaps.float()) + 1e-4
        return self.first_cutpoint.float() + torch.cat([gaps.new_zeros(1), gaps.cumsum(0)])

    def forward(self, pooled):
        return self.score(pooled).float() - self.cutpoints()[None, :]


def ordinal_targets(labels):
    return (labels[:, None] > torch.arange(1, 6, device=labels.device)[None, :]).float()


class EssayRegressor(torch.nn.Module):
    def __init__(self, source, pretrained=True, pooling=None, ordinal_aux=None):
        super().__init__()
        self.encoder = (AutoModel.from_pretrained(source, local_files_only=True) if pretrained
                        else AutoModel.from_config(AutoConfig.from_pretrained(source, local_files_only=True)))
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)
        self.pooling = pooling or getattr(self.encoder.config, 'essay_pooling', 'mean')
        if self.pooling not in {'mean', 'attention'}:
            raise ValueError(f'Unknown pooling: {self.pooling}')
        self.encoder.config.essay_pooling = self.pooling
        self.encoder_frozen = False
        if self.pooling == 'attention':
            # Zero logits start from exactly mean pooling, without changing the
            # random stream used for the common head or training dropout.
            with torch.random.fork_rng(devices=[]):
                self.attention = torch.nn.Linear(self.encoder.config.hidden_size, 1, bias=False)
                torch.nn.init.zeros_(self.attention.weight)
        self.ordinal_aux = (getattr(self.encoder.config, 'essay_ordinal_aux', False)
                            if ordinal_aux is None else ordinal_aux)
        self.encoder.config.essay_ordinal_aux = self.ordinal_aux
        if self.ordinal_aux:
            # Preserve the shared encoder/head initialization and dropout RNG.
            with torch.random.fork_rng(devices=[]):
                self.ordinal_head = OrdinalHead(self.encoder.config.hidden_size)

    def freeze_encoder(self):
        self.encoder_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, return_ordinal=False, **inputs):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.encoder_frozen):
            hidden = self.encoder(**inputs).last_hidden_state.float()
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        if self.pooling == 'attention':
            logits = self.attention(hidden).float().masked_fill(mask == 0, -1e9)
            weights = logits.softmax(dim=1) * mask
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-9)
            pooled = (hidden * weights).sum(1)
        else:
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        raw = self.head(pooled).squeeze(-1).float()
        if return_ordinal:
            if not self.ordinal_aux:
                raise ValueError('This checkpoint has no ordinal auxiliary head')
            return raw, self.ordinal_head(pooled)
        return raw

    @classmethod
    def load(cls, directory, device='cpu'):
        model = cls(directory, pretrained=False)
        model.load_state_dict(torch.load(Path(directory) / 'model.pt', map_location='cpu', weights_only=True))
        return model.to(device).eval()


def fit_thresholds(y, raw):
    # Deterministic coordinate search on calibration only, with strict ordering.
    thresholds = FIXED.copy()
    best = metrics(y, raw, thresholds)['qwk']
    candidates = np.unique(np.r_[FIXED, np.quantile(raw, np.linspace(0, 1, 501))])
    for _ in range(10):
        changed = False
        for k in range(5):
            lower = thresholds[k - 1] if k else -np.inf
            upper = thresholds[k + 1] if k < 4 else np.inf
            chosen = thresholds[k]
            for value in candidates[(candidates > lower + 1e-6) & (candidates < upper - 1e-6)]:
                trial = thresholds.copy()
                trial[k] = value
                score = metrics(y, raw, trial)['qwk']
                if score is not None and score > best + 1e-12:
                    best, chosen = score, value
            changed |= chosen != thresholds[k]
            thresholds[k] = chosen
        if not changed:
            break
    return thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--model-size', choices=['small', 'base'], default='small')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--batch-size', type=int, choices=[2, 4], default=4)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--pooling', choices=['mean', 'attention'], default='mean')
    parser.add_argument('--encoder-checkpoint', type=Path,
                        help='Completed baseline run; copy its encoder and freshly initialize the scoring head')
    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ordinal-weight', type=float, default=0.,
                        help='Auxiliary cumulative BCE weight; zero keeps regression-only architecture')
    args = parser.parse_args()
    if args.epochs < 1 or args.eval_batch_size < 1 or args.max_length < 2:
        parser.error('epochs and eval batch size must be positive; max length must be at least 2')
    if not math.isfinite(args.ordinal_weight) or args.ordinal_weight < 0:
        parser.error('ordinal weight must be finite and nonnegative')
    if args.ordinal_weight and args.freeze_encoder:
        parser.error('Auxiliary ordinal experiments require a trainable shared encoder')
    torch.set_num_threads(8)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = pd.read_csv(ROOT / 'train.csv', dtype={'essay_id': str})
    if frame.essay_id.duplicated().any() or frame.full_text.isna().any() or not frame.score.isin(range(1, 7)).all():
        raise ValueError('Invalid training data')
    frame = prepare_split(frame, ROOT / 'splits/deberta_seed42.csv')
    if args.prepare_only:
        print(frame.split.value_counts().to_json(), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU access required')
    suffix = '' if args.model_size == 'small' else f'_{args.model_size}'
    length_suffix = '' if args.max_length == 512 else f'_{args.max_length}'
    output = args.output_dir or ROOT / f'outputs_deberta{suffix}{length_suffix}{"_smoke" if args.smoke else ""}'
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists():
        raise ValueError(f'{output} already contains a run; use a fresh output directory')
    snapshot = Path((ROOT / f'models/deberta{suffix}_snapshot.txt').read_text().strip())
    if not snapshot.is_absolute():
        snapshot = ROOT / snapshot
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    data, tokens, lengths = {}, {}, {}
    for name in ['train', 'selection', 'calibration']:
        part = frame[frame.split == name].copy().reset_index(drop=True)
        full = tokenizer(part.full_text.tolist(), truncation=False)['input_ids']
        if args.smoke:
            # Stress actual longest token sequences, including the maximum length.
            indices = np.argsort([-len(ids) for ids in full])[:256 if name == 'train' else 32]
            part = part.iloc[indices].reset_index(drop=True)
            full = [full[i] for i in indices]
        lengths[name] = np.array([len(ids) for ids in full])
        tokens[name] = tokenizer(part.full_text.tolist(), truncation=True, max_length=args.max_length)['input_ids']
        data[name] = part
    bf16 = torch.cuda.is_bf16_supported()
    accumulation = 32 // args.batch_size
    source_checkpoint = None
    source_hash = None
    if args.encoder_checkpoint:
        source_run = args.encoder_checkpoint.resolve()
        source_config = json.loads((source_run / 'config.json').read_text())
        if (source_config['model'] != f'microsoft/deberta-v3-{args.model_size}'
                or source_config['max_length'] != args.max_length
                or source_config['split_sha256'] != hashlib.sha256((ROOT / 'splits/deberta_seed42.csv').read_bytes()).hexdigest()
                or source_config['smoke']):
            raise ValueError('Encoder source must be a complete run with matching model, length and split')
        if not (source_run / 'report.json').exists():
            raise ValueError('Encoder source training is incomplete')
        source_checkpoint = source_run / 'model' / 'model.pt'
        with source_checkpoint.open('rb') as stream:
            source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    config = {**vars(args), 'output_dir': str(output),
              'encoder_checkpoint': str(args.encoder_checkpoint.resolve()) if args.encoder_checkpoint else None,
              'encoder_checkpoint_sha256': source_hash,
              'model': f'microsoft/deberta-v3-{args.model_size}', 'revision': snapshot.name,
              'max_length': args.max_length, 'truncation': 'right', 'gradient_accumulation': accumulation,
              'gradient_checkpointing': not args.freeze_encoder, 'dynamic_padding': True, 'precision': 'bf16' if bf16 else 'fp32',
              'encoder_lr': 0. if args.freeze_encoder else 2e-5, 'head_lr': 1e-4, 'weight_decay': .01, 'warmup_ratio': .1,
              'loss': 'FP32 MSE + ordinal_weight * mean cumulative BCE' if args.ordinal_weight else 'FP32 MSE',
              'prediction_head': 'regression', 'selection_metric': 'regression B0 QWK',
              'training_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'split_sha256': hashlib.sha256((ROOT / 'splits/deberta_seed42.csv').read_bytes()).hexdigest(),
              'lengths': {k: {'rows': len(v), 'truncated': int((v > args.max_length).sum()),
                              'truncated_fraction': float((v > args.max_length).mean()),
                              'max_tokens': int(v.max())} for k, v in lengths.items()}}
    write_json(output / 'config.json', config)
    print(json.dumps(config), flush=True)
    model = EssayRegressor(snapshot, pooling=args.pooling, ordinal_aux=args.ordinal_weight > 0)
    if source_checkpoint:
        source_state = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
        model.encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in source_state.items()
                                       if k.startswith('encoder.')}, strict=True)
        del source_state
    def parameter_hash(prefix):
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if name.startswith(prefix):
                digest.update(name.encode())
                digest.update(parameter.detach().contiguous().numpy().tobytes())
        return digest.hexdigest()

    write_json(output / 'initialization.json', {
        'encoder_sha256': parameter_hash('encoder.'), 'regression_head_sha256': parameter_hash('head.'),
        'ordinal_head_sha256': parameter_hash('ordinal_head.') if model.ordinal_aux else None,
        'cpu_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        'ordinal_cutpoints': model.ordinal_head.cutpoints().detach().tolist() if model.ordinal_aux else None})
    if args.freeze_encoder:
        model.freeze_encoder()
    else:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.cuda()
    groups = [{'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.')], 'lr': 1e-4}]
    if not args.freeze_encoder:
        groups.insert(0, {'params': model.encoder.parameters(), 'lr': 2e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    batches = math.ceil(len(data['train']) / args.batch_size)
    epochs = 1 if args.smoke else args.epochs
    total_steps = math.ceil(batches / accumulation) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * .1), total_steps)

    def inputs(name, indices):
        return tokenizer.pad({'input_ids': [tokens[name][i] for i in indices]}, padding=True, return_tensors='pt').to('cuda')

    def predict(name):
        model.eval()
        order = np.argsort([len(t) for t in tokens[name]])
        raw = np.empty(len(order), dtype=np.float32)
        ordinal_logits = np.empty((len(order), 5), dtype=np.float32) if model.ordinal_aux else None
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            for offset in range(0, len(order), args.eval_batch_size):
                indices = order[offset:offset + args.eval_batch_size]
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                    if model.ordinal_aux:
                        regression, logits = model(return_ordinal=True, **inputs(name, indices))
                        raw[indices] = regression.cpu().numpy()
                        ordinal_logits[indices] = logits.cpu().numpy()
                    else:
                        raw[indices] = model(**inputs(name, indices)).cpu().numpy()
        torch.cuda.synchronize()
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite predictions')
        if ordinal_logits is not None and not np.isfinite(ordinal_logits).all():
            raise ValueError('Nonfinite ordinal predictions')
        return raw, time.monotonic() - start, ordinal_logits

    def ordinal_diagnostics(name, logits):
        if logits is None:
            return None
        values = torch.from_numpy(logits)
        expected = (1 + values.sigmoid().sum(1)).numpy()
        labels = torch.tensor(data[name].score.to_numpy(), dtype=torch.float32)
        return {'B0': metrics(labels.numpy(), expected, FIXED),
                'bce': float(torch.nn.functional.binary_cross_entropy_with_logits(values, ordinal_targets(labels))),
                'cutpoints': model.ordinal_head.cutpoints().detach().cpu().tolist()}

    best, history = -np.inf, []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        start = time.monotonic()
        order = np.random.default_rng(args.seed + epoch).permutation(len(data['train']))
        losses = []
        mse_losses, ordinal_losses = [], []
        optimizer.zero_grad(set_to_none=True)
        for batch in range(batches):
            indices = order[batch * args.batch_size:(batch + 1) * args.batch_size]
            labels = torch.tensor(data['train'].score.to_numpy()[indices], device='cuda', dtype=torch.float32)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                if model.ordinal_aux:
                    raw, ordinal_logits = model(return_ordinal=True, **inputs('train', indices))
                else:
                    raw = model(**inputs('train', indices))
            mse_loss = torch.nn.functional.mse_loss(raw.float(), labels)
            ordinal_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                ordinal_logits.float(), ordinal_targets(labels)) if model.ordinal_aux else mse_loss.new_zeros(()))
            loss = mse_loss + args.ordinal_weight * ordinal_loss if model.ordinal_aux else mse_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            window_start = (batch // accumulation) * accumulation * args.batch_size
            window_samples = min(32, len(order) - window_start)
            (loss * len(indices) / window_samples).backward()
            losses.append(float(loss.detach()))
            mse_losses.append(float(mse_loss.detach()))
            ordinal_losses.append(float(ordinal_loss.detach()))
            if (batch + 1) % accumulation == 0 or batch + 1 == batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch % 100 == 0 or batch + 1 == batches:
                progress = {'epoch': epoch + 1, 'batch': batch + 1, 'batches': batches,
                            'loss': float(np.mean(losses[-100:])), 'seconds': time.monotonic() - start,
                            'mse': float(np.mean(mse_losses[-100:])),
                            'ordinal_bce': float(np.mean(ordinal_losses[-100:])),
                            'weighted_ordinal_bce': args.ordinal_weight * float(np.mean(ordinal_losses[-100:])),
                            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
                write_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        torch.cuda.synchronize()
        train_seconds = time.monotonic() - start
        raw, elapsed, ordinal_logits = predict('selection')
        score = metrics(data['selection'].score, raw, FIXED)
        history.append({'epoch': epoch + 1, 'train_mse': float(np.mean(mse_losses)),
                        'train_ordinal_bce': float(np.mean(ordinal_losses)), 'train_loss': float(np.mean(losses)),
                        'selection_ordinal': ordinal_diagnostics('selection', ordinal_logits), 'train_seconds': train_seconds,
                        'selection_seconds': elapsed, 'selection': score,
                        'estimated_remaining_seconds': (epochs - epoch - 1) * (train_seconds + elapsed)})
        if score['qwk'] > best:
            best, best_epoch = score['qwk'], epoch + 1
            checkpoint = output / 'model'
            checkpoint.mkdir(exist_ok=True)
            model.encoder.config.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            torch.save(model.state_dict(), checkpoint / 'model.pt')
        write_json(output / 'history.json', history)
        print(json.dumps(history[-1]), flush=True)
    del optimizer, scheduler, model
    torch.cuda.empty_cache()
    model = EssayRegressor.load(output / 'model', 'cuda')
    calibration, calibration_time, calibration_ordinal = predict('calibration')
    thresholds = fit_thresholds(data['calibration'].score.to_numpy(), calibration)
    write_json(output / 'thresholds.json', {'thresholds': thresholds.tolist(), 'fit_split': 'calibration',
                                           'method': 'deterministic coordinate QWK search', 'selected_epoch': best_epoch})
    report = {'selected_epoch': best_epoch, 'final_validation_evaluated': False,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30, 'splits': {}}
    for name in ['train', 'selection', 'calibration']:
        raw, elapsed, ordinal_logits = ((calibration, calibration_time, calibration_ordinal)
                                       if name == 'calibration' else predict(name))
        y = data[name].score.to_numpy()
        truncated = lengths[name] > args.max_length
        report['splits'][name] = {'B0': metrics(y, raw, FIXED), 'B1': metrics(y, raw, thresholds),
            'truncated_B0': metrics(y[truncated], raw[truncated], FIXED),
            'truncated_B1': metrics(y[truncated], raw[truncated], thresholds),
            'untruncated_B0': metrics(y[~truncated], raw[~truncated], FIXED),
            'untruncated_B1': metrics(y[~truncated], raw[~truncated], thresholds),
            'inference_seconds': elapsed, 'seconds_per_essay': elapsed / len(raw)}
        predictions = data[name][['essay_id', 'score']].assign(raw_prediction=raw, B0=integer_scores(raw),
            B1=integer_scores(raw, thresholds), token_length=lengths[name], truncated=truncated)
        if ordinal_logits is not None:
            report['splits'][name]['ordinal'] = ordinal_diagnostics(name, ordinal_logits)
            for k in range(5):
                predictions[f'ordinal_logit_gt_{k + 1}'] = ordinal_logits[:, k]
            predictions['ordinal_expected'] = (1 + torch.from_numpy(ordinal_logits).sigmoid().sum(1)).numpy()
        predictions.to_csv(output / f'{name}_predictions.csv', index=False)
    write_json(output / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
