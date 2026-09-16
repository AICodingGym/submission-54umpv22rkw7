"""Score essays with a saved bottleneck checkpoint; no reconstruction at inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from deberta_baseline import FIXED, integer_scores
from .model import BottleneckRegressor
from .batching import encode_texts, pad_batch


class BottleneckScorer:
    def __init__(self, directory, device='cpu', version='B1'):
        directory = Path(directory)
        config = json.loads((directory / 'config.json').read_text())
        self.max_length = config['max_length']
        self.device = torch.device(device)
        self.bf16 = (self.device.type == 'cuda' and config['precision'] == 'bf16'
                     and torch.cuda.is_bf16_supported())
        if version not in {'B0', 'B1'}:
            raise ValueError('version must be B0 or B1')
        self.thresholds = (np.asarray(json.loads((directory / 'thresholds.json').read_text())['thresholds'])
                           if version == 'B1' else FIXED)
        if (self.thresholds.shape != (5,) or not np.isfinite(self.thresholds).all()
                or not (np.diff(self.thresholds) > 0).all()):
            raise ValueError('Expected five finite, strictly increasing thresholds')
        self.tokenizer = AutoTokenizer.from_pretrained(directory / 'model', local_files_only=True)
        if self.tokenizer.padding_side != 'right':
            raise ValueError('Only right padding is supported')
        self.model = BottleneckRegressor.load(directory / 'model', self.device)

    def predict(self, texts, batch_size=4):
        texts = [texts] if isinstance(texts, str) else list(texts)
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Provide nonempty essay text')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        tokens, _, _ = encode_texts(self.tokenizer, texts, self.max_length)
        order = np.argsort([len(ids) for ids in tokens])
        raw = np.empty(len(texts), dtype=np.float32)
        with torch.inference_mode():
            for offset in range(0, len(order), batch_size):
                indices = order[offset:offset + batch_size]
                inputs = pad_batch(self.tokenizer, tokens, indices, self.device)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                    raw[indices] = self.model(**inputs)['scores'].cpu().numpy()
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite predictions')
        return raw, integer_scores(raw, self.thresholds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--version', choices=['B0', 'B1'], default='B1')
    parser.add_argument('--batch-size', type=int, default=4)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--text')
    source.add_argument('--file', type=Path)
    source.add_argument('--csv', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.csv:
        frame = pd.read_csv(args.csv, dtype={'essay_id': str}, keep_default_na=False)
        if not {'essay_id', 'full_text'}.issubset(frame.columns):
            parser.error('CSV must contain essay_id and full_text')
        if not frame.essay_id.is_unique or frame.essay_id.str.strip().eq('').any():
            parser.error('Essay IDs must be nonempty and unique')
        texts = frame.full_text.tolist()
    else:
        texts = [args.file.read_text(encoding='utf-8') if args.file else args.text]
    raw, scores = BottleneckScorer(args.model_dir, args.device, args.version).predict(texts, args.batch_size)
    if args.csv:
        print(frame[['essay_id']].assign(raw_prediction=raw, score=scores).to_csv(index=False), end='')
    else:
        print(json.dumps({'raw_prediction': float(raw[0]), 'score': int(scores[0]), 'version': args.version}))


if __name__ == '__main__':
    main()
