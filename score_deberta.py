"""Reload the trained DeBERTa baseline and score new English essays offline."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from deberta_baseline import EssayRegressor, FIXED, ROOT, integer_scores


class DebertaScorer:
    def __init__(self, directory=ROOT / 'outputs_deberta', device='cpu', version='B1'):
        directory = Path(directory)
        config = json.loads((directory / 'config.json').read_text())
        self.max_length = config['max_length']
        self.device = torch.device(device)
        self.bf16 = (self.device.type == 'cuda' and config['precision'] == 'bf16'
                     and torch.cuda.is_bf16_supported())
        if version not in {'B0', 'B1'}:
            raise ValueError('version must be B0 or B1')
        self.thresholds = np.asarray(json.loads((directory / 'thresholds.json').read_text())['thresholds']) if version == 'B1' else FIXED
        if self.thresholds.shape != (5,) or not np.isfinite(self.thresholds).all() or not (np.diff(self.thresholds) > 0).all():
            raise ValueError('Expected five finite, strictly increasing thresholds')
        self.tokenizer = AutoTokenizer.from_pretrained(directory / 'model', local_files_only=True)
        self.model = EssayRegressor.load(directory / 'model', self.device)

    def predict(self, texts, batch_size=4):
        texts = [texts] if isinstance(texts, str) else list(texts)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Provide non-empty essay text')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        raw = np.empty(len(texts), dtype=np.float32)
        # Match training evaluation's length ordering and dynamic padding.
        tokens = self.tokenizer(texts, truncation=True, max_length=self.max_length)['input_ids']
        order = np.argsort([len(t) for t in tokens])
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                indices = order[start:start + batch_size]
                inputs = self.tokenizer.pad({'input_ids': [tokens[i] for i in indices]},
                                            padding=True, return_tensors='pt').to(self.device)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                    raw[indices] = self.model(**inputs).cpu().numpy()
        if not np.isfinite(raw).all():
            raise ValueError('Model produced non-finite predictions')
        return raw, integer_scores(raw, self.thresholds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--text')
    source.add_argument('--file', type=Path, help='UTF-8 essay file')
    source.add_argument('--csv', type=Path, help='CSV with essay_id and full_text')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'outputs_deberta')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--version', choices=['B0', 'B1'], default='B1')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(8)
    try:
        if args.csv:
            frame = pd.read_csv(args.csv, dtype={'essay_id': str}, keep_default_na=False)
            if not {'essay_id', 'full_text'}.issubset(frame.columns):
                raise ValueError('CSV needs essay_id and full_text')
            if frame.essay_id.str.strip().eq('').any() or not frame.essay_id.is_unique:
                raise ValueError('Essay IDs must be non-empty and unique')
            texts = frame.full_text.tolist()
        else:
            texts = [args.file.read_text(encoding='utf-8') if args.file else args.text]
        scorer = DebertaScorer(args.model_dir, args.device, args.version)
        raw, scores = scorer.predict(texts, args.batch_size)
        if args.csv:
            print(frame[['essay_id']].assign(raw_prediction=raw, score=scores).to_csv(index=False), end='')
        else:
            print(json.dumps({'raw_prediction': float(raw[0]), 'score': int(scores[0]), 'version': args.version}))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
