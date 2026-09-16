"""CPU checks for full-text, batch-adaptive lengths and saved length policies."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import DebertaV2Model, PreTrainedTokenizerFast

from deberta_baseline import ROOT
from .batching import encode_texts, pad_batch
from .model import BottleneckConfig, BottleneckRegressor
from .score import BottleneckScorer
from .verify import require, tiny_encoder


def main():
    torch.set_num_threads(2)
    backend = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, 'word': 2}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                        unk_token='[UNK]', model_max_length=64)
    texts = [' '.join(['word'] * size) for size in [7, 19, 600]]
    tokens, lengths, truncated = encode_texts(tokenizer, texts)
    np.testing.assert_array_equal(lengths, [7, 19, 600])
    require(not truncated.any() and len(tokens[-1]) == 600, 'Default unexpectedly truncates')
    require(pad_batch(tokenizer, tokens, [0, 1], 'cpu')['input_ids'].shape == (2, 19), 'First batch width')
    require(pad_batch(tokenizer, tokens, [2], 'cpu')['input_ids'].shape == (1, 600), 'Second batch width')
    capped, _, flags = encode_texts(tokenizer, texts, 16)
    require([len(ids) for ids in capped] == [7, 16, 16], 'Explicit cap ignored')
    np.testing.assert_array_equal(flags, [False, True, True])

    config = tiny_encoder().config
    config.position_biased_input = False
    encoder = DebertaV2Model(config)
    model = BottleneckRegressor(encoder, BottleneckConfig(latent_dim=8, num_heads=2,
                                num_latents=2, reconstruction_weight=0.)).eval()
    with torch.inference_mode():
        result = model(**pad_batch(tokenizer, tokens, [2], 'cpu'))
    require(torch.isfinite(result['scores']).all(), 'Long relative-position input failed')
    absolute = BottleneckRegressor(tiny_encoder(), BottleneckConfig(reconstruction_weight=0.))
    try:
        absolute(**pad_batch(tokenizer, tokens, [2], 'cpu'))
    except ValueError as error:
        require('absolute position' in str(error), 'Wrong absolute-position error')
    else:
        raise AssertionError('Unsupported absolute-position length accepted')

    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bottleneck-lengths-', dir=artifacts) as directory:
        directory = Path(directory)
        source, output = directory / 'source', directory / 'run'
        encoder.save_pretrained(source)
        tokenizer.save_pretrained(source)
        command = [sys.executable, '-m', 'essay_bottleneck.train', '--source', str(source),
                   '--output-dir', str(output), '--device', 'cpu', '--smoke', '--batch-size', '2',
                   '--effective-batch-size', '8', '--latent-dim', '8', '--num-heads', '2', '--num-latents', '2']
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        if run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
        saved = json.loads((output / 'config.json').read_text())
        require(saved['max_length'] is None and saved['padding'] == 'batch_longest', 'Saved length policy')
        require(saved['truncation'] == 'none', 'Saved truncation policy')
        require(all(v['truncated'] == 0 for v in saved['lengths'].values()), 'Training truncated essays')
        selection = pd.read_csv(output / 'selection_predictions.csv')
        require(not selection.truncated.any(), 'Prediction truncation flags')
        frame = pd.read_csv(ROOT / 'train.csv').set_index('essay_id')
        scorer = BottleneckScorer(output)
        raw, scores = scorer.predict(frame.loc[selection.essay_id, 'full_text'].tolist())
        np.testing.assert_allclose(raw, selection.raw_prediction, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(scores, selection.B1)
        observed = []
        hook = scorer.model.encoder.register_forward_pre_hook(
            lambda module, args, kwargs: observed.append(kwargs['input_ids'].shape[1]), with_kwargs=True)
        scorer.predict(texts, batch_size=2)
        hook.remove()
        require(observed == [19, 600], 'Reloaded scorer lost dynamic full-text lengths')
    print(json.dumps({'status': 'passed', 'device': 'cpu',
                      'batch_widths': observed, 'explicit_cap_checked': True,
                      'uncapped_training_and_reload_checked': True,
                      'quality_evaluation': False}, indent=2))


if __name__ == '__main__':
    main()
