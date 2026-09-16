"""Offline CPU verification with a tiny random DeBERTa; not a quality benchmark."""
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
from transformers import DebertaV2Config, DebertaV2Model, PreTrainedTokenizerFast

from deberta_baseline import ROOT
from .model import BottleneckConfig, BottleneckRegressor
from .score import BottleneckScorer


def tiny_encoder():
    return DebertaV2Model(DebertaV2Config(vocab_size=32, hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=64, relative_attention=True, position_buckets=16,
        max_relative_positions=64, hidden_dropout_prob=0., attention_probs_dropout_prob=0.,
        pad_token_id=0))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    inputs = dict(input_ids=torch.tensor([[2, 3, 4, 5], [6, 7, 0, 0]]),
                  attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]))
    labels = torch.tensor([2., 5.])
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    checks = []
    with tempfile.TemporaryDirectory(prefix='bottleneck-check-', dir=artifacts) as temporary:
        root = Path(temporary)
        for finetune in [False, True]:
            model = BottleneckRegressor(tiny_encoder(), BottleneckConfig(num_latents=2,
                latent_dim=8, num_heads=2, dropout=0., finetune_encoder=finetune))
            model.train()
            require(model.encoder.training == finetune, 'Frozen encoder must remain in eval mode')
            if finetune:
                require(not model.teacher.training, 'Teacher dropout must remain disabled')
                teacher_before = {k: v.clone() for k, v in model.teacher.state_dict().items()}
            encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
            result = model(**inputs, labels=labels)
            require(torch.isfinite(result['loss']).item(), 'Finite training loss')
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
            result['loss'].backward()
            require(model.readout.queries.grad.abs().sum().item() > 0, 'Readout needs gradients')
            require(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters()),
                    'Decoder needs gradients')
            require(any(p.grad is not None for p in model.encoder.parameters()) == finetune,
                    'Encoder gradient boundary')
            if finetune:
                require(all(p.grad is None for p in model.teacher.parameters()), 'Teacher got gradients')
            optimizer.step()
            if finetune:
                require(all(torch.equal(v, teacher_before[k]) for k, v in model.teacher.state_dict().items()),
                        'Teacher changed during optimizer step')
                require(any(not torch.equal(v, encoder_before[k]) for k, v in model.encoder.state_dict().items()),
                        'Trainable encoder failed to update')
            else:
                require(all(torch.equal(v, encoder_before[k]) for k, v in model.encoder.state_dict().items()),
                        'Frozen encoder changed')
            model.eval()
            padded = dict(input_ids=torch.nn.functional.pad(inputs['input_ids'], (0, 3), value=17),
                          attention_mask=torch.nn.functional.pad(inputs['attention_mask'], (0, 3)))
            with torch.no_grad():
                reference = model(**inputs, labels=labels)
                extended = model(**padded, labels=labels)
            torch.testing.assert_close(reference['scores'], extended['scores'], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(reference['reconstruction_loss'], extended['reconstruction_loss'],
                                       atol=1e-6, rtol=1e-5)
            model.zero_grad(set_to_none=True)
            model(**inputs, labels=labels)['reconstruction_loss'].backward()
            require(model.readout.queries.grad.abs().sum().item() > 0, 'Auxiliary loss bypasses latents')
            require(all(p.grad is None for p in model.head.parameters()), 'Reconstruction must not use score head')
            checkpoint = root / f'model-{finetune}'
            model.save(checkpoint)
            restored = BottleneckRegressor.load(checkpoint)
            require(restored.teacher is None, 'Deployment unnecessarily includes teacher')

            def forbid_decoder(*args):
                raise AssertionError('Decoder used at inference')

            restored.decoder.register_forward_pre_hook(forbid_decoder)
            with torch.inference_mode():
                torch.testing.assert_close(restored(**inputs)['scores'], reference['scores'], atol=0, rtol=0)
            checks.append(f'finetune={finetune}: gradient boundaries, fixed teacher, padding, reload, inference')
        try:
            model(inputs['input_ids'], torch.zeros_like(inputs['attention_mask']))
        except ValueError:
            checks.append('all-padding input rejected')
        else:
            raise AssertionError('Empty mask accepted')
        for pooling in ['mean', 'latent']:
            control = BottleneckRegressor(tiny_encoder(), BottleneckConfig(pooling=pooling,
                latent_dim=8, num_heads=2, reconstruction_weight=0.))
            require(control.decoder is None and control.teacher is None, 'Disabled branch allocated')
            control(**inputs, labels=labels)['loss'].backward()
        checks.append('mean and latent score-only controls')

        # Exercise the real CLI and fixed data split using a deliberately tiny
        # random encoder/tokenizer, never the main thread's trained checkpoints.
        source = root / 'tiny-source'
        tiny_encoder().save_pretrained(source)
        tokenizer = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, 'the': 2, 'a': 3,
                                         'I': 4, 'and': 5, 'is': 6, '.': 7}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]',
                                 unk_token='[UNK]', model_max_length=64).save_pretrained(source)
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        for mode in ['reconstruction', 'score_only', 'mean']:
            output = root / mode
            command = [sys.executable, '-m', 'essay_bottleneck.train', '--source', str(source),
                       '--output-dir', str(output), '--device', 'cpu', '--smoke', '--max-length', '32',
                       '--batch-size', '4', '--effective-batch-size', '8',
                       '--latent-dim', '8', '--num-heads', '2', '--num-latents', '2']
            if mode != 'reconstruction':
                command += ['--reconstruction-weight', '0', '--finetune-encoder']
            if mode == 'mean':
                command += ['--pooling', 'mean']
            run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(run.stdout + run.stderr)
            report = json.loads((output / 'report.json').read_text())
            require(not report['final_validation_evaluated'], 'Unexpected final-validation access')
            require(set(report['splits']) == {'selection', 'calibration'}, 'Unexpected evaluated split')
            selection = pd.read_csv(output / 'selection_predictions.csv')
            splits = pd.read_csv(ROOT / 'splits/deberta_seed42.csv')
            require(set(selection.essay_id) <= set(splits.loc[splits.split == 'selection', 'essay_id']),
                    'Selection membership mismatch')
            texts = pd.read_csv(ROOT / 'train.csv').set_index('essay_id').loc[selection.essay_id, 'full_text'].tolist()
            scorer = BottleneckScorer(output)
            raw, scores = scorer.predict(texts)
            np.testing.assert_allclose(raw, selection.raw_prediction, rtol=0, atol=1e-7)
            np.testing.assert_array_equal(scores, selection.B1)
            for invalid in ['', '   ']:
                try:
                    scorer.predict(invalid)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Blank essay accepted')
            repeat = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
            require(repeat.returncode != 0 and 'Refusing to overwrite' in repeat.stderr, 'Existing run overwritten')
            checks.append(f'{mode}: CLI training, saved predictions, scoring reload, overwrite protection')
    print(json.dumps({'status': 'passed', 'device': 'cpu', 'checks': checks,
                      'quality_evaluation': False, 'formal_training_started': False}, indent=2))


if __name__ == '__main__':
    main()
