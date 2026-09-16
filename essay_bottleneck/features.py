"""CPU feature caching for an encoder that remains frozen for the entire run."""
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from deberta_baseline import write_json
from .batching import pad_batch


def cache_encoder_features(encoder, tokenizer, tokens, device, bf16, batch_size, progress_path=None):
    if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError('Feature caching requires an eval-mode, fully frozen encoder')
    lengths = np.asarray([len(ids) for ids in tokens])
    expected_bytes = int(lengths.sum()) * encoder.config.hidden_size * 4
    memory_info = Path('/proc/meminfo')
    if memory_info.exists():
        available = next((int(line.split()[1]) * 1024 for line in memory_info.read_text().splitlines()
                          if line.startswith('MemAvailable:')), None)
        if available is not None and expected_bytes > available * 0.7:
            raise ValueError(f'Feature cache needs {expected_bytes / 2**30:.2f} GiB; insufficient free RAM')
    started = time.monotonic()
    features = [None] * len(tokens)
    order = np.argsort(lengths)
    # no_grad creates ordinary tensors that downstream trainable layers may save for backward.
    with torch.no_grad():
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            inputs = pad_batch(tokenizer, tokens, indices, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                hidden = encoder(**inputs).last_hidden_state.float().cpu()
            for row, index in enumerate(indices):
                features[index] = hidden[row, :lengths[index]].clone()
            if progress_path and (offset % (100 * batch_size) == 0 or offset + batch_size >= len(order)):
                write_json(progress_path, {'phase': 'caching_frozen_encoder',
                    'rows': min(offset + batch_size, len(order)), 'total_rows': len(order),
                    'seconds': time.monotonic() - started})
    actual_bytes = sum(value.numel() * value.element_size() for value in features)
    if actual_bytes != expected_bytes:
        raise ValueError('Feature cache shape mismatch')
    digest = hashlib.sha256()
    for value in features:
        digest.update(value.shape[0].to_bytes(8, 'little'))
        digest.update(memoryview(value.numpy()))
    return features, {'rows': len(tokens), 'tokens': int(lengths.sum()), 'dtype': 'float32',
                      'storage': 'CPU RAM, run-local', 'bytes': actual_bytes,
                      'sha256': digest.hexdigest(),
                      'seconds': time.monotonic() - started,
                      'encoder_precision': 'bf16' if bf16 else 'fp32'}


def pad_cached_features(features, indices, device):
    selected = [features[index] for index in indices]
    if not selected:
        raise ValueError('Cannot collate an empty feature batch')
    padded = torch.zeros(len(selected), max(value.shape[0] for value in selected),
                         selected[0].shape[1], dtype=selected[0].dtype)
    for row, value in enumerate(selected):
        padded[row, :value.shape[0]] = value
    return padded.to(device)
