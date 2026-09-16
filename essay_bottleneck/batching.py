"""Shared full-text tokenization and per-batch dynamic padding."""
import numpy as np


def group_within_effective_batches(order, lengths, effective_batch_size):
    """Reduce microbatch padding without changing each optimizer batch's members."""
    if effective_batch_size < 1:
        raise ValueError('effective_batch_size must be positive')
    order, lengths = np.asarray(order), np.asarray(lengths)
    if not len(order):
        return order.copy()
    windows = [order[start:start + effective_batch_size]
               for start in range(0, len(order), effective_batch_size)]
    return np.concatenate([window[np.argsort(lengths[window], kind='stable')]
                           for window in windows])


def encode_texts(tokenizer, texts, max_length=None):
    """Keep every token unless the caller explicitly requests a length cap."""
    if max_length is not None and max_length < 2:
        raise ValueError('An explicit max_length must be at least 2')
    full = tokenizer(texts, truncation=False, padding=False)['input_ids']
    lengths = np.array([len(ids) for ids in full], dtype=np.int64)
    tokens = (full if max_length is None else
              tokenizer(texts, truncation=True, max_length=max_length, padding=False)['input_ids'])
    truncated = lengths > np.array([len(ids) for ids in tokens], dtype=np.int64)
    return tokens, lengths, truncated


def pad_batch(tokenizer, tokens, indices, device):
    """Each microbatch uses its own longest sequence, with no global padding."""
    return tokenizer.pad({'input_ids': [tokens[i] for i in indices]}, padding='longest',
                         return_tensors='pt').to(device)
