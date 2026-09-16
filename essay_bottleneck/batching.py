"""Shared full-text tokenization and per-batch dynamic padding."""
import numpy as np


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
