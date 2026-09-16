"""Deterministic, per-essay features; no corpus statistics or external data."""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def summarize(features, prefix, values):
    values = np.asarray(values if len(values) else [0], dtype=float)
    for name, value in zip(
        ("mean", "std", "min", "max", "median"),
        (values.mean(), values.std(), values.min(), values.max(), np.median(values)),
    ):
        features[f"{prefix}_{name}"] = float(value)


def essay_features(text):
    words = WORD.findall(text.lower())
    counts = Counter(words)
    n = len(words)
    denominator = max(n, 1)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if WORD.search(s)]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if WORD.search(p)]
    features = {
        "characters": len(text), "words": n, "unique_words": len(counts),
        "sentences": len(sentences), "paragraphs": len(paragraphs),
        "type_token_ratio": len(counts) / denominator,
        "unique_sqrt_words": len(counts) / np.sqrt(denominator),
        "hapax_ratio": sum(c == 1 for c in counts.values()) / denominator,
        "uppercase_ratio": sum(c.isupper() for c in text) / max(len(text), 1),
        "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
        "whitespace_ratio": sum(c.isspace() for c in text) / max(len(text), 1),
        "lexical_entropy": -sum((c / denominator) * np.log(c / denominator) for c in counts.values()),
    }
    summarize(features, "word_length", [len(w) for w in words])
    summarize(features, "sentence_words", [len(WORD.findall(s)) for s in sentences])
    summarize(features, "paragraph_words", [len(WORD.findall(p)) for p in paragraphs])
    for length in (4, 6, 8, 10):
        features[f"words_at_least_{length}_ratio"] = sum(len(w) >= length for w in words) / denominator
    for name, punctuation in {
        "comma": ",", "period": ".", "semicolon": ";", "colon": ":",
        "question": "?", "exclamation": "!", "quote": '"', "apostrophe": "'",
    }.items():
        features[f"{name}_count"] = text.count(punctuation)
        features[f"{name}_per_word"] = text.count(punctuation) / denominator
    for name, vocabulary in {
        "first_person": {"i", "me", "my", "mine", "we", "us", "our"},
        "second_person": {"you", "your", "yours"},
        "connectives": {"because", "therefore", "however", "although", "moreover", "furthermore", "thus", "hence", "consequently", "nevertheless"},
        "modals": {"can", "could", "should", "would", "must", "might", "may"},
    }.items():
        features[f"{name}_ratio"] = sum(counts[w] for w in vocabulary) / denominator
    return features


class EssayFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame([essay_features(text) for text in X]).astype(np.float32)
