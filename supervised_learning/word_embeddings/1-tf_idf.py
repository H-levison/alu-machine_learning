#!/usr/bin/env python3
"""Defines the tf_idf function."""
import re
import numpy as np


def tokenize(sentence):
    """Splits a sentence into lowercase words.

    Words of a single character are dropped, which also removes the
    "s" left behind by possessives such as "children's".

    Args:
        sentence: the sentence to split.

    Returns:
        The list of words of the sentence.
    """
    return re.findall(r'\b\w\w+\b', sentence.lower())


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding.

    The inverse document frequency is smoothed and every embedding is
    normalized to a unit L2 norm.

    Args:
        sentences: list of sentences to analyze.
        vocab: list of the vocabulary words to use for the analysis.
            If None, all words within sentences are used.

    Returns:
        A tuple (embeddings, features) where embeddings is a
        numpy.ndarray of shape (s, f) containing the embeddings, s
        being the number of sentences and f the number of features,
        and features is the list of the features used.
    """
    words = [tokenize(sentence) for sentence in sentences]
    if vocab is None:
        vocab = sorted({word for sentence in words for word in sentence})

    index = {word: i for i, word in enumerate(vocab)}
    counts = np.zeros((len(sentences), len(vocab)))
    for row, sentence in enumerate(words):
        for word in sentence:
            if word in index:
                counts[row, index[word]] += 1

    document_frequency = np.sum(counts > 0, axis=0)
    idf = np.log((1 + len(sentences)) / (1 + document_frequency)) + 1
    embeddings = counts * idf

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms, vocab
