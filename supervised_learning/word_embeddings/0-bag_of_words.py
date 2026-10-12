#!/usr/bin/env python3
"""Defines the bag_of_words function."""
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


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix.

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
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for row, sentence in enumerate(words):
        for word in sentence:
            if word in index:
                embeddings[row, index[word]] += 1
    return embeddings, vocab
