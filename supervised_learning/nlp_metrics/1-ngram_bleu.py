#!/usr/bin/env python3
"""Defines the ngram_bleu function."""
import numpy as np


def ngrams(words, n):
    """Lists the n-grams of a sentence.

    Args:
        words: list of the words in the sentence.
        n: the size of the n-grams.

    Returns:
        The list of n-grams, each one being a tuple of n words.
    """
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations, each one being a
            list of the words in the translation.
        sentence: list containing the model proposed sentence.
        n: the size of the n-gram to use for evaluation.

    Returns:
        The n-gram BLEU score.
    """
    candidates = ngrams(sentence, n)
    reference_grams = [ngrams(reference, n) for reference in references]
    matches = 0
    for gram in set(candidates):
        limit = max(grams.count(gram) for grams in reference_grams)
        matches += min(candidates.count(gram), limit)
    precision = matches / len(candidates) if candidates else 0

    length = len(sentence)
    closest = min((len(reference) for reference in references),
                  key=lambda size: (abs(size - length), size))
    if length > closest:
        penalty = 1
    else:
        penalty = np.exp(1 - closest / length)
    return penalty * precision
