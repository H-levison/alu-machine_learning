#!/usr/bin/env python3
"""Defines the uni_bleu function."""
import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence.

    Args:
        references: list of reference translations, each one being a
            list of the words in the translation.
        sentence: list containing the model proposed sentence.

    Returns:
        The unigram BLEU score.
    """
    length = len(sentence)
    matches = 0
    for word in set(sentence):
        limit = max(reference.count(word) for reference in references)
        matches += min(sentence.count(word), limit)
    precision = matches / length

    closest = min((len(reference) for reference in references),
                  key=lambda size: (abs(size - length), size))
    if length > closest:
        penalty = 1
    else:
        penalty = np.exp(1 - closest / length)
    return penalty * precision
