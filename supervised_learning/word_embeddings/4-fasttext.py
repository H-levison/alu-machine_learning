#!/usr/bin/env python3
"""Defines the fasttext_model function."""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Creates and trains a gensim fastText model.

    Args:
        sentences: list of sentences to be trained on.
        size: the dimensionality of the embedding layer.
        min_count: the minimum number of occurrences of a word for use
            in training.
        negative: the size of negative sampling.
        window: the maximum distance between the current and predicted
            word within a sentence.
        cbow: True to train with CBOW, False to train with Skip-gram.
        iterations: the number of iterations to train over.
        seed: the seed for the random number generator.
        workers: the number of worker threads to train the model.

    Returns:
        The trained model.
    """
    model = FastText(size=size, min_count=min_count, window=window,
                     negative=negative, sg=0 if cbow else 1,
                     iter=iterations, seed=seed, workers=workers)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
