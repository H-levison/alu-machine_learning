#!/usr/bin/env python3
"""Defines the gensim_to_keras function."""


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer.

    Args:
        model: a trained gensim word2vec model.

    Returns:
        The trainable keras Embedding.
    """
    return model.wv.get_keras_embedding(True)
