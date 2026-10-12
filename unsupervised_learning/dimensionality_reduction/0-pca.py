#!/usr/bin/env python3
"""Defines the pca function that keeps a fraction of the variance."""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset, where
            n is the number of data points and d is the number of
            dimensions in each point. All dimensions have a mean of 0
            across all data points.
        var: the fraction of the variance that the PCA transformation
            should maintain.

    Returns:
        The weights matrix W, a numpy.ndarray of shape (d, nd) where
        nd is the new dimensionality of the transformed X, that
        maintains var fraction of the original variance of X.
    """
    _, s, vh = np.linalg.svd(X)
    cumulative = np.cumsum(s) / np.sum(s)
    nd = np.argmax(cumulative >= var) + 1
    return vh.T[:, :nd]
