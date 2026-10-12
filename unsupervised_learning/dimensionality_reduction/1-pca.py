#!/usr/bin/env python3
"""Defines the pca function that reduces to a given dimensionality."""
import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset, where
            n is the number of data points and d is the number of
            dimensions in each point.
        ndim: the new dimensionality of the transformed X.

    Returns:
        T, a numpy.ndarray of shape (n, ndim) containing the
        transformed version of X.
    """
    centered = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(centered)
    return centered @ vh.T[:, :ndim]
