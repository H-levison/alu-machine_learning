#!/usr/bin/env python3
"""Defines the variance function."""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        C: numpy.ndarray of shape (k, d) containing the centroid means
            of each cluster.

    Returns:
        The total variance, or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    return np.sum(np.min(distances, axis=1) ** 2)
