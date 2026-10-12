#!/usr/bin/env python3
"""Defines the initialize function for K-means."""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means.

    The centroids are drawn from a multivariate uniform distribution
    bounded by the minimum and maximum values of X along each
    dimension.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer containing the number of clusters.

    Returns:
        A numpy.ndarray of shape (k, d) containing the initialized
        centroids, or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if type(k) is not int or k <= 0:
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)
    return np.random.uniform(low, high, size=(k, X.shape[1]))
