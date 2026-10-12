#!/usr/bin/env python3
"""Defines the kmeans function."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset.

    The centroids are initialized with a multivariate uniform
    distribution. A centroid whose cluster is empty is reinitialized,
    and the algorithm stops early once the centroids stop changing.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer containing the number of clusters.
        iterations: positive integer containing the maximum number of
            iterations to perform.

    Returns:
        A tuple (C, clss) where C is a numpy.ndarray of shape (k, d)
        containing the centroid means and clss is a numpy.ndarray of
        shape (n,) containing the index of the cluster each data point
        belongs to, or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    d = X.shape[1]
    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        previous = C.copy()
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        for j in range(k):
            points = X[clss == j]
            if len(points) == 0:
                C[j] = np.random.uniform(low, high, size=(d,))
            else:
                C[j] = points.mean(axis=0)
        if np.array_equal(previous, C):
            break

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    return C, np.argmin(distances, axis=1)
