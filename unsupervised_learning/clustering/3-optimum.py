#!/usr/bin/env python3
"""Defines the optimum_k function."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: positive integer containing the minimum number of
            clusters to check for (inclusive).
        kmax: positive integer containing the maximum number of
            clusters to check for (inclusive). Defaults to n.
        iterations: positive integer containing the maximum number of
            iterations for K-means.

    Returns:
        A tuple (results, d_vars) where results is a list containing
        the outputs of K-means for each cluster size and d_vars is a
        list containing the difference in variance from the smallest
        cluster size for each cluster size, or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmin) is not int or kmin <= 0:
        return None, None
    if type(kmax) is not int or kmax <= 0 or kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    results = []
    variances = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        variances.append(variance(X, C))

    d_vars = (variances[0] - np.array(variances)).tolist()
    return results, d_vars
