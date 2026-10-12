#!/usr/bin/env python3
"""Defines the expectation function."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        pi: numpy.ndarray of shape (k,) containing the priors of each
            cluster.
        m: numpy.ndarray of shape (k, d) containing the centroid means
            of each cluster.
        S: numpy.ndarray of shape (k, d, d) containing the covariance
            matrices of each cluster.

    Returns:
        A tuple (g, l) where g is a numpy.ndarray of shape (k, n)
        containing the posterior probabilities of each data point in
        each cluster and l is the total log likelihood, or
        (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if not isinstance(m, np.ndarray) or m.shape != (k, d):
        return None, None
    if not isinstance(S, np.ndarray) or S.shape != (k, d, d):
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    total = np.sum(g, axis=0)
    return g / total, np.sum(np.log(total))
