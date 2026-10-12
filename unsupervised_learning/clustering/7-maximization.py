#!/usr/bin/env python3
"""Defines the maximization function."""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        g: numpy.ndarray of shape (k, n) containing the posterior
            probabilities of each data point in each cluster.

    Returns:
        A tuple (pi, m, S) where pi is a numpy.ndarray of shape (k,)
        containing the updated priors, m is a numpy.ndarray of shape
        (k, d) containing the updated centroid means and S is a
        numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices, or (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    if g.shape[1] != n:
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    k = g.shape[0]
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        weight = np.sum(g[i])
        pi[i] = weight / n
        m[i] = g[i] @ X / weight
        difference = X - m[i]
        S[i] = (g[i] * difference.T) @ difference / weight
    return pi, m, S
