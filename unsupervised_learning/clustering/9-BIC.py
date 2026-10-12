#!/usr/bin/env python3
"""Defines the BIC function."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: positive integer containing the minimum number of
            clusters to check for (inclusive).
        kmax: positive integer containing the maximum number of
            clusters to check for (inclusive). Defaults to n.
        iterations: positive integer containing the maximum number of
            iterations of the EM algorithm.
        tol: non-negative float containing the tolerance of the EM
            algorithm.
        verbose: boolean that determines whether the EM algorithm
            prints information to the standard output.

    Returns:
        A tuple (best_k, best_result, l, b) where best_k is the best
        value of k based on its BIC, best_result is the tuple
        (pi, m, S) of the best number of clusters, l is a
        numpy.ndarray containing the log likelihood of each cluster
        size tested and b is a numpy.ndarray containing the BIC value
        of each cluster size tested, or (None, None, None, None) on
        failure.
    """
    failure = (None, None, None, None)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return failure
    n, d = X.shape
    if kmax is None:
        kmax = n
    if type(kmin) is not int or kmin <= 0:
        return failure
    if type(kmax) is not int or kmax <= 0 or kmin >= kmax:
        return failure
    if type(iterations) is not int or iterations <= 0:
        return failure
    if type(tol) not in (int, float) or tol < 0:
        return failure
    if type(verbose) is not bool:
        return failure

    results = []
    likelihoods = []
    bics = []
    for k in range(kmin, kmax + 1):
        pi, m, S, _, likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return failure
        parameters = k * d + k * d * (d + 1) / 2 + k - 1
        results.append((pi, m, S))
        likelihoods.append(likelihood)
        bics.append(parameters * np.log(n) - 2 * likelihood)

    best = int(np.argmin(bics))
    return (kmin + best, results[best], np.array(likelihoods),
            np.array(bics))
