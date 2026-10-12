#!/usr/bin/env python3
"""Defines the expectation_maximization function."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Performs the expectation maximization for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        k: positive integer containing the number of clusters.
        iterations: positive integer containing the maximum number of
            iterations of the algorithm.
        tol: non-negative float containing the tolerance of the log
            likelihood, used to determine early stopping.
        verbose: boolean that determines whether information about the
            algorithm is printed every 10 iterations and after the
            last one.

    Returns:
        A tuple (pi, m, S, g, l) with the priors of shape (k,), the
        centroid means of shape (k, d), the covariance matrices of
        shape (k, d, d), the probabilities of shape (k, n) and the log
        likelihood of the model, or (None, None, None, None, None) on
        failure.
    """
    failure = (None, None, None, None, None)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return failure
    if type(k) is not int or k <= 0:
        return failure
    if type(iterations) is not int or iterations <= 0:
        return failure
    if type(tol) not in (int, float) or tol < 0:
        return failure
    if type(verbose) is not bool:
        return failure

    message = "Log Likelihood after {} iterations: {}"
    pi, m, S = initialize(X, k)
    g, likelihood = expectation(X, pi, m, S)
    count = 0
    for i in range(iterations):
        if verbose and i % 10 == 0:
            print(message.format(i, likelihood.round(5)))
        pi, m, S = maximization(X, g)
        g, updated = expectation(X, pi, m, S)
        converged = abs(updated - likelihood) <= tol
        likelihood = updated
        count = i + 1
        if converged:
            break

    if verbose:
        print(message.format(count, likelihood.round(5)))
    return pi, m, S, g, likelihood
