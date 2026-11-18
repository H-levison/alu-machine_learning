#!/usr/bin/env python3
"""Defines the regular function."""
import numpy as np


def regular(P):
    """Determines the steady state probabilities of a regular chain.

    A markov chain is regular when some power of its transition matrix
    has only positive entries. The steady state is then the left
    eigenvector of P associated with the eigenvalue 1, normalized so
    that its entries sum to 1.

    Args:
        P: square 2D numpy.ndarray of shape (n, n) representing the
            transition matrix, where P[i, j] is the probability of
            transitioning from state i to state j.

    Returns:
        numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape[1] != n:
        return None
    if not np.all(P >= 0) or not np.allclose(np.sum(P, axis=1), 1):
        return None

    # A chain is regular iff some power of P is strictly positive.
    power = np.copy(P)
    is_regular = False
    for _ in range(n * n + 1):
        if np.all(power > 0):
            is_regular = True
            break
        power = np.matmul(power, P)
    if not is_regular:
        return None

    values, vectors = np.linalg.eig(P.T)
    index = np.argmin(np.abs(values - 1))
    if not np.isclose(values[index], 1):
        return None
    steady = np.real(vectors[:, index])
    total = np.sum(steady)
    if np.isclose(total, 0):
        return None
    return (steady / total).reshape((1, n))
