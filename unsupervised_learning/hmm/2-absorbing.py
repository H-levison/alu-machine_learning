#!/usr/bin/env python3
"""Defines the absorbing function."""
import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing.

    A chain is absorbing when it has at least one absorbing state and
    every state can reach an absorbing state in a finite number of
    steps.

    Args:
        P: square 2D numpy.ndarray of shape (n, n) representing the
            standard transition matrix, where P[i, j] is the
            probability of transitioning from state i to state j.

    Returns:
        True if it is absorbing, or False on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]
    if P.shape[1] != n:
        return False
    if not np.all(P >= 0) or not np.allclose(np.sum(P, axis=1), 1):
        return False

    absorbs = np.isclose(np.diag(P), 1)
    if not np.any(absorbs):
        return False
    if np.all(absorbs):
        return True

    # Propagate reachability of an absorbing state through the chain.
    reaches = np.copy(absorbs)
    for _ in range(n):
        updated = reaches | np.any((P > 0) & reaches, axis=1)
        if np.array_equal(updated, reaches):
            break
        reaches = updated
    return bool(np.all(reaches))
