#!/usr/bin/env python3
"""Defines the markov_chain function."""
import numpy as np


def markov_chain(P, s, t=1):
    """Determines the probability of a markov chain being in a state.

    Args:
        P: square 2D numpy.ndarray of shape (n, n) representing the
            transition matrix, where P[i, j] is the probability of
            transitioning from state i to state j.
        s: numpy.ndarray of shape (1, n) representing the probability
            of starting in each state.
        t: number of iterations the markov chain has been through.

    Returns:
        numpy.ndarray of shape (1, n) representing the probability of
        being in a specific state after t iterations, or None on
        failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape[1] != n:
        return None
    if not isinstance(s, np.ndarray) or s.shape != (1, n):
        return None
    if not isinstance(t, int) or t < 0:
        return None
    if not np.all(P >= 0) or not np.allclose(np.sum(P, axis=1), 1):
        return None

    state = s
    for _ in range(t):
        state = np.matmul(state, P)
    return state
