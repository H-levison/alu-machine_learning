#!/usr/bin/env python3
"""Defines the epsilon_greedy function."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy to determine the next action.

    A value p is sampled uniformly. If it is greater than epsilon the
    best known action is exploited, otherwise a random action is
    chosen to explore.

    Args:
        Q: numpy.ndarray containing the q-table.
        state: the current state.
        epsilon: the epsilon to use for the calculation.

    Returns:
        the next action index.
    """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        return np.argmax(Q[state, :])
    return np.random.randint(0, Q.shape[1])
