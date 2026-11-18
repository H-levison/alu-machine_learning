#!/usr/bin/env python3
"""Defines the viterbi function."""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of hidden states.

    Args:
        Observation: numpy.ndarray of shape (T,) containing the index
            of each observation.
        Emission: numpy.ndarray of shape (N, M) containing the emission
            probability of an observation given a hidden state.
        Transition: numpy.ndarray of shape (N, N) containing the
            transition probabilities.
        Initial: numpy.ndarray of shape (N, 1) containing the
            probability of starting in a particular hidden state.

    Returns:
        path, P, or None, None on failure. path is a list of length T
        containing the most likely sequence of hidden states and P is
        the probability of obtaining that sequence.
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return None, None
    if np.any(Observation < 0) or np.any(Observation >= M):
        return None, None
    if not np.allclose(np.sum(Emission, axis=1), 1):
        return None, None
    if not np.allclose(np.sum(Transition, axis=1), 1):
        return None, None
    if not np.isclose(np.sum(Initial), 1):
        return None, None

    T = Observation.shape[0]
    V = np.zeros((N, T))
    back = np.zeros((N, T), dtype=int)
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        steps = V[:, t - 1].reshape((-1, 1)) * Transition
        back[:, t] = np.argmax(steps, axis=0)
        V[:, t] = np.max(steps, axis=0) * Emission[:, Observation[t]]

    state = int(np.argmax(V[:, -1]))
    path = [state]
    for t in range(T - 1, 0, -1):
        state = int(back[state, t])
        path.insert(0, state)
    return path, np.max(V[:, -1])
