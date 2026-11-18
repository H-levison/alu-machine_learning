#!/usr/bin/env python3
"""Defines the forward function."""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden markov model.

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
        P, F, or None, None on failure. P is the likelihood of the
        observations given the model and F is a numpy.ndarray of shape
        (N, T) containing the forward path probabilities.
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
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        step = np.matmul(F[:, t - 1], Transition)
        F[:, t] = step * Emission[:, Observation[t]]
    return np.sum(F[:, -1]), F
