#!/usr/bin/env python3
"""Defines the baum_welch function."""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Runs the forward algorithm and returns the path probabilities."""
    N = Emission.shape[0]
    T = Observation.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        step = np.matmul(F[:, t - 1], Transition)
        F[:, t] = step * Emission[:, Observation[t]]
    return F


def backward(Observation, Emission, Transition, Initial):
    """Runs the backward algorithm and returns the path probabilities."""
    N = Emission.shape[0]
    T = Observation.shape[0]
    B = np.zeros((N, T))
    B[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        future = Emission[:, Observation[t + 1]] * B[:, t + 1]
        B[:, t] = np.matmul(Transition, future)
    return B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden markov model.

    Expectation-maximization is repeated `iterations` times: the
    forward and backward probabilities give the expected state and
    transition counts, which are then used to re-estimate the
    transition and emission probabilities.

    Args:
        Observations: numpy.ndarray of shape (T,) containing the index
            of each observation.
        Transition: numpy.ndarray of shape (M, M) containing the
            initialized transition probabilities.
        Emission: numpy.ndarray of shape (M, N) containing the
            initialized emission probabilities.
        Initial: numpy.ndarray of shape (M, 1) containing the
            initialized starting probabilities.
        iterations: number of expectation-maximization steps to run.

    Returns:
        the converged Transition, Emission, or None, None on failure.
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    M, N = Emission.shape
    if not isinstance(Transition, np.ndarray) or Transition.shape != (M, M):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (M, 1):
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    if np.any(Observations < 0) or np.any(Observations >= N):
        return None, None

    T = Observations.shape[0]
    if T < 2:
        return Transition, Emission

    Transition = np.copy(Transition)
    Emission = np.copy(Emission)

    for _ in range(iterations):
        F = forward(Observations, Emission, Transition, Initial)
        B = backward(Observations, Emission, Transition, Initial)

        # xi[i, j, t]: probability of being in state i at time t and in
        # state j at time t + 1, given the observations.
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            emitted = Emission[:, Observations[t + 1]] * B[:, t + 1]
            denominator = np.matmul(np.matmul(F[:, t], Transition), emitted)
            if denominator == 0:
                return None, None
            for i in range(M):
                numerator = F[i, t] * Transition[i, :] * emitted
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        occupancy = np.sum(gamma, axis=1).reshape((-1, 1))
        Transition = np.sum(xi, axis=2) / occupancy

        # The emission update also needs the final time step.
        last = F[:, T - 1] * B[:, T - 1]
        last = last / np.sum(last)
        gamma = np.hstack((gamma, last.reshape((-1, 1))))

        counts = np.zeros((M, N))
        for k in range(N):
            counts[:, k] = np.sum(gamma[:, Observations == k], axis=1)
        Emission = counts / np.sum(gamma, axis=1).reshape((-1, 1))

    return Transition, Emission
