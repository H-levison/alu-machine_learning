#!/usr/bin/env python3
"""Defines the policy and policy_gradient functions."""
import numpy as np


def policy(matrix, weight):
    """Computes the policy with a weight of a matrix.

    The state is projected onto the weight matrix and the result is
    passed through a softmax, giving a probability for every action.

    Args:
        matrix: matrix representing the current observation of the
            environment.
        weight: matrix of weights.

    Returns:
        the probability of each action.
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """Computes the Monte-Carlo policy gradient.

    An action is sampled from the policy, and the gradient of the log
    probability of that action with respect to the weights is
    returned alongside it.

    Args:
        state: matrix representing the current observation of the
            environment.
        weight: matrix of random weight.

    Returns:
        the action and the gradient, in this order.
    """
    probabilities = policy(state, weight)
    action = np.random.choice(probabilities.shape[1], p=probabilities[0])

    # Jacobian of the softmax, restricted to the sampled action, then
    # divided by the action probability to get the log gradient.
    column = probabilities.reshape(-1, 1)
    softmax_grad = np.diagflat(column) - np.dot(column, column.T)
    log_grad = softmax_grad[action] / probabilities[0, action]
    gradient = np.dot(state.T, log_grad[None, :])

    return action, gradient
