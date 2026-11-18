#!/usr/bin/env python3
"""Defines the q_init function."""
import numpy as np


def q_init(env):
    """Initializes the Q-table.

    Args:
        env: the FrozenLakeEnv instance.

    Returns:
        the Q-table as a numpy.ndarray of zeros, of shape
        (number of states, number of actions).
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
