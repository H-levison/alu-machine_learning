#!/usr/bin/env python3
"""Defines the play function."""
import numpy as np


def play(env, Q, max_steps=100):
    """Has the trained agent play an episode.

    The Q-table is always exploited, and every state of the board is
    displayed on the console.

    Args:
        env: the FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        max_steps: maximum number of steps in the episode.

    Returns:
        the total rewards for the episode.
    """
    state = env.reset()
    env.render()
    total_reward = 0

    for _ in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward
        if done:
            break

    return total_reward
