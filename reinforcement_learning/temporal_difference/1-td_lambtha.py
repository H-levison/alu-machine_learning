#!/usr/bin/env python3
"""Defines the td_lambtha function."""
import numpy as np


def reset_env(env):
    """Resets an environment and returns the starting state.

    Supports both the classic gym API, where reset returns the state,
    and the newer API, where it returns a (state, info) tuple.
    """
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    return state


def step_env(env, action):
    """Takes one step in an environment.

    Returns:
        state, reward, done, flattening the 4-tuple of the classic gym
        API and the 5-tuple of the newer API.
    """
    output = env.step(action)
    if len(output) == 5:
        state, reward, terminated, truncated, _ = output
        return state, reward, terminated or truncated
    state, reward, done, _ = output
    return state, reward, done


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Performs the TD(lambda) algorithm.

    An eligibility trace is kept for every state. It decays by
    gamma * lambtha at each step and is incremented for the state just
    visited, so a single TD error updates every recently visited state
    in proportion to its trace.

    Args:
        env: the openAI environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: function that takes in a state and returns the next
            action to take.
        lambtha: the eligibility trace factor.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.

    Returns:
        V, the updated value estimate.
    """
    states = V.shape[0]
    for _ in range(episodes):
        state = reset_env(env)
        traces = np.zeros(states)
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done = step_env(env, action)

            traces *= gamma * lambtha
            traces[state] += 1

            delta = reward + gamma * V[next_state] - V[state]
            V += alpha * delta * traces

            if done:
                break
            state = next_state
    return V
