#!/usr/bin/env python3
"""Defines the sarsa_lambtha function."""
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


def epsilon_greedy(Q, state, epsilon):
    """Chooses an action with an epsilon-greedy policy.

    Args:
        Q: numpy.ndarray of shape (s, a) containing the Q table.
        state: the current state.
        epsilon: the probability of exploring instead of exploiting.

    Returns:
        the index of the chosen action.
    """
    if np.random.uniform() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[state]))


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Performs the SARSA(lambda) algorithm.

    Actions are chosen with an epsilon-greedy policy whose epsilon
    decays exponentially between episodes. Eligibility traces are kept
    per state-action pair, so one TD error updates every recently taken
    pair in proportion to its trace.

    Args:
        env: the openAI environment instance.
        Q: numpy.ndarray of shape (s, a) containing the Q table.
        lambtha: the eligibility trace factor.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.
        epsilon: the initial threshold for epsilon greedy.
        min_epsilon: the minimum value that epsilon should decay to.
        epsilon_decay: the decay rate for updating epsilon between
            episodes.

    Returns:
        Q, the updated Q table.
    """
    initial_epsilon = epsilon
    for episode in range(episodes):
        state = reset_env(env)
        action = epsilon_greedy(Q, state, epsilon)
        traces = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, done = step_env(env, action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            traces *= gamma * lambtha
            traces[state, action] += 1

            delta = reward + gamma * Q[next_state, next_action]
            delta -= Q[state, action]
            Q += alpha * delta * traces

            if done:
                break
            state = next_state
            action = next_action

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)
    return Q
