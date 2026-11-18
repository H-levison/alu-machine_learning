#!/usr/bin/env python3
"""Defines the train function."""
import numpy as np
from policy_gradient import policy_gradient


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


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Implements a full training with Monte-Carlo policy gradient.

    Each episode is played to completion, and the weights are then
    nudged along the gradient of every visited state, scaled by the
    discounted return that followed it.

    Args:
        env: initial environment.
        nb_episodes: number of episodes used for training.
        alpha: the learning rate.
        gamma: the discount factor.
        show_result: when True, the environment is rendered every 1000
            episodes.

    Returns:
        all values of the score, the sum of all rewards during one
        episode loop.
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state = reset_env(env)[None, :]
        gradients = []
        rewards = []
        score = 0
        render = show_result and episode % 1000 == 0

        while True:
            if render:
                env.render()
            action, gradient = policy_gradient(state, weight)
            next_state, reward, done = step_env(env, action)
            gradients.append(gradient)
            rewards.append(reward)
            score += reward
            state = next_state[None, :]
            if done:
                break

        for step, gradient in enumerate(gradients):
            future = rewards[step:]
            discounted = sum(reward * gamma ** power
                             for power, reward in enumerate(future))
            weight += alpha * gradient * discounted

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score), end="\r",
              flush=False)

    return scores
