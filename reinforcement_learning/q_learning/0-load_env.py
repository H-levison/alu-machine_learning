#!/usr/bin/env python3
"""Defines the load_frozen_lake function."""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the pre-made FrozenLakeEnv environment from OpenAI's gym.

    Args:
        desc: either None or a list of lists containing a custom
            description of the map to load for the environment.
        map_name: either None or a string containing the pre-made map
            to load. If both desc and map_name are None, a randomly
            generated 8x8 map is loaded.
        is_slippery: boolean determining if the ice is slippery.

    Returns:
        the environment.
    """
    return gym.make('FrozenLake-v0', desc=desc, map_name=map_name,
                    is_slippery=is_slippery)
