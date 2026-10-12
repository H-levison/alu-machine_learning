#!/usr/bin/env python3
"""Defines the positional_encoding function for a transformer."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer.

    Args:
        max_seq_len: integer representing the maximum sequence length.
        dm: the model depth.

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors.
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]
    angles = positions / np.power(10000, 2 * (dimensions // 2) / dm)

    encoding = np.zeros((max_seq_len, dm))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding
