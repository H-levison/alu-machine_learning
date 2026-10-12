#!/usr/bin/env python3
"""Defines the RNNCell class that represents a cell of a simple RNN."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Initializes the weights and biases of the cell.

        Args:
            i: the dimensionality of the data.
            h: the dimensionality of the hidden state.
            o: the dimensionality of the outputs.
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the
                previous hidden state.
            x_t: numpy.ndarray of shape (m, i) containing the data
                input for the cell.

        Returns:
            A tuple (h_next, y) with the next hidden state and the
            softmax output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Wh + self.bh)

        z = h_next @ self.Wy + self.by
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, y
