#!/usr/bin/env python3
"""Defines the GRUCell class that represents a gated recurrent unit."""
import numpy as np


def sigmoid(x):
    """Calculates the sigmoid activation of x."""
    return 1 / (1 + np.exp(-x))


class GRUCell:
    """Represents a gated recurrent unit."""

    def __init__(self, i, h, o):
        """Initializes the weights and biases of the cell.

        Args:
            i: the dimensionality of the data.
            h: the dimensionality of the hidden state.
            o: the dimensionality of the outputs.
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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
        z = sigmoid(concat @ self.Wz + self.bz)
        r = sigmoid(concat @ self.Wr + self.br)

        concat = np.concatenate((r * h_prev, x_t), axis=1)
        h_inter = np.tanh(concat @ self.Wh + self.bh)
        h_next = (1 - z) * h_prev + z * h_inter

        out = h_next @ self.Wy + self.by
        exp = np.exp(out - np.max(out, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, y
