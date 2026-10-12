#!/usr/bin/env python3
"""Defines the BidirectionalCell class of a bidirectional RNN."""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """Initializes the weights and biases of the cell.

        Args:
            i: the dimensionality of the data.
            h: the dimensionality of the hidden states.
            o: the dimensionality of the outputs.
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculates the forward hidden state for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the
                previous hidden state.
            x_t: numpy.ndarray of shape (m, i) containing the data
                input for the cell.

        Returns:
            The next hidden state.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(concat @ self.Whf + self.bhf)

    def backward(self, h_next, x_t):
        """Calculates the backward hidden state for one time step.

        Args:
            h_next: numpy.ndarray of shape (m, h) containing the next
                hidden state.
            x_t: numpy.ndarray of shape (m, i) containing the data
                input for the cell.

        Returns:
            The previous hidden state.
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        return np.tanh(concat @ self.Whb + self.bhb)
