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

    def output(self, H):
        """Calculates all of the outputs for the RNN.

        Args:
            H: numpy.ndarray of shape (t, m, 2 * h) containing the
                concatenated hidden states from both directions,
                excluding their initialized states.

        Returns:
            A numpy.ndarray of shape (t, m, o) containing the softmax
            outputs.
        """
        out = H @ self.Wy + self.by
        exp = np.exp(out - np.max(out, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
