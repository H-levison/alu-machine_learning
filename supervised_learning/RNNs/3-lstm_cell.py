#!/usr/bin/env python3
"""Defines the LSTMCell class that represents an LSTM unit."""
import numpy as np


def sigmoid(x):
    """Calculates the sigmoid activation of x."""
    return 1 / (1 + np.exp(-x))


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """Initializes the weights and biases of the cell.

        Args:
            i: the dimensionality of the data.
            h: the dimensionality of the hidden state.
            o: the dimensionality of the outputs.
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the
                previous hidden state.
            c_prev: numpy.ndarray of shape (m, h) containing the
                previous cell state.
            x_t: numpy.ndarray of shape (m, i) containing the data
                input for the cell.

        Returns:
            A tuple (h_next, c_next, y) with the next hidden state,
            the next cell state and the softmax output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        f = sigmoid(concat @ self.Wf + self.bf)
        u = sigmoid(concat @ self.Wu + self.bu)
        c_inter = np.tanh(concat @ self.Wc + self.bc)
        o = sigmoid(concat @ self.Wo + self.bo)

        c_next = f * c_prev + u * c_inter
        h_next = o * np.tanh(c_next)

        out = h_next @ self.Wy + self.by
        exp = np.exp(out - np.max(out, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, c_next, y
