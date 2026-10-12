#!/usr/bin/env python3
"""Defines the rnn function that runs a simple RNN forward."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: instance of RNNCell used for the forward propagation.
        X: numpy.ndarray of shape (t, m, i) containing the data, where
            t is the maximum number of time steps, m is the batch size
            and i is the dimensionality of the data.
        h_0: numpy.ndarray of shape (m, h) containing the initial
            hidden state.

    Returns:
        A tuple (H, Y) with all of the hidden states, including the
        initial one, and all of the outputs.
    """
    H = [h_0]
    Y = []
    for x_t in X:
        h_next, y = rnn_cell.forward(H[-1], x_t)
        H.append(h_next)
        Y.append(y)
    return np.array(H), np.array(Y)
