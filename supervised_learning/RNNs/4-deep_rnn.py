#!/usr/bin/env python3
"""Defines the deep_rnn function that runs a deep RNN forward."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances of length l used for the
            forward propagation, where l is the number of layers.
        X: numpy.ndarray of shape (t, m, i) containing the data, where
            t is the maximum number of time steps, m is the batch size
            and i is the dimensionality of the data.
        h_0: numpy.ndarray of shape (l, m, h) containing the initial
            hidden state of every layer.

    Returns:
        A tuple (H, Y) with all of the hidden states, including the
        initial ones, and all of the outputs of the last layer.
    """
    H = [h_0]
    Y = []
    for x_t in X:
        layer_input = x_t
        h_step = []
        for layer, cell in enumerate(rnn_cells):
            layer_input, y = cell.forward(H[-1][layer], layer_input)
            h_step.append(layer_input)
        H.append(np.array(h_step))
        Y.append(y)
    return np.array(H), np.array(Y)
