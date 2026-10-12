#!/usr/bin/env python3
"""Defines the bi_rnn function that runs a bidirectional RNN."""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for a bidirectional RNN.

    Args:
        bi_cell: instance of BidirectionalCell used for the forward
            propagation.
        X: numpy.ndarray of shape (t, m, i) containing the data, where
            t is the maximum number of time steps, m is the batch size
            and i is the dimensionality of the data.
        h_0: numpy.ndarray of shape (m, h) containing the initial
            hidden state in the forward direction.
        h_t: numpy.ndarray of shape (m, h) containing the initial
            hidden state in the backward direction.

    Returns:
        A tuple (H, Y) with all of the concatenated hidden states and
        all of the outputs.
    """
    forward = []
    h_prev = h_0
    for x_t in X:
        h_prev = bi_cell.forward(h_prev, x_t)
        forward.append(h_prev)

    backward = []
    h_next = h_t
    for x_t in X[::-1]:
        h_next = bi_cell.backward(h_next, x_t)
        backward.append(h_next)
    backward.reverse()

    H = np.concatenate((np.array(forward), np.array(backward)), axis=-1)
    return H, bi_cell.output(H)
