#!/usr/bin/env python3
"""Defines the SelfAttention class used for machine translation."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates the attention for machine translation."""

    def __init__(self, units):
        """Initializes the attention layer.

        Args:
            units: integer representing the number of hidden units in
                the alignment model.
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculates the context vector and the attention weights.

        Args:
            s_prev: tensor of shape (batch, units) containing the
                previous decoder hidden state.
            hidden_states: tensor of shape (batch, input_seq_len,
                units) containing the outputs of the encoder.

        Returns:
            A tuple (context, weights) where context is a tensor of
            shape (batch, units) containing the context vector for the
            decoder and weights is a tensor of shape
            (batch, input_seq_len, 1) containing the attention weights.
        """
        s_expanded = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_expanded) +
                                  self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
