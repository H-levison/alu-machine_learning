#!/usr/bin/env python3
"""Defines the RNNEncoder class used for machine translation."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodes a sequence for machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the encoder.

        Args:
            vocab: integer representing the size of the input
                vocabulary.
            embedding: integer representing the dimensionality of the
                embedding vector.
            units: integer representing the number of hidden units in
                the RNN cell.
            batch: integer representing the batch size.
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, recurrent_initializer='glorot_uniform',
            return_sequences=True, return_state=True)

    def initialize_hidden_state(self):
        """Initializes the hidden states of the RNN cell to zeros.

        Returns:
            A tensor of shape (batch, units) containing the initialized
            hidden states.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Encodes the input sequence.

        Args:
            x: tensor of shape (batch, input_seq_len) containing the
                input to the encoder layer as word indices within the
                vocabulary.
            initial: tensor of shape (batch, units) containing the
                initial hidden state.

        Returns:
            A tuple (outputs, hidden) where outputs is a tensor of
            shape (batch, input_seq_len, units) containing the outputs
            of the encoder and hidden is a tensor of shape
            (batch, units) containing its last hidden state.
        """
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded, initial_state=initial)
        return outputs, hidden
