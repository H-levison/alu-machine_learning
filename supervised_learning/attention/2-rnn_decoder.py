#!/usr/bin/env python3
"""Defines the RNNDecoder class used for machine translation."""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes a sequence for machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the decoder.

        Args:
            vocab: integer representing the size of the output
                vocabulary.
            embedding: integer representing the dimensionality of the
                embedding vector.
            units: integer representing the number of hidden units in
                the RNN cell.
            batch: integer representing the batch size.
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, recurrent_initializer='glorot_uniform',
            return_sequences=True, return_state=True)
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Decodes one word of the target sequence.

        Args:
            x: tensor of shape (batch, 1) containing the previous word
                in the target sequence as an index of the target
                vocabulary.
            s_prev: tensor of shape (batch, units) containing the
                previous decoder hidden state.
            hidden_states: tensor of shape (batch, input_seq_len,
                units) containing the outputs of the encoder.

        Returns:
            A tuple (y, s) where y is a tensor of shape (batch, vocab)
            containing the output word as a one hot vector in the
            target vocabulary and s is a tensor of shape (batch, units)
            containing the new decoder hidden state.
        """
        context, _ = self.attention(s_prev, hidden_states)
        embedded = self.embedding(x)
        inputs = tf.concat([tf.expand_dims(context, 1), embedded], axis=-1)
        outputs, s = self.gru(inputs)
        outputs = tf.reshape(outputs, (-1, int(outputs.shape[2])))
        y = self.F(outputs)
        return y, s
