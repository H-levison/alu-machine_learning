#!/usr/bin/env python3
"""Defines the EncoderBlock class of a transformer."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Represents an encoder block of a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the encoder block.

        Args:
            dm: the dimensionality of the model.
            h: the number of heads.
            hidden: the number of hidden units in the fully connected
                layer.
            drop_rate: the dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Calculates the output of the encoder block.

        Args:
            x: tensor of shape (batch, input_seq_len, dm) containing
                the input to the encoder block.
            training: boolean to determine if the model is training.
            mask: the mask to be applied for multi head attention.

        Returns:
            A tensor of shape (batch, input_seq_len, dm) containing the
            output of the block.
        """
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)

        feed_forward = self.dense_output(self.dense_hidden(out1))
        feed_forward = self.dropout2(feed_forward, training=training)
        return self.layernorm2(out1 + feed_forward)
