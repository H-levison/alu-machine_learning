#!/usr/bin/env python3
"""Defines the sdp_attention function."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention.

    Args:
        Q: tensor with its last two dimensions as (..., seq_len_q, dk)
            containing the query matrix.
        K: tensor with its last two dimensions as (..., seq_len_v, dk)
            containing the key matrix.
        V: tensor with its last two dimensions as (..., seq_len_v, dv)
            containing the value matrix.
        mask: tensor that can be broadcast into
            (..., seq_len_q, seq_len_v) containing the optional mask.

    Returns:
        A tuple (output, weights) where output is a tensor with its
        last two dimensions as (..., seq_len_q, dv) containing the
        scaled dot product attention and weights is a tensor with its
        last two dimensions as (..., seq_len_q, seq_len_v) containing
        the attention weights.
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled += mask * -1e9

    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
