#!/usr/bin/env python3
"""Calculates the softmax cross-entropy loss of a prediction"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Returns a tensor with the softmax cross-entropy loss of the prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
