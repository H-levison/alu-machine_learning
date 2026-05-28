#!/usr/bin/env python3
"""Calculates the accuracy of a neural network prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Returns a tensor with the decimal accuracy of the prediction"""
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
