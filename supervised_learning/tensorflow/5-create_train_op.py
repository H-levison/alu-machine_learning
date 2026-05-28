#!/usr/bin/env python3
"""Creates the training operation for a neural network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Returns a gradient descent training operation"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
