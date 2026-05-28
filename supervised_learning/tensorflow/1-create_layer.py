#!/usr/bin/env python3
"""Creates a layer for a neural network"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a dense layer with He et al. initialization"""
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(prev, n, activation=activation,
                            kernel_initializer=initializer, name='layer')
    return layer
