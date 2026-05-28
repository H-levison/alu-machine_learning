#!/usr/bin/env python3
"""Creates the forward propagation graph for a neural network"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Builds the forward propagation graph layer by layer"""
    output = x
    for n, activation in zip(layer_sizes, activations):
        output = create_layer(output, n, activation)
    return output
