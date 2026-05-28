#!/usr/bin/env python3
"""Creates a vanilla autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Returns encoder, decoder, and full autoencoder models"""
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(encoder_input, latent)

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, decoder_output)

    # Autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(auto_input, auto_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
