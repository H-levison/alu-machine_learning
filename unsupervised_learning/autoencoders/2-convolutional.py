#!/usr/bin/env python3
"""Creates a convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Returns encoder, decoder, and full convolutional autoencoder models"""
    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    encoded = encoder_input
    for f in filters:
        encoded = keras.layers.Conv2D(f, (3, 3), padding='same',
                                      activation='relu')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoder = keras.Model(encoder_input, encoded)

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    decoded = decoder_input
    reversed_filters = list(reversed(filters))
    for i, f in enumerate(reversed_filters):
        padding = 'valid' if i == len(reversed_filters) - 1 else 'same'
        decoded = keras.layers.Conv2D(f, (3, 3), padding=padding,
                                      activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), padding='same',
                                  activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, decoded)

    # Autoencoder
    auto_input = keras.Input(shape=input_dims)
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(auto_input, auto_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
