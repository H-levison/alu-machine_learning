#!/usr/bin/env python3
"""Creates a variational autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Returns encoder, decoder, and full variational autoencoder models"""
    K = keras.backend

    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        mean, lv = args
        eps = K.random_normal(shape=K.shape(mean))
        return mean + K.exp(lv / 2) * eps

    z = keras.layers.Lambda(sampling)([mu, log_var])
    encoder = keras.Model(encoder_input, [z, mu, log_var])

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    y = decoder_input
    for nodes in reversed(hidden_layers):
        y = keras.layers.Dense(nodes, activation='relu')(y)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(y)
    decoder = keras.Model(decoder_input, decoder_output)

    # Autoencoder with KL divergence loss
    auto_input = keras.Input(shape=(input_dims,))
    z_out, mu_out, log_var_out = encoder(auto_input)
    auto_output = decoder(z_out)
    auto = keras.Model(auto_input, auto_output)

    kl_loss = -0.5 * K.sum(
        1 + log_var_out - K.square(mu_out) - K.exp(log_var_out), axis=-1)
    auto.add_loss(K.mean(kl_loss))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
