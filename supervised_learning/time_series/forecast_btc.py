#!/usr/bin/env python3
"""Creates, trains and validates a keras model that forecasts BTC.

The model uses the past 24 hours of BTC data to predict the closing
price of the following hour. Run preprocess_data.py first.
"""
import numpy as np
import tensorflow as tf

DATA = 'btc_preprocessed.npz'
MODEL = 'btc_forecast.h5'
WINDOW = 24
BATCH_SIZE = 64
EPOCHS = 30


def make_windows(data, window=WINDOW):
    """Cuts a time series into input windows and their targets.

    Args:
        data: numpy.ndarray of shape (hours, features) whose first
            feature is the closing price.
        window: the number of hours used to make a prediction.

    Returns:
        A tuple (inputs, targets) where inputs has the shape
        (samples, window, features) and targets has the shape
        (samples, 1) and holds the closing price of the hour that
        follows each window.
    """
    samples = len(data) - window
    inputs = np.array([data[i:i + window] for i in range(samples)])
    targets = data[window:, :1]
    return inputs.astype(np.float32), targets.astype(np.float32)


def make_dataset(data, shuffle=False):
    """Creates the tf.data.Dataset that feeds the model.

    Args:
        data: numpy.ndarray of shape (hours, features).
        shuffle: whether the windows should be shuffled.

    Returns:
        A tuple (dataset, steps) with the batched, repeating dataset
        and the number of batches in one pass over the data.
    """
    inputs, targets = make_windows(data)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(len(inputs))
    dataset = dataset.batch(BATCH_SIZE).repeat()
    return dataset, int(np.ceil(len(inputs) / BATCH_SIZE))


def build_model(features):
    """Builds the LSTM model, compiled with the MSE cost function.

    Args:
        features: the number of features of each hour of data.

    Returns:
        The compiled keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(WINDOW, features)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def forecast_btc():
    """Trains the model, validates it and reports its test error."""
    data = np.load(DATA)
    train, train_steps = make_dataset(data['train'], shuffle=True)
    valid, valid_steps = make_dataset(data['valid'])
    test, test_steps = make_dataset(data['test'])

    model = build_model(data['train'].shape[1])
    model.summary()
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.fit(train, epochs=EPOCHS, steps_per_epoch=train_steps,
              validation_data=valid, validation_steps=valid_steps,
              callbacks=[stop])

    mse = model.evaluate(test, steps=test_steps)
    rmse_usd = np.sqrt(mse) * data['std'][0]
    print("Test MSE (scaled): {}".format(mse))
    print("Test RMSE (USD): {}".format(rmse_usd))
    model.save(MODEL)


if __name__ == '__main__':
    forecast_btc()
