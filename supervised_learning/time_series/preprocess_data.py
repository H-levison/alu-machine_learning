#!/usr/bin/env python3
"""Preprocesses the coinbase and bitstamp datasets for BTC forecasting.

The raw datasets hold one row per 60 second window. This script merges
them, keeps the recent and useful part of the data, resamples it to one
row per hour, rescales it and saves the result as a numpy archive.
"""
import numpy as np
import pandas as pd

COINBASE = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
BITSTAMP = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
OUTPUT = 'btc_preprocessed.npz'
# Unix time of 2017-01-01: older prices are too far from current ones
START = 1483228800
# The closing price must stay first: it is the value to forecast
FEATURES = ['Close', 'Weighted_Price', 'Volume_(BTC)']


def load_raw(path):
    """Loads a raw dataset indexed by the start time of its windows.

    Args:
        path: the path of the csv file to load.

    Returns:
        A pandas.DataFrame indexed by Unix time.
    """
    data = pd.read_csv(path)
    return data.set_index('Timestamp').sort_index()


def merge_exchanges(coinbase, bitstamp):
    """Merges both exchanges, filling coinbase gaps with bitstamp.

    Args:
        coinbase: pandas.DataFrame containing the coinbase data.
        bitstamp: pandas.DataFrame containing the bitstamp data.

    Returns:
        A pandas.DataFrame containing the merged data.
    """
    return coinbase.combine_first(bitstamp)


def to_hourly(data):
    """Resamples the 60 second windows into hourly windows.

    Hours without any transaction keep the previous closing and
    weighted prices and have a volume of zero.

    Args:
        data: pandas.DataFrame indexed by Unix time.

    Returns:
        A pandas.DataFrame with one row per hour holding FEATURES.
    """
    data = data[data.index >= START]
    data.index = pd.to_datetime(data.index, unit='s')
    hourly = data.resample('60min').agg({'Close': 'last',
                                         'Weighted_Price': 'mean',
                                         'Volume_(BTC)': 'sum'})
    prices = ['Close', 'Weighted_Price']
    hourly[prices] = hourly[prices].ffill()
    return hourly.dropna()[FEATURES]


def split_and_scale(values, train=0.7, valid=0.2):
    """Splits the data chronologically and standardizes it.

    The mean and standard deviation are computed on the training set
    only, so that no information leaks from the future.

    Args:
        values: numpy.ndarray of shape (hours, features).
        train: the fraction of the data used for training.
        valid: the fraction of the data used for validation.

    Returns:
        A tuple (train_set, valid_set, test_set, mean, std).
    """
    train_end = int(len(values) * train)
    valid_end = int(len(values) * (train + valid))
    mean = values[:train_end].mean(axis=0)
    std = values[:train_end].std(axis=0)
    scaled = (values - mean) / std
    return (scaled[:train_end], scaled[train_end:valid_end],
            scaled[valid_end:], mean, std)


def preprocess(coinbase_path=COINBASE, bitstamp_path=BITSTAMP,
               output=OUTPUT):
    """Preprocesses the raw datasets and saves the result.

    Args:
        coinbase_path: the path of the coinbase csv file.
        bitstamp_path: the path of the bitstamp csv file.
        output: the path of the numpy archive to create.
    """
    data = merge_exchanges(load_raw(coinbase_path), load_raw(bitstamp_path))
    hourly = to_hourly(data)
    values = hourly.values.astype(np.float32)
    train, valid, test, mean, std = split_and_scale(values)
    np.savez(output, train=train, valid=valid, test=test,
             mean=mean, std=std)
    print("Saved {} hours to {}: {} train, {} validation, {} test".format(
        len(values), output, len(train), len(valid), len(test)))


if __name__ == '__main__':
    preprocess()
