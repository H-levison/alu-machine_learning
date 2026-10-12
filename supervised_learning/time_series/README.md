# Time Series Forecasting

Forecasts the price of Bitcoin (BTC) with an RNN: the model reads the
past 24 hours of BTC data and predicts the closing price of the
following hour.

## Files

- `preprocess_data.py` preprocesses the raw coinbase and bitstamp
  datasets and saves the result to `btc_preprocessed.npz`.
- `forecast_btc.py` creates, trains and validates the keras model, then
  saves it to `btc_forecast.h5`.

## Usage

Place the two raw datasets in this directory, then run:

```
./preprocess_data.py
./forecast_btc.py
```

The expected dataset file names are
`coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv` and
`bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv`. They can be
changed at the top of `preprocess_data.py`.

## Preprocessing choices

- **Data points**: only data from 2017 onwards is kept, since older
  prices are too far from recent ones to be useful. Gaps in the coinbase
  data are filled with the bitstamp data.
- **Features**: the closing price, the volume-weighted average price and
  the amount of BTC transacted are kept. The open, high and low prices
  and the amount of USD transacted are dropped as they carry nearly the
  same information.
- **Time window**: the 60 second windows are resampled into hourly
  windows, because the model works with hours. An hour without any
  transaction keeps the previous prices and has a volume of zero.
- **Rescaling**: every feature is standardized with the mean and
  standard deviation of the training set only.
- **Split**: the data is split chronologically into training (70%),
  validation (20%) and test (10%) sets.
- **Saving**: the three sets are saved in a numpy archive together with
  the mean and standard deviation, so predictions can be converted back
  to USD.

## Model

- One LSTM layer of 64 units followed by a dense layer of 1 unit.
- Mean-squared error (MSE) as the cost function, with the Adam
  optimizer.
- Data is fed with a `tf.data.Dataset` of 24 hour windows in batches
  of 64.
- Training stops early when the validation loss stops improving.
