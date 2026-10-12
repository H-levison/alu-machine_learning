#!/usr/bin/env python3
"""Defines the pdf function."""
import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data points
            whose PDF should be evaluated.
        m: numpy.ndarray of shape (d,) containing the mean of the
            distribution.
        S: numpy.ndarray of shape (d, d) containing the covariance of
            the distribution.

    Returns:
        A numpy.ndarray of shape (n,) containing the PDF values of
        each data point, with a minimum value of 1e-300, or None on
        failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    d = X.shape[1]
    if not isinstance(m, np.ndarray) or m.shape != (d,):
        return None
    if not isinstance(S, np.ndarray) or S.shape != (d, d):
        return None

    difference = X - m
    inverse = np.linalg.inv(S)
    exponent = -0.5 * np.sum(difference @ inverse * difference, axis=1)
    denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(S))
    return np.maximum(np.exp(exponent) / denominator, 1e-300)
