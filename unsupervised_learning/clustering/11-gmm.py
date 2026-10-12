#!/usr/bin/env python3
"""Defines the gmm function using sklearn."""
import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: the number of clusters.

    Returns:
        A tuple (pi, m, S, clss, bic) where pi is a numpy.ndarray of
        shape (k,) containing the cluster priors, m is a numpy.ndarray
        of shape (k, d) containing the centroid means, S is a
        numpy.ndarray of shape (k, d, d) containing the covariance
        matrices, clss is a numpy.ndarray of shape (n,) containing the
        cluster index of each data point and bic is the BIC value of
        the model.
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    return (model.weights_, model.means_, model.covariances_,
            model.predict(X), model.bic(X))
