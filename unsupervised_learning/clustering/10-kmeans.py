#!/usr/bin/env python3
"""Defines the kmeans function using sklearn."""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: the number of clusters.

    Returns:
        A tuple (C, clss) where C is a numpy.ndarray of shape (k, d)
        containing the centroid means of each cluster and clss is a
        numpy.ndarray of shape (n,) containing the index of the
        cluster each data point belongs to.
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return model.cluster_centers_, model.labels_
