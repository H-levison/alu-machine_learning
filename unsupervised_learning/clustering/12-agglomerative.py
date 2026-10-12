#!/usr/bin/env python3
"""Defines the agglomerative function."""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset.

    The clustering uses Ward linkage, and the dendrogram is displayed
    with each cluster in a different color.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        dist: the maximum cophenetic distance for all clusters.

    Returns:
        A numpy.ndarray of shape (n,) containing the cluster index of
        each data point.
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return scipy.cluster.hierarchy.fcluster(linkage, t=dist,
                                            criterion='distance')
