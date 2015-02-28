import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx

def correlation_matrix(mat):
    return pairwise_distances(mat, metric=lambda x, y: pearsonr(x, y)[1])

def window(data, window_size, stride):
    for i in range(0, len(data), stride):
        if i+window_size <= len(data):
            yield np.array(data[i:i+window_size])

def cross_correlation_matrix(mat):
    return pairwise_distances(mat, metric=overlapping_correlate)

def overlapping_correlate(a, b):
    max = 0.
    for c, d in zip(window(a, 400, 200), window(b, 400, 200)):
        c = (c - np.mean(c)) / (np.std(c) * len(c))
        d = (d - np.mean(d)) /  np.std(d)
        temp = np.correlate(c,d)
        if temp > max:
            max = temp 
    return max

def mask_correlation(i):
    if i <= .05 and i > 0:
        return 1.
    else:
        return 0.

def mask_cross_correlation(i):
    if i >= .7:
        return 1.
    else:
        return 0.

def correlation_adjacency_matrix(mat):
    f = np.vectorize(mask_correlation)
    return f(correlation_matrix(mat))

def cross_correlation_adjacency_matrix(ccm):
    f = np.vectorize(mask_cross_correlation)
    return f(ccm)

def correlation_graph(mat):
    mat = correlation_matrix(mat)
    mat = correlation_adjacency_matrix(mat)
    return nx.Graph(mat)

def cross_correlation_graph(mat):
    mat = cross_correlation_matrix(mat)
    mat = cross_correlation_adjacency_matrix(mat)
    return nx.Graph(mat)
