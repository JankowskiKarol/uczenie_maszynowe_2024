import numpy as np

from numpy import linalg as la
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster._kmeans import _tolerance
from scipy.sparse.csgraph import connected_components
from time import time

INIT_KMPP = 'k-means++'
INIT_RANDOM = 'random'
INIT_NEIGHBORHOOD = "neighborhood"

# link tolerance from sklearn
tolerance = _tolerance

def initialize_centers(X, n_clusters, init, const_mat=None, random_state=None):
    # print(init)
    if isinstance(init, np.ndarray) and init.shape == (n_clusters, len(X.T)):
        return np.copy(init)

    if isinstance(init, str):
        if INIT_KMPP in init:
            return kmeans_plusplus(X, n_clusters, random_state=random_state)[0]

        elif INIT_NEIGHBORHOOD in init:
            n_sub, labels = connected_components((const_mat == 1), directed=False, return_labels=True)
            sub_sizes = np.bincount(labels)
            center_ind = np.argsort(sub_sizes)[::-1][:n_clusters]

            # Just to be sure
            assert(len(center_ind) == n_clusters)

            return np.array([X[labels == _].mean(axis=0) for _ in center_ind])

        # also INIT_RANDOM
    else:
        rng = np.random.default_rng(random_state)
        seeds = rng.permutation(len(X))[:n_clusters]
        # print(X[seeds])
        return np.copy(X[seeds])