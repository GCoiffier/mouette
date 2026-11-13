import numpy as np
from numba import njit, prange


# @njit(cache=True, parallel=True)
def _mean_per_element(elements, values):
    n = elements.shape[0]
    ne = elements.shape[1]
    nv = 1 if len(values.shape)==1 else values.shape[1]
    bary = np.empty((n, nv), dtype=float)
    for i in prange(n):
        acc = np.zeros(nv, dtype=float)
        for j in range(ne):
            acc += values[elements[i,j]]
        bary[i] = acc/ne
    return bary

# @njit(cache=True, parallel=True)
def _weighted_broadcast(elements, values, output, weights, normalize):
    # elements = faces
    # values = values per face
    # output = values per vertex
    ne = elements.shape[0]
    nv = elements.shape[1]
    total_weights = np.zeros((output.shape[0],1))
    for i in prange(ne):
        for j in range(nv):
            output[elements[i,j]] += values[i]*weights[i,j]
            total_weights[elements[i,j]] += weights[i,j]
    if normalize:
        output /= total_weights
    return output
