import mouette as M
import numpy as np

### Polylines ###

def pl_one_edge():
    return M.mesh.from_arrays(V=[[0.,0.,0.], [1.,0.,0.]], E=[[0,1]])

def pl_sample1():
    return M.mesh.from_arrays(V=[[0.,0.,0.], [1.,0.,1.], [2.,1.,0.]], E=[[0,1], [1,2]])


def pl_sample2():
    return M.mesh.from_arrays(np.random.random((6,3)), E=[[0,1], [1,5], [3,5], [2,3], [0,2], [0,3], [3,4], [2,4]])

def pl_two_components():
    return M.mesh.from_arrays(np.random.random((10,3)), E=[[0,1], [1,4], [3,5], [2,3], [0,2], [0,3], [2,4], [6,7], [7,8], [7,9], [6,9]])

POLYLINES = {
    "one_edge"   :  pl_one_edge(),
    "polyline1"  :  pl_sample1(),
    "polyline2"  :  pl_sample2(),
    "pl_two_cmp" :  pl_two_components()
}

### Larger / edge-case polylines for connectivity tests ###
# Kept out of POLYLINES on purpose: that dict is parametrized over by many other test files.

def _unique_edges(E):
    """Removes self-loops and duplicate edges (regardless of orientation)"""
    E = np.asarray(E, dtype=np.int64)
    E = E[E[:,0] != E[:,1]]
    return np.unique(np.sort(E, axis=1), axis=0)

def pl_chain(n=1000):
    """Open path 0-1-2-...-(n-1)"""
    a = np.arange(n-1)
    return M.mesh.from_arrays(np.random.default_rng(10).random((n,3)), E=np.stack([a, a+1], axis=1))

def pl_loop(n=100):
    """Closed cycle, including the closing edge (n-1, 0)"""
    a = np.arange(n)
    return M.mesh.from_arrays(np.random.default_rng(11).random((n,3)), E=np.stack([a, (a+1)%n], axis=1))

def pl_random_graph(n=1000, seed=0):
    """About 3n random edges (average degree ~6), deduplicated and shuffled"""
    rng = np.random.default_rng(seed)
    E = _unique_edges(rng.integers(0, n, (3*n, 2)))
    rng.shuffle(E)
    return M.mesh.from_arrays(rng.random((n,3)), E=E)

def pl_hubs(n=2000, n_hubs=5, seed=1):
    """A few high-degree vertices: every non-hub vertex is linked to a random hub, and hubs are chained"""
    rng = np.random.default_rng(seed)
    others = np.arange(n_hubs, n)
    E = np.concatenate([
        np.stack([rng.integers(0, n_hubs, others.size), others], axis=1),
        np.stack([np.arange(n_hubs-1), np.arange(1, n_hubs)], axis=1)
    ])
    rng.shuffle(E)
    return M.mesh.from_arrays(rng.random((n,3)), E=E)

def pl_isolated_vertices():
    """Vertices 0, 4 and 8 (the first, a middle one and the last) have no adjacent edge"""
    return M.mesh.from_arrays(np.random.default_rng(12).random((9,3)), E=[[1,2], [2,3], [3,1], [5,6], [6,7]])

def pl_reversed_edges():
    """Edges stored as (a,b) with a > b"""
    return M.mesh.from_arrays(np.random.default_rng(13).random((5,3)), E=[[1,0], [2,1], [3,2], [3,0], [4,2]])

POLYLINES_CONNECTIVITY = {
    "chain"    : pl_chain(),
    "loop"     : pl_loop(),
    "random"   : pl_random_graph(),
    "hubs"     : pl_hubs(),
    "isolated" : pl_isolated_vertices(),
    "reversed" : pl_reversed_edges(),
}

ALL_POLYLINES = {**POLYLINES, **POLYLINES_CONNECTIVITY}
