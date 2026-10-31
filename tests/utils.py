import mouette as M
import numpy as np
import warnings
from scipy.spatial import cKDTree

def as_tuples(elems, vmap=None):
    """Elements as tuples of python ints, with vertex indices optionally remapped through vmap"""
    if vmap is None: return [tuple(int(i) for i in e) for e in elems]
    return [tuple(int(vmap[i]) for i in e) for e in elems]

def compare_mesh(mesh1, mesh2, reorder_vertices=False, tol=1e-8):
    assert type(mesh1) == type(mesh2)
    V1, V2 = np.asarray(mesh1.vertices), np.asarray(mesh2.vertices)
    assert V1.shape == V2.shape
    vmap = None
    if reorder_vertices: # format does not preserve vertex order (stl): match vertices by position
        dist, vmap = cKDTree(V1).query(V2)
        assert np.all(dist < tol)
    else:
        assert np.allclose(V1, V2, atol=tol)
    if hasattr(mesh1, "edges"): # edges may be rebuilt from faces: compare as sets
        assert set(map(frozenset, as_tuples(mesh1.edges))) == set(map(frozenset, as_tuples(mesh2.edges, vmap)))
    if hasattr(mesh1, "cells"): # faces of a volume mesh are rebuilt from cells: compare as sets
        assert set(map(frozenset, as_tuples(mesh1.faces))) == set(map(frozenset, as_tuples(mesh2.faces, vmap)))
    elif hasattr(mesh1, "faces"):
        assert as_tuples(mesh1.faces) == as_tuples(mesh2.faces, vmap)
    if hasattr(mesh1, "cells"):
        assert as_tuples(mesh1.cells) == as_tuples(mesh2.cells, vmap)

def build_test_io(mesh, tmp_path, fmt, dim=None):
    filepath = tmp_path / f"test_mesh.{fmt}"
    with warnings.catch_warnings():
        # dropping attributes on formats that cannot store them is expected here
        warnings.filterwarnings("ignore", message="Attributes are not exported", category=UserWarning)
        M.mesh.save(mesh, str(filepath))
    mesh2 = M.mesh.load(str(filepath), dim)
    compare_mesh(mesh, mesh2, reorder_vertices=(fmt=="stl"))
