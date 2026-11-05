import mouette as M
import numpy as np
import pytest
from data import *

KDTree = pytest.importorskip("scipy.spatial").KDTree
igl = pytest.importorskip("igl")

def query_points(V, n, seed=0):
    """Random points in the bounding box of V enlarged by 20%, so that some queries fall outside of the geometry."""
    rng = np.random.default_rng(seed)
    pmin, pmax = V.min(axis=0), V.max(axis=0)
    margin = 0.2 * np.maximum(pmax - pmin, 1e-3)
    return rng.uniform(pmin - margin, pmax + margin, size=(n, V.shape[1]))

### Points ###

@pytest.mark.parametrize("leaf_size", [1, 4, 10, 32])
@pytest.mark.parametrize("mesh", surfaces + point_clouds)
def test_bvh_points_vs_kdtree(mesh, leaf_size):
    V = np.asarray(mesh.vertices, dtype=float)
    Q = query_points(V, 2000)
    bvh = M.spatial.BoundingVolumeHierarchy(V, leaf_size=leaf_size)
    d, p = bvh.query_distance(Q)
    d_ref, _ = KDTree(V).query(Q)
    assert np.allclose(d, d_ref, atol=1e-10)
    # the closest point is at the returned distance and is one of the input vertices (robust to ties)
    assert np.allclose(np.linalg.norm(Q - p, axis=1), d, atol=1e-10)
    d_p, _ = KDTree(V).query(p)
    assert np.allclose(d_p, 0., atol=1e-12)

@pytest.mark.parametrize("leaf_size", [1, 10])
def test_bvh_points_2d_vs_kdtree(leaf_size):
    V = np.random.default_rng(1).random((5000, 2))
    Q = query_points(V, 2000)
    bvh = M.spatial.BoundingVolumeHierarchy(V, leaf_size=leaf_size)
    d, p = bvh.query_distance(Q)
    d_ref, _ = KDTree(V).query(Q)
    assert np.allclose(d, d_ref, atol=1e-10)
    assert np.allclose(np.linalg.norm(Q - p, axis=1), d, atol=1e-10)

### Triangles ###

triangle_surfaces = surfaces + [surf_spot(), surf_cube(), surf_tetrahedron(), surf_torus()]

@pytest.mark.parametrize("leaf_size", [1, 4, 10, 32])
@pytest.mark.parametrize("mesh", triangle_surfaces)
def test_bvh_triangles_vs_igl(mesh, leaf_size):
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int64)
    Q = query_points(V, 2000)
    bvh = M.spatial.BoundingVolumeHierarchy(V, F, leaf_size=leaf_size)
    d, p = bvh.query_distance(Q)
    # the default (winding number) sign type alters the magnitude on open or inconsistently oriented meshes
    d_ref = np.abs(igl.signed_distance(Q, V, F, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_UNSIGNED)[0])
    assert np.allclose(d, d_ref, atol=1e-8)
    assert np.allclose(np.linalg.norm(Q - p, axis=1), d, atol=1e-8)

### Oriented queries ###

def test_bvh_oriented_with_points():
    # oriented=True is ignored on a point cloud BVH
    V = np.asarray(surf_spot().vertices, dtype=float)
    Q = query_points(V, 100)
    bvh = M.spatial.BoundingVolumeHierarchy(V)
    d, p = bvh.query_distance(Q)
    d_or, p_or = bvh.query_distance(Q, oriented=True)
    assert np.allclose(d_or, d) and np.allclose(p_or, p)

@pytest.mark.parametrize("leaf_size", [1, 4, 10, 32])
@pytest.mark.parametrize("mesh", closed_triangle_surfaces)
def test_bvh_triangles_oriented_vs_igl(mesh, leaf_size):
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int64)
    Q = query_points(V, 2000)
    bvh = M.spatial.BoundingVolumeHierarchy(V, F, leaf_size=leaf_size)
    d, p = bvh.query_distance(Q, oriented=True)
    d_ref = igl.signed_distance(Q, V, F, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0]
    assert np.allclose(d, d_ref, atol=1e-8)
    assert np.allclose(np.linalg.norm(Q - p, axis=1), np.abs(d), atol=1e-8)

@pytest.mark.parametrize("leaf_size", [1, 10])
def test_bvh_edges_2d_oriented(leaf_size):
    # regular polygon inscribed in the unit circle, edges running counter-clockwise
    n = 200
    t = 2 * np.pi * np.arange(n) / n
    V = np.stack((np.cos(t), np.sin(t)), axis=1)
    E = np.stack((np.arange(n), (np.arange(n) + 1) % n), axis=1)
    Q = np.random.default_rng(3).uniform(-1.5, 1.5, size=(2000, 2))
    r = np.linalg.norm(Q, axis=1)
    Q = Q[(r < 0.99) | (r > 1.01)] # skip a thin band around the boundary, where the polygon differs from the circle
    inside = np.linalg.norm(Q, axis=1) < 1.
    d_unoriented, _ = M.spatial.BoundingVolumeHierarchy(V, E, leaf_size=leaf_size).query_distance(Q)
    d, _ = M.spatial.BoundingVolumeHierarchy(V, E, leaf_size=leaf_size).query_distance(Q, oriented=True)
    assert np.all((d < 0) == inside)
    assert np.allclose(np.abs(d), d_unoriented)
    # reversing the orientation of the edges flips the sign
    d_rev, _ = M.spatial.BoundingVolumeHierarchy(V, E[:, ::-1], leaf_size=leaf_size).query_distance(Q, oriented=True)
    assert np.allclose(d_rev, -d)

### Other ###

@pytest.mark.parametrize("mesh", [surf_spot(), surf_cube(), surf_torus()])
def test_bvh_query_on_vertices(mesh):
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int64)
    bvh = M.spatial.BoundingVolumeHierarchy(V, F)
    d, p = bvh.query_distance(V)
    assert np.allclose(d, 0., atol=1e-10)
    assert np.allclose(p, V, atol=1e-10)

def test_bvh_single_query():
    mesh = surf_spot()
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int64)
    bvh = M.spatial.BoundingVolumeHierarchy(V, F)
    Q = query_points(V, 10)
    d, p = bvh.query_distance(Q)
    d0, p0 = bvh.query_distance(Q[0])
    assert d0.shape == (1,) and p0.shape == (1, 3)
    assert np.isclose(d0[0], d[0])
    assert np.allclose(p0[0], p[0])

def test_bvh_invalid_simplices():
    V = np.random.default_rng(2).random((10, 2))
    F = np.array([[0, 1, 2], [3, 4, 5]])
    with pytest.raises(Exception):
        M.spatial.BoundingVolumeHierarchy(V, F)
