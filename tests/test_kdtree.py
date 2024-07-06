import mouette as M
import numpy as np
import pytest

@pytest.mark.parametrize("dim", [2,3])
def test_kdtree_knn(dim):
    domain = M.geometry.AABB.unit_cube(dim)
    points = M.sampling.sample_AABB(domain, 1_000)
    tree = M.spatial.KDTree(points)
    query_pt = M.Vec.random(dim)
    nn = tree.query(query_pt, 30)
    assert len(nn)==30
    dist_max = max([M.geometry.distance(query_pt, points[idx,:]) for idx in nn])
    for i,pt in enumerate(points):
        if M.geometry.distance(query_pt, pt)<= dist_max:
            assert i in nn
        else:
            assert i not in nn

@pytest.mark.parametrize("dim", [2,3])
def test_kdtree_knn_compare_linear(dim):
    domain = M.geometry.AABB.unit_cube(dim)
    points = M.sampling.sample_AABB(domain, 1_000)
    tree = M.spatial.KDTree(points)
    query_pt = M.Vec.random(dim)
    nn = tree.query(query_pt, 50)
    distances = np.array([M.geometry.distance(p, query_pt) for p in points])
    nn_brute = distances.argsort()[:50]
    assert np.all(nn == nn_brute)

@pytest.mark.parametrize("dim", [2,3])
def test_kdtree_radius(dim):
    domain = M.geometry.AABB.unit_cube(dim)
    points = M.sampling.sample_AABB(domain, 1_000)
    tree = M.spatial.KDTree(points)
    query_pt = M.Vec.random(dim)
    radius = 0.1
    nn = tree.query_radius(query_pt, radius)
    for i,pt in enumerate(points):
        if M.geometry.distance(query_pt, pt) <= radius:
            assert i in nn
        else:
            assert i not in nn

def test_kdtree_highdim():
    domain = M.geometry.AABB.unit_cube(10)
    points = M.sampling.sample_AABB(domain, 1_000)
    tree = M.spatial.KDTree(points)
    query_pt = M.Vec.random(10)
    nn = tree.query(query_pt, 10)
    assert len(nn)==10
    dist_max = max([M.geometry.distance(query_pt, points[idx,:]) for idx in nn])
    for i,pt in enumerate(points):
        if M.geometry.distance(query_pt, pt)<= dist_max:
            assert i in nn
        else:
            assert i not in nn
