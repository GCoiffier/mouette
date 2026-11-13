import mouette as M
from data import *

@pytest.mark.parametrize("m", list(POLYLINES.values()) + list(ALL_SURFACES.values()) + list(VOLUMES.values()))
def test_edge_spanning_tree(m):
    _ = M.processing.trees.EdgeSpanningTree(m)()
    assert True


@pytest.mark.parametrize("m", [ALL_SURFACES["circle"], ALL_SURFACES["half_sphere"]])
def test_edge_spanning_tree_avoid_edges(m):
    _ = M.processing.trees.EdgeSpanningTree(m, avoid_boundary=True, avoid_edges={0})()
    assert True

@pytest.mark.parametrize("m", [ALL_SURFACES["circle"], ALL_SURFACES["half_sphere"]])
def test_edge_spanning_tree_traverse(m):
    tree = M.processing.trees.EdgeSpanningTree(m, starting_vertex=0, avoid_boundary=True)()
    for _ in tree.traverse(): continue
    assert True

@pytest.mark.parametrize("m", list(POLYLINES.values()) + list(ALL_SURFACES.values()) + list(VOLUMES.values()))
def test_edge_minimal_spanning_tree(m):
    _ = M.processing.trees.EdgeMinimalSpanningTree(m, avoid_boundary=True)()
    assert True

@pytest.mark.parametrize("m", list(POLYLINES.values()) + list(ALL_SURFACES.values()) + list(VOLUMES.values()))
def test_edge_spanning_forest(m):
    _ = M.processing.trees.EdgeSpanningForest(m)()
    assert True

@pytest.mark.parametrize("m", [ALL_SURFACES["circle"], ALL_SURFACES["half_sphere"]])
def test_edge_spanning_forest_traverse(m):
    forest = M.processing.trees.EdgeSpanningForest(m)()
    for _ in forest.traverse(): continue
    assert True

@pytest.mark.parametrize("m", ALL_SURFACES.items())
def test_face_spanning_tree(m):
    name, m = m
    tree = M.processing.trees.FaceSpanningTree(m)()
    assert True

def test_face_spanning_tree_avoid_edges():
    m = ALL_SURFACES["circle"]
    tree = M.processing.trees.FaceSpanningTree(m,0,{10,12,42})()
    for _ in tree.traverse(): continue
    assert True

@pytest.mark.parametrize("m", ALL_SURFACES.items())
def test_face_spanning_forest(m):
    name, m = m
    _ = M.processing.trees.FaceSpanningForest(m)()
    assert True

def test_cell_spanning_tree():
    m = VOLUMES["cuboid"]
    tree = M.processing.trees.CellSpanningTree(m)()
    assert True
    for _ in tree.traverse(): continue
    assert True

def test_cell_spanning_forest():
    m = VOLUMES["cuboid"]
    tree = M.processing.trees.CellSpanningForest(m)()
    assert True
    for _ in tree.traverse(): continue
    assert True