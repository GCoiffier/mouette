import mouette as M
from data import *

@pytest.mark.parametrize("m", surfaces)
def test_cotan_laplacian(m):
    mat = M.operators.laplacian(m, False)
    n = len(m.vertices)
    assert mat.shape == (n,n)
    
@pytest.mark.parametrize("m", surfaces + polylines)
def test_graph_laplacian(m):
    mat = M.operators.graph_laplacian(m)
    n = len(m.vertices)
    assert mat.shape == (n,n)

@pytest.mark.parametrize("m", surfaces)
def test_weight_matrix(m):
    mat = M.operators.area_weight_matrix(m)
    n = len(m.vertices)
    assert mat.shape == (n,n)

@pytest.mark.parametrize("m", surfaces + polylines)
def test_adjacency_matrix(m):
    mat = M.operators.adjacency_matrix(m)
    n = len(m.vertices)
    assert mat.shape == (n,n)