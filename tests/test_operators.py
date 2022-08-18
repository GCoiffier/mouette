import pygeomesh as GEO
from data import *

@pytest.mark.parametrize("m", surfaces)
def test_cotan_laplacian(m):
    mat = GEO.operators.laplacian(m, False)
    n = len(m.vertices)
    assert mat.shape == (n,n)
    
@pytest.mark.parametrize("m", surfaces + polylines)
def test_graph_laplacian(m):
    mat = GEO.operators.graph_laplacian(m)
    assert True

@pytest.mark.parametrize("m", surfaces)
def test_weight_matrix(m):
    mat = GEO.operators.area_weight_matrix(m)
    n = len(m.vertices)
    assert mat.shape == (n,n)

@pytest.mark.parametrize("m", surfaces + polylines)
def test_connectivity_matrix(m):
    mat = GEO.operators.connectivity_matrix(m)
    n = len(m.vertices)
    assert mat.shape == (n,n)