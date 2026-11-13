import mouette as M
from data import *

@pytest.mark.parametrize("s", TRI_SURFACES.items())
def test_cotan_laplacian(s):
    name, m = s
    mat = M.operators.laplacian(m, False)
    n = len(m.vertices)
    assert mat.shape == (n,n)
    
@pytest.mark.parametrize("s", list(ALL_SURFACES.items()) + list(POLYLINES.items()))
def test_graph_laplacian(s):
    name, m = s
    mat = M.operators.graph_laplacian(m)
    n = len(m.vertices)
    assert mat.shape == (n,n)

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_weight_matrix(s):
    name, m = s
    mat = M.operators.area_weight_matrix(m)
    n = len(m.vertices)
    assert mat.shape == (n,n)

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_weight_matrix_faces(s):
    name, m = s
    if len(m.faces)==1: 
        assert True
    else:
        mat = M.operators.area_weight_matrix_faces(m)
        mat2 = M.operators.area_weight_matrix_faces(m, inverse=True)
        n = len(m.faces)
        assert mat.shape == (n,n)
        assert mat2.shape == (n,n)

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_weight_matrix_edges(s):
    name, m = s
    mat = M.operators.area_weight_matrix_edges(m)
    mat2 = M.operators.area_weight_matrix_edges(m, inverse=True)
    n = len(m.edges)
    assert mat.shape == (n,n)
    assert mat2.shape == (n,n)

@pytest.mark.parametrize("s", list(ALL_SURFACES.items()) + list(POLYLINES.items()))
def test_adjacency_matrix(s):
    name, m = s
    mat = M.operators.adjacency_matrix(m)
    n = len(m.vertices)
    assert mat.shape == (n,n)

@pytest.mark.parametrize("s", TRI_SURFACES.items())
def test_gradient(s):
    name, m = s
    conn = M.processing.SurfaceConnectionFaces(m)
    grad = M.operators.gradient(m, conn)
    grad_no_complex = M.operators.gradient(m, conn, as_complex=False)
    signal = np.random.random(len(m.vertices))
    assert (grad @ signal).size == len(m.faces)
    assert (grad_no_complex @ signal).size == 2*len(m.faces)

def test_divgrad_equals_lap():
    # div(grad f) = laplacian(f)
    sphere = M.procedural.icosphere()
    local_bases = M.processing.SurfaceConnectionFaces(sphere)
    grad = M.operators.gradient(sphere, local_bases)
    div = grad.transpose().conjugate() @ M.operators.area_weight_matrix_faces(sphere)
    lap = M.operators.laplacian(sphere)
    signal = np.random.random(len(sphere.vertices))
    s1 = div @ grad @ signal
    s2 = lap @ signal
    assert np.allclose(s1, s2)