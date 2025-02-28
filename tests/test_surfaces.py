import mouette as M
from data.surfaces import *
from utils import *

def test_new():
    m = M.mesh.SurfaceMesh()
    assert len(m.vertices)==0
    assert len(m.edges)==0
    assert len(m.faces)==0

# ### Io tests ###

@pytest.mark.parametrize("s", surfaces)
def test_io_obj(s, tmp_path):
    build_test_io(s, tmp_path, "obj", 2)

@pytest.mark.parametrize("s", surfaces)
def test_io_geogram_ascii(s, tmp_path):
    build_test_io(s, tmp_path, "geogram_ascii", 2)

@pytest.mark.parametrize("s", surfaces)
def test_io_off(s, tmp_path):
    build_test_io(s, tmp_path, "off", 2)

@pytest.mark.parametrize("s", surfaces)
def test_io_medit(s, tmp_path):
    build_test_io(s, tmp_path, "mesh", 2)

# @pytest.mark.parametrize("s", surfaces)
# def test_io_ply(s, tmp_path):
#     build_test_io(s, tmp_path, "ply", 2)

### connectivity Tests ###

@pytest.mark.parametrize("s", [surf_triangle()])
def test_adj(s):
    assert s.connectivity.vertex_to_vertices(0) == [2,1]

@pytest.mark.parametrize("s", [surf_quad_subdiv()])
def test_sorted_neighborhood(s):
    assert s.connectivity.vertex_to_vertices(0) == [3, 1]
    assert s.connectivity.vertex_to_vertices(1) == [0, 4, 5, 2]
    assert s.connectivity.vertex_to_vertices(5) == [2, 1, 4, 8]
    assert s.connectivity.vertex_to_vertices(4) == [1, 3, 7, 5]

@pytest.mark.parametrize("s", surfaces)
def test_adj_V2V(s):
    _ = s.connectivity.vertex_to_vertices(0)
    assert True

@pytest.mark.parametrize("arg", [
    (surf_quad_subdiv(), 4, [3,0,1,2]),
    (surf_one_ring(), 0, [9, 8, 7, 6, 5, 4, 3, 2, 1,0])
])
def test_adj_V2F(arg):
    s,v,l = arg
    assert s.connectivity.vertex_to_faces(v) == l

@pytest.mark.parametrize("s", surfaces)
def test_adj_F2F(s):
    _ = s.connectivity.face_to_faces(0)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_adj_F2V(s):
    _ = s.connectivity.face_to_vertices(0)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_adj_F2E(s):
    _ = s.connectivity.face_to_edges(0)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_adj_F2V(s):
    _ = s.connectivity.face_to_vertices(0)
    assert True

### Boundary / Interior Tests ###

@pytest.mark.parametrize("s, expected", [(surf_square(), 4), (surf_cube(), 0)])
def test_bnd(s, expected):
    assert len(s.boundary_vertices) == expected

@pytest.mark.parametrize("s, expected", [(surf_square(), 0), (surf_cube(), 8)])
def test_interior(s, expected):
    assert len(s.interior_vertices) == expected
