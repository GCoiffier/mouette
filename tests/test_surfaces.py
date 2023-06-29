import mouette as M
from data.surfaces import *
from utils import *

import pytest

def test_new():
    m = M.mesh.empty()
    m = M.mesh.instanciate(m, dim=2)
    assert len(m.vertices)==0
    assert len(m.edges)==0
    assert len(m.faces)==0

def test_copy():
    m = M.mesh.empty()
    m.vertices += [(0.,0.,0.), (0.,1.,0.), (0.,0.,1.)]
    m.faces += [(0,1,2)]
    m = M.mesh.instanciate(m, dim=2)
    m2 = M.mesh.copy(m)
    assert compare_mesh(m,m2)

# ### Io tests ###

@pytest.mark.parametrize("s", surfaces)
def test_io_obj(s, tmp_path):
    assert build_test_io(s, tmp_path, "obj", 2)

@pytest.mark.parametrize("s",  surfaces)
def test_io_geogram_ascii(s, tmp_path):
    assert build_test_io(s, tmp_path, "geogram_ascii", 2)

@pytest.mark.parametrize("s", surfaces)
def test_io_off(s, tmp_path):
    assert build_test_io(s, tmp_path, "off", 2)

@pytest.mark.parametrize("s", surfaces)
def test_io_medit(s, tmp_path):
    assert build_test_io(s, tmp_path, "mesh", 2)

### connectivity Tests ###

@pytest.mark.parametrize("s", [surf_triangle()])
def test_adj(s):
    assert s.connectivity.vertex_to_vertex(0) == [1,2]

@pytest.mark.parametrize("s", surfaces)
def test_adj_V2V(s):
    _ = s.connectivity.vertex_to_vertex(0)
    _ = s.connectivity.n_VtoV(0)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_adj_V2F(s):
    _ = s.connectivity.vertex_to_face(0)
    _ = s.connectivity.n_VtoF(0)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_adj_F2F(s):
    _ = s.connectivity.face_to_face(0)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_adj_F2V(s):
    _ = s.connectivity.face_to_vertex(0)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_adj_F2E(s):
    _ = s.connectivity.face_to_edge(0)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_adj_F2V(s):
    _ = s.connectivity.face_to_vertex(0)
    assert True

### Boundary / Interior Tests ###

@pytest.mark.parametrize("s, expected", [(surf_square(), 4), (surf_cube(), 0)])
def test_bnd(s, expected):
    assert len(s.boundary_vertices) == expected

@pytest.mark.parametrize("s, expected", [(surf_square(), 0), (surf_cube(), 8)])
def test_interior(s, expected):
    assert len(s.interior_vertices) == expected
