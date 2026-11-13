import pytest
import mouette as M
from data.surfaces import *
from utils import *

def test_new():
    m = M.mesh.RawMeshData().instantiate(2)
    assert len(m.vertices)==0
    assert len(m.edges)==0
    assert len(m.faces)==0

# ### Io tests ###

@pytest.mark.parametrize("ext", ["obj", "geogram_ascii", "off", "mesh", "meshb", "stl", "ply", "vtk", "msh"])
@pytest.mark.parametrize("s", TRI_SURFACES.items())
def test_io_tri_compatible_formats(s, ext, tmp_path):
    name, s = s
    build_test_io(s, tmp_path, ext, 2)


@pytest.mark.parametrize("ext", ["obj", "geogram_ascii", "mesh", "meshb", "ply", "vtk", "msh"])
@pytest.mark.parametrize("s", QUAD_SURFACES.items())
def test_io_generic_formats(s, ext, tmp_path):
    name, s = s
    build_test_io(s, tmp_path, ext, 2)

@pytest.mark.parametrize("ext", ["vtk", "vtu"])
def test_io_vertex_attributes_roundtrip(ext, tmp_path):
    ### Make sure that the .vtk conversion is ok in terms of byte order (big-endian vs small-endian)
    m = surf_square_triangulated()
    attr = m.vertices.create_attribute("val", float)
    for i in range(len(m.vertices)): attr[i] = 0.5*i
    filepath = str(tmp_path / f"test_mesh.{ext}")
    M.mesh.save(m, filepath)
    m2 = M.mesh.load(filepath)
    assert m2.vertices.has_attribute("val")
    assert np.allclose(np.asarray(m2.vertices.get_attribute("val")).ravel(), [0., 0.5, 1., 1.5])

@pytest.mark.parametrize("ext", ["vtk", "vtu"])
def test_io_element_attributes_roundtrip(ext, tmp_path):
    ### Edge and face attributes should be loaded back on the right container
    m = surf_square_triangulated()
    fattr = m.faces.create_attribute("fval", float)
    fattr[0], fattr[1] = 1.5, 2.5
    eattr = m.edges.create_attribute("eval", int)
    for e in range(len(m.edges)): eattr[e] = e+1
    filepath = str(tmp_path / f"test_mesh.{ext}")
    M.mesh.save(m, filepath)
    m2 = M.mesh.load(filepath)
    assert m2.faces.has_attribute("fval") and not m2.edges.has_attribute("fval")
    assert m2.edges.has_attribute("eval") and not m2.faces.has_attribute("eval")
    assert np.allclose(np.asarray(m2.faces.get_attribute("fval")).ravel(), [1.5, 2.5])
    assert np.array_equal(np.asarray(m2.edges.get_attribute("eval")).ravel(), np.arange(1, len(m.edges)+1))

### connectivity Tests ###

def test_adj():
    s = ALL_SURFACES["triangle"]
    assert s.connectivity.vertex_to_vertices(0) == [2,1]

# @pytest.mark.parametrize("s", [surf_quad_irregular()])
# def test_sorted_neighborhood(s):
#     assert s.connectivity.vertex_to_vertices(0) == [3, 1]
#     assert s.connectivity.vertex_to_vertices(1) == [0, 4, 5, 2]
#     assert s.connectivity.vertex_to_vertices(5) == [2, 1, 4, 8]
#     assert s.connectivity.vertex_to_vertices(4) == [1, 3, 7, 5]

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_adj_V2V(s):
    name, mesh = s
    _ = mesh.connectivity.vertex_to_vertices(0)
    assert True

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_adj_F2F(s):
    name, mesh = s
    _ = mesh.connectivity.face_to_faces(0)
    assert True

def test_adj_V2F():
    s = M.procedural.ring(10, 0.1)
    assert s.connectivity.vertex_to_faces(0) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_adj_F2V(s):
    name, mesh = s
    _ = mesh.connectivity.face_to_vertices(0)
    assert True

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_adj_F2E(s):
    name, mesh = s
    _ = mesh.connectivity.face_to_edges(0)
    assert True

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_adj_F2V(s):
    name, mesh = s
    _ = mesh.connectivity.face_to_vertices(0)
    assert True

### Boundary / Interior Tests ###

@pytest.mark.parametrize("mesh, expected_bnd, expected_in", [
    (ALL_SURFACES["square_triangulated"], 4, 0), 
    (ALL_SURFACES["cube_quad"], 0, 8)
])
def test_bnd(mesh, expected_bnd, expected_in):
    assert len(mesh.boundary_vertices) == expected_bnd
    assert len(mesh.interior_vertices) == expected_in



### Reordering
def test_reorder_vertices():
    m = ALL_SURFACES["circle"]
    permut = np.random.permutation(len(m.vertices))
    permuted = M.mesh.reorder_vertices(m, permut)
    compare_mesh(m, permuted, reorder_vertices=True)