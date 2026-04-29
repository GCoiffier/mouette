import mouette as M
from utils import *
from data import *

def _make_test_quantity(mesh, attr):
    val = attr(mesh)
    val = attr(mesh, persistent=False)
    val = attr(mesh, persistent=False, dense=False)

##### Vertices 

@pytest.mark.parametrize("s", surfaces + polylines)
def test_degree(s):
    _make_test_quantity(s, M.attributes.degree)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_angle_defect(s):
    _make_test_quantity(s, M.attributes.angle_defects)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_vertex_normals(s):
    _make_test_quantity(s, M.attributes.vertex_normals)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_border_normals(s):
    _make_test_quantity(s, M.attributes.border_normals)
    assert True

##### Edges
@pytest.mark.parametrize("s", surfaces)
def test_edge_length(s):
    _make_test_quantity(s, M.attributes.edge_length)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_edge_middle_point(s):
    _make_test_quantity(s, M.attributes.edge_middle_point)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_curvature_matrices(s):
    _ = M.attributes.curvature_matrices(s)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_cotan_weights(s):
    _make_test_quantity(s, M.attributes.cotan_weights)
    assert True


##### Faces

@pytest.mark.parametrize("s", surfaces)
def test_face_area(s):
    _make_test_quantity(s, M.attributes.face_area)
    assert True


@pytest.mark.parametrize("s", surfaces)
def test_face_normals(s):
    _make_test_quantity(s, M.attributes.face_normals)
    assert True


@pytest.mark.parametrize("s", surfaces)
def test_face_barycenter(s):
    _make_test_quantity(s, M.attributes.face_barycenter)
    assert True


@pytest.mark.parametrize("s", surfaces)
def test_face_circumcenter(s):
    _make_test_quantity(s, M.attributes.face_circumcenter)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_face_near_border(s):
    _make_test_quantity(s, M.attributes.face_near_border)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_triangle_aspect_ratio(s):
    _make_test_quantity(s, M.attributes.triangle_aspect_ratio)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_parallel_transport_curvature(s):
    conn = M.processing.SurfaceConnectionVertices(s)
    val = M.attributes.parallel_transport_curvature(s, conn)
    val = M.attributes.parallel_transport_curvature(s, conn, persistent=False)
    val = M.attributes.parallel_transport_curvature(s, conn, persistent=False, dense=False)
    assert True


##### Corners

@pytest.mark.parametrize("s", surfaces)
def test_corner_angles(s):
    _make_test_quantity(s, M.attributes.corner_angles)
    assert True


@pytest.mark.parametrize("s", surfaces)
def test_cotangent(s):
    _make_test_quantity(s, M.attributes.cotangent)
    assert True

##### Cells

@pytest.mark.parametrize("s", volumes)
def test_cell_volume(s):
    _make_test_quantity(s, M.attributes.cell_volume)
    assert True

@pytest.mark.parametrize("s", volumes)
def test_cell_barycenter(s):
    _make_test_quantity(s, M.attributes.cell_barycenter)
    assert True

@pytest.mark.parametrize("s", volumes)
def test_cell_faces_on_boundary(s):
    _make_test_quantity(s, M.attributes.cell_faces_on_boundary)
    assert True

##### Global quantities 

@pytest.mark.parametrize("s", surfaces)
def test_euler_characteristic(s):
    e = M.attributes.euler_characteristic(s)
    assert True

@pytest.mark.parametrize("s", surfaces + polylines)
def test_mean_edge_length(s):
    m = M.attributes.mean_edge_length(s)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_mean_face_area(s):
    m = M.attributes.mean_face_area(s)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_total_area(s):
    m = M.attributes.total_area(s)
    assert True

@pytest.mark.parametrize("s", volumes)
def test_mean_cell_volume(s):
    m = M.attributes.mean_cell_volume(s)
    assert True

@pytest.mark.parametrize("s", surfaces+point_clouds)
def test_barycenter(s):
    m = M.attributes.barycenter(s)
    assert True