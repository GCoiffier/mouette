import mouette as M
from utils import *
from data import *

## Vertices
@pytest.mark.parametrize("s", surfaces + polylines)
def test_degree(s):
    deg = M.attributes.degree(s)
    deg = M.attributes.degree(s, persistent=False)
    deg = M.attributes.degree(s, dense=False)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_angle_defect(s):
    dfct = M.attributes.angle_defects(s)
    dfct = M.attributes.angle_defects(s, persistent=False)
    dfct = M.attributes.angle_defects(s, dense=False)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_vertex_normals(s):
    vn = M.attributes.vertex_normals(s)
    vn = M.attributes.vertex_normals(s, persistent=False)
    vn = M.attributes.vertex_normals(s, dense=False)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_border_normals(s):
    bn = M.attributes.border_normals(s) # not dense by default
    bn = M.attributes.border_normals(s, persistent=False)
    bn = M.attributes.border_normals(s, dense=True)
    assert True

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