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