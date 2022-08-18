import pygeomesh as GEO
from utils import *
from data import *

## Vertices
@pytest.mark.parametrize("s", surfaces + polylines)
def test_degree(s):
    deg = GEO.attributes.degree(s)
    deg = GEO.attributes.degree(s, persistent=False)
    deg = GEO.attributes.degree(s, dense=False)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_angle_defect(s):
    dfct = GEO.attributes.angle_defects(s)
    dfct = GEO.attributes.angle_defects(s, persistent=False)
    dfct = GEO.attributes.angle_defects(s, dense=False)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_vertex_normals(s):
    vn = GEO.attributes.vertex_normals(s)
    vn = GEO.attributes.vertex_normals(s, persistent=False)
    vn = GEO.attributes.vertex_normals(s, dense=False)
    assert True

@pytest.mark.parametrize("s", surfaces)
def test_border_normals(s):
    bn = GEO.attributes.border_normals(s) # not dense by default
    bn = GEO.attributes.border_normals(s, persistent=False)
    bn = GEO.attributes.border_normals(s, dense=True)
    assert True