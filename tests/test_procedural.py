import pygeomesh as GEO
from data.polylines import *

def test_rings():
    r = GEO.procedural.ring(10, 0.1, False, 1)
    assert len(r.vertices)==11
    assert len(r.faces) == 10
    r = GEO.procedural.ring(10, 0.3, True, 2)
    assert len(r.vertices) == 22
    assert len(r.faces) == 20

def test_flat_ring():
    r = GEO.procedural.flat_ring(10, 0.2, 1)
    assert len(r.vertices) == 12
    assert len(r.faces) == 10

def test_cylinder():
    cy1 = GEO.procedural.cylinder(GEO.Vec.random(3), GEO.Vec.random(3), N=10)
    assert len(cy1.vertices) == 2*10 + 2
    cy2 = GEO.procedural.cylinder(GEO.Vec.random(3), GEO.Vec.random(3), N=20, fill_caps=False)
    assert len(cy2.vertices) == 2*20

@pytest.mark.parametrize("p", polylines)
def test_cylindrify(p):
    m = GEO.procedural.cylindrify_edges(p, N=10)
    assert isinstance(m, GEO.mesh.SurfaceMesh)

def test_aacube():
    m = GEO.procedural.axis_aligned_cube()
    assert len(m.vertices) == 8
    m = GEO.procedural.axis_aligned_cube(colored=True)
    assert len(m.vertices) == 8

def test_hexa():
    P1, P2, P3, P4 = GEO.Vec.random(3), GEO.Vec.random(3), GEO.Vec.random(3), GEO.Vec.random(3)
    m = GEO.procedural.hexahedron_4pts(P1,P2,P3,P4)
    assert len(m.vertices) == 8
    m = GEO.procedural.hexahedron_4pts(P1,P2,P3,P4, colored=True, volume=True)
    assert len(m.vertices) == 8

