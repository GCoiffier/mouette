import mouette as M
from data.polylines import *

def test_rings():
    r = M.procedural.ring(10, 0.1, False, 1)
    assert len(r.vertices)==11
    assert len(r.faces) == 10
    r = M.procedural.ring(10, 0.3, True, 2)
    assert len(r.vertices) == 22
    assert len(r.faces) == 20

def test_flat_ring():
    r = M.procedural.flat_ring(10, 0.2, 1)
    assert len(r.vertices) == 12
    assert len(r.faces) == 10

def test_cylinder():
    cy1 = M.procedural.cylinder(M.Vec.random(3), M.Vec.random(3), N=10)
    assert len(cy1.vertices) == 2*10 + 2
    cy2 = M.procedural.cylinder(M.Vec.random(3), M.Vec.random(3), N=20, fill_caps=False)
    assert len(cy2.vertices) == 2*20

@pytest.mark.parametrize("p", polylines)
def test_cylindrify(p):
    m = M.procedural.cylindrify_edges(p, N=10)
    assert isinstance(m, M.mesh.SurfaceMesh)

def test_aacube():
    m = M.procedural.axis_aligned_cube()
    assert len(m.vertices) == 8
    m = M.procedural.axis_aligned_cube(colored=True)
    assert len(m.vertices) == 8

def test_hexa():
    P1, P2, P3, P4 = M.Vec.random(3), M.Vec.random(3), M.Vec.random(3), M.Vec.random(3)
    m = M.procedural.hexahedron_4pts(P1,P2,P3,P4)
    assert len(m.vertices) == 8
    m = M.procedural.hexahedron_4pts(P1,P2,P3,P4, colored=True, volume=True)
    assert len(m.vertices) == 8

