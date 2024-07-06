import mouette as M
import numpy as np
from data import *

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

def test_triangle():
    P0 = M.Vec(0,0,0)
    P1 = M.Vec(1,0,0)
    P2 = M.Vec(0,1,1)
    tri = M.procedural.triangle(P0,P1,P2)
    assert len(tri.faces)==1

def test_quad():
    P0 = M.Vec(1,0,0)
    P1 = M.Vec(2,0,0)
    P2 = M.Vec(0,1,1)
    quad = M.procedural.quad(P0,P1,P2)
    assert np.all(quad.vertices[2] ==  P1 + P2 - P0)
    assert len(quad.faces) == 1
    quadtri = M.procedural.quad(P0, P1, P2, triangulate=True)
    assert len(quadtri.faces) == 2

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

def test_octahedron():
    oct = M.procedural.octahedron()
    assert len(oct.vertices) == 6

def test_icosahedron():
    ico = M.procedural.icosahedron()
    assert len(ico.vertices)==12
    assert len(ico.faces)==20

def test_dodecahedron():
    dod = M.procedural.dodecahedron()
    assert len(dod.vertices) == 20
    assert len(dod.faces) == 12

def test_torus():
    torus = M.procedural.torus(50, 20, 1.,0.2)
    assert len(torus.vertices)==50*20
    assert len(torus.faces)==50*20

    torus2 = M.procedural.torus(30, 10, 1.,0.2, triangulate=True)
    assert len(torus2.vertices) == 30*10
    assert len(torus2.faces) == 2*30*10

def test_unit_grid():
    grid = M.procedural.unit_grid(10,10,generate_uvs=True)
    assert grid.vertices.has_attribute("uv_coords")
    assert len(grid.vertices)==10*10
    assert len(grid.faces)==9*9

    grid2 = M.procedural.unit_grid(10,10,triangulate=True)
    assert len(grid2.vertices)==10*10
    assert len(grid2.faces)==2*9*9

def test_unit_triangle():
    tri = M.procedural.unit_triangle(10,10, generate_uvs=True)
    assert len(tri.vertices)==(10*11)//2

def test_sphere_uv():
    sph = M.procedural.sphere_uv(20,30,radius=1.2)
    assert len(sph.vertices) == 20*30 + 2  # don't forget the poles

def test_sphere_ico():
    sph = M.procedural.icosphere(3, M.Vec.random(3),1.2)
    assert len(sph.vertices) == 942

def test_sphere_fibonacci():
    sph = M.procedural.sphere_fibonacci(300, build_surface=True)
    assert len(sph.vertices) == 300

@pytest.mark.parametrize("p", point_clouds)
def test_spherify_vertices(p):
    sph = M.procedural.spherify_vertices(p)
    assert True
