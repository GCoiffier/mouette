import mouette as M
import numpy as np

### Surfaces ###

def surf_triangle():
    return M.mesh.from_arrays(
        np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
        ]), F=np.array([[0,1,2]]))
    

def surf_square():
    return M.mesh.from_arrays(
        np.array([
            [1.,0.,0.],
            [0.,1.,0.],
            [1.,1.,0.],
            [1.,0.,0.],
    ]), F = np.array([[0,1,3], [1,2,3]]))


def surf_cube():
    return M.mesh.from_arrays(
        np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            [0., 1., 1.],
        ]),
        F = np.array([
        [0,2,1], [0,3,2],
        [0,1,5], [0,5,4],
        [1,2,6], [1,6,5],
        [2,3,7], [2,7,6],
        [3,0,4], [3,4,7],
        [4,5,6], [4,6,7]
        ])
    )


def surf_tetrahedron():
    return M.mesh.from_arrays(
        np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ]),
        F=np.array([
            [0,1,2],
            [0,1,3],
            [1,2,3],
            [2,0,3]
        ])
    )

def surf_one_ring():
    return M.procedural.ring(10,0.1)

def surf_circle():
    return M.mesh.load("tests/data/circle.mesh")

def surf_half_sphere():
    return M.mesh.load("tests/data/half_sphere1.obj")

def surf_spline():
    return M.mesh.load("tests/data/spline03.mesh")

def surf_feat():
    return M.mesh.load("tests/data/feature2.obj")

def surf_two_pieces():
    return M.mesh.load("tests/data/two_pieces.obj")

def surf_pointy():
    return M.mesh.load("tests/data/pointy.obj")

def surf_quad_subdiv():
    return M.mesh.load("tests/data/quad_split2.geogram_ascii")

def surf_spot():
    return M.mesh.load("tests/data/spot_1000.obj")

def surf_cube_subdiv():
    return M.mesh.load("tests/data/cube_subdiv.obj")

def surf_torus(triangulate=True):
    return M.procedural.torus(triangulate=triangulate)

def surf_uv_sphere_quads():
    # example of a quad mesh
    return M.mesh.load("tests/data/uv_sphere_nopoles_quad.obj")

# all surface meshes
surfaces = [
    surf_triangle(),
    surf_one_ring(),
    surf_circle(),
    surf_half_sphere(),
    surf_spline(),
    surf_feat(),
    surf_two_pieces(),
    surf_pointy(),
    surf_cube_subdiv(),
    surf_torus(),
    surf_uv_sphere_quads()
]

# surface meshes that are closed (no boundary) and triangulated
closed_triangle_surfaces = [surf_cube_subdiv(), surf_spot(), surf_cube(), surf_tetrahedron(), surf_torus()]
