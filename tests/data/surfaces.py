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
    

def surf_square_quad():
    return M.mesh.from_arrays(
        np.array([
            [0.,0.,0.],
            [0.,1.,0.],
            [1.,1.,0.],
            [1.,0.,0.],
    ]), F = np.array([[0,1,2,3]]))


def surf_square_triangulated():
    return M.mesh.from_arrays(
        np.array([
            [0.,0.,0.],
            [0.,1.,0.],
            [1.,1.,0.],
            [1.,0.,0.],
    ]), F = np.array([[0,1,3], [1,2,3]]))

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

CLOSED_TRI_SURFACES = {
    "cube_subdiv"        : M.mesh.load("tests/data/cube_subdiv.obj"), 
    "spot"               : M.mesh.load("tests/data/spot_1000.obj"), 
    "cube_triangulated"  : M.procedural.axis_aligned_cube(triangulate=True), 
    "tetrahedron"        : surf_tetrahedron(), 
    "torus_triangulated" : M.procedural.torus(triangulate=True)
}

TRI_SURFACES = {
    "triangle"            : surf_triangle(),
    "square_triangulated" : surf_square_triangulated(),
    "ring10"              : M.procedural.ring(10,0.1),
    "circle"              : M.mesh.load("tests/data/circle.mesh"),
    "half_sphere"         : M.mesh.load("tests/data/half_sphere1.obj"),
    "spline"              : M.mesh.load("tests/data/spline03.mesh"),
    "feature"             : M.mesh.load("tests/data/feature2.obj"),
    "two_pieces"          : M.mesh.load("tests/data/two_pieces.obj"),
    "pointy"              : M.mesh.load("tests/data/pointy.obj")
}
TRI_SURFACES.update(CLOSED_TRI_SURFACES)

QUAD_SURFACES = {
    "cube_quad"        : M.procedural.axis_aligned_cube(),
    "sphere_quad"      : M.mesh.load("tests/data/uv_sphere_nopoles_quad.obj"),
    "square_quad"      : surf_square_quad(),
    "torus_quad"       : M.procedural.torus(triangulate=False)
}

# IRREG_SURFACES = {
#     "square_irregular" : M.mesh.load("tests/data/quad_irregular.geogram_ascii")
# }
    
ALL_SURFACES = {}
ALL_SURFACES.update(TRI_SURFACES)
ALL_SURFACES.update(QUAD_SURFACES)
# ALL_SURFACES.update(IRREG_SURFACES)

