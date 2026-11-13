import mouette as M
import pytest
import os

### Volumes ###

def vol_tetrahedron():
    return M.mesh.from_arrays(
        [[0., 0., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.]],
        C=[[0,1,2,3]]
    )

VOLUMES = {
    "tetrahedron" : vol_tetrahedron(),
    "join"        : M.mesh.load(os.path.join("tests/data/join.tet")),
    "cuboid"      : M.mesh.load(os.path.join("tests/data/p01.mesh")),
    "cube"        : M.mesh.load(os.path.join("tests/data/cube86.mesh")),
}