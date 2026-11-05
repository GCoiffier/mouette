import mouette as M
import pytest
import os

### Volumes ###

def vol_tetrahedron():
    return M.mesh.from_arrays(
        [M.Vec(0., 0., 0.),
        M.Vec(1., 0., 0.),
        M.Vec(0., 0., 1.),
        M.Vec(0., 1., 0.)],
        C=[[0,1,2,3]]
    )

def vol_join():
    return M.mesh.load(os.path.join("tests/data/join.tet"))

def vol_cube():
    return M.mesh.load(os.path.join("tests/data/cube86.mesh"))

def vol_cuboid():
    return M.mesh.load(os.path.join("tests/data/p01.mesh"))

volumes = [
    vol_tetrahedron(),
    vol_cuboid(),
    vol_join(),
    vol_cube(),
]