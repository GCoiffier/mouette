import mouette as M
import pytest
import os

### Volumes ###

def vol_tetrahedron():
    m = M.mesh.new_volume()
    m.vertices += [
        M.Vec(0., 0., 0.),
        M.Vec(1., 0., 0.),
        M.Vec(0., 0., 1.),
        M.Vec(0., 1., 0.),
    ]
    m.edges += [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]
    m.faces += [[0,1,2], [0,1,3], [1,2,3], [2,0,3]]
    m.cells.append([0,1,2,3])
    return m

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