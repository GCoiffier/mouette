import mouette as M
import numpy as np

### Point Clouds ###

def pc_sample1():
    return M.mesh.from_arrays([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])

def pc_grid():
    V = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                V.append([float(x), float(y), float(z)])
    return M.mesh.from_arrays(V)

POINT_CLOUDS = {
    "pc1"           : pc_sample1(),
    "pc_random10"   : M.mesh.from_arrays(np.random.random((10,3))),
    "pc_random1000" : M.mesh.from_arrays(np.random.random((1000,3))),
    "pc_grid_5^3"   : pc_grid()
}