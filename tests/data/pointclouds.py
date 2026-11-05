import mouette as M
import numpy as np

### Point Clouds ###

def pc_sample1():
    return M.mesh.from_arrays(np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]))

def pc_random10():
    return M.mesh.from_arrays(np.random.random((10,3)))

def pc_grid():
    vertices = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                vertices.append(M.Vec(float(x), float(y), float(z)))
    return M.mesh.from_arrays(np.ascontiguousarray(vertices))

point_clouds = [
    pc_sample1(),
    pc_random10(),
    pc_grid()
]