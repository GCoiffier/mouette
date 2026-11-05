import mouette as M
import pytest
import os
import numpy as np

### Polylines ###

def pl_one_edge():
    m = M.mesh.RawMeshData()
    m.vertices.set_data([[0.,0.,0.], [1.,0.,0.]])
    m.edges.set_data([[0,1]])
    return M.mesh.PolyLine(m)

def pl_sample1():
    m = M.mesh.RawMeshData()
    m.vertices.set_data([
        [0.,0.,0.],
        [1.,0.,1.],
        [2.,1.,0.]
    ])
    m.edges.set_data([[0,1], [1,2]])
    return M.mesh.PolyLine(m)

def pl_sample2():
    m = M.mesh.RawMeshData()
    m.vertices.set_data(np.random.random((6,3)))
    m.edges.set_data([[0,1], [1,5], [3,5], [2,3], [0,2], [0,3], [3,4], [2,4]])
    return M.mesh.PolyLine(m)

def pl_two_components():
    m = M.mesh.RawMeshData()
    m.vertices.set_data(np.random.random((10,3)))
    m.edges.set_data([[0,1], [1,4], [3,5], [2,3], [0,2], [0,3], [2,4], [6,7], [7,8], [7,9], [6,9]])
    return M.mesh.PolyLine(m)

polylines = [
    pl_one_edge(),
    pl_sample1(),
    pl_sample2(),
    pl_two_components()
]