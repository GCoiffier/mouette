import mouette as M
import numpy as np

### Polylines ###

def pl_one_edge():
    return M.mesh.from_arrays(V=[[0.,0.,0.], [1.,0.,0.]], E=[[0,1]])

def pl_sample1():
    return M.mesh.from_arrays(V=[[0.,0.,0.], [1.,0.,1.], [2.,1.,0.]], E=[[0,1], [1,2]])


def pl_sample2():
    return M.mesh.from_arrays(np.random.random((6,3)), E=[[0,1], [1,5], [3,5], [2,3], [0,2], [0,3], [3,4], [2,4]])

def pl_two_components():
    return M.mesh.from_arrays(np.random.random((10,3)), E=[[0,1], [1,4], [3,5], [2,3], [0,2], [0,3], [2,4], [6,7], [7,8], [7,9], [6,9]])

POLYLINES = {
    "one_edge"   :  pl_one_edge(),
    "polyline1"  :  pl_sample1(),
    "polyline2"  :  pl_sample2(),
    "pl_two_cmp" :  pl_two_components()
}