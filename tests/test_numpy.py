import mouette as M
import numpy as np

def test_mesh_from_arrays0():
    V = np.array([
        [0.,0.,.0],
        [1.,0.,0.],
        [0.,1.,0.]
    ])
    pc = M.mesh.from_arrays(V)
    assert isinstance(pc, M.mesh.PointCloud)
    assert len(pc.vertices)==3


def test_mesh_from_arrays1():
    V = np.array([
        [0.,0.,.0],
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.]
    ])
    E = np.array([
        [0,1],
        [0,2],
        [0,3],
        [1,2]
    ])
    pl = M.mesh.from_arrays(V, E)
    assert isinstance(pl, M.mesh.PolyLine)
    assert len(pl.vertices)==4
    assert len(pl.edges)==4

def test_mesh_from_arrays2():
    V = np.array([
        [0.,0.,.0],
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.]
    ])
    F = np.array([
        [0,1,2],
        [1,2,3],
        [0,2,3],
        [0,1,3]
    ])
    m = M.mesh.from_arrays(V, F=F)
    assert isinstance(m, M.mesh.SurfaceMesh)
    assert len(m.vertices) == 4
    assert len(m.faces) == 4
    assert len(m.edges) == 6 # automatically generated from faces

def test_mesh_from_arrays3():
    V = np.array([
        [0.,0.,.0],
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.]
    ])
    C = np.array([[0,1,2,3]])
    tet = M.mesh.from_arrays(V, C=C)
    assert isinstance(tet, M.mesh.VolumeMesh)
    assert len(tet.vertices)==4
    assert len(tet.edges)==6
    assert len(tet.faces)==4
    assert len(tet.cells)==1


def test_register_array():
    V = np.array([
        [0.,0.,.0],
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,2.,1.]
    ])
    pc = M.mesh.from_arrays(V)
    D = np.array([0.,1.,2.,3.])
    pc.vertices.register_array_as_attribute("D", D)
    D_attr = pc.vertices.get_attribute("D")
    assert D_attr[0] == 0.
    assert D_attr[2] == 2.