import numpy as np
from math import pi, cos, sin
from ..geometry import Vec, angle_3pts, rotate_2d
from ..mesh.datatypes import SurfaceMesh
from ..mesh import mesh
from ..mesh.mesh_data import RawMeshData

def ring(N : int, defect : float, open :bool = False, n_cover:int = 1) -> SurfaceMesh:
    """
    Computes a ring of triangles with prescribed number of triangles and angle defect at the center.

    Parameters:
        N (int): number of triangles in the ring
        defect (float): target angle defect to achieve. Position of the central point is adjusted via dichotomy to match this value.
        open (bool) : whether to connect the last vertex to the first.
        n_cover (int, optional): Number of covering of the ring. Defaults to 1.

    Raises:
        Exception: Fails if N<3

    Returns:
        SurfaceMesh: the ring
    """

    V, F = [], []
    max_defect = 2*pi-0.01
    defect = max(min(defect,max_defect), 0.) # 0 is ok, but 2pi is point at infinity

    if N<3:
        raise Exception("N should be > 3 for a valid ring. Aborting")

    V.append(Vec(0.,0.,0.))
    V.append(Vec(1.,0.,0.))

    for i in range(1, N*n_cover):
        V.append(Vec( cos(2*i*pi/N), sin(2*i*pi/N), 0.))
        nxt = i+1 if open else (i+1)%(N*n_cover+1)
        F.append((0, i, nxt))
    if open:
        V.append(V[1])
        F.append((0, N*n_cover, N*n_cover+1))
    else:
        F.append((0, N*n_cover, 1))

    # Determine position of center vertex through dichotomy
    P1 = Vec(0.,0.,0.)
    P2 = Vec(0.,0.,10.)
    A = V[1]
    B = V[2]
    stop = False
    while not stop:
        dfct1 = 2*pi - N*angle_3pts(A,P1,B)
        dfct2 = 2*pi - N*angle_3pts(A,P2,B)

        Pmid = (P1 + P2)/2
        middfct = 2*pi - N*angle_3pts(A,Pmid,B)
        if defect>dfct2:
            P1 = P2
            P2 = 2*P2
        elif defect>middfct:
            P1 = Pmid
        elif defect<middfct:
            P2 = Pmid
        stop = (abs(dfct1 - dfct2) < 1e-6)
    V[0] = (P1 + P2)/2
    return mesh.from_arrays(V, F=F)

def flat_ring(N : int, defect : float, n_cover:int = 1) -> SurfaceMesh:
    V, E, F = [], [], []
    max_defect = 2*pi-0.01
    defect = max(min(defect,max_defect), 0.) # 0 is ok, but 2pi is point at infinity

    V.append(Vec(0.,0.,0.))
    
    dir = Vec(1.,0.,0.)
    V.append(dir)
    E.append((0,1))
    ang = (2*pi-defect)/N
    for i in range(N*n_cover):
        dir = rotate_2d(dir, ang)
        dir = Vec(dir.x, dir.y, 0.)
        V.append(dir)
        F.append((0,i+1,i+2))
    return mesh.from_arrays(V, E=E, F=F)
