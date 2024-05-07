import numpy as np
import scipy.sparse as sp
import cmath

from ..geometry import *
from ..processing.connection import SurfaceConnectionFaces
from ..mesh.datatypes import *
from ..attributes.misc_faces import face_area

@allowed_mesh_types(SurfaceMesh)
def gradient(mesh: SurfaceMesh, conn: SurfaceConnectionFaces = None, as_complex: bool = True) -> sp.csc_matrix:
    """
    Computes the gradient operator, i.e. a |F| x |V| matrix G such that for any scalar function f defined over vertices of a surface mesh, Gf is its gradient inside faces.

    Gf maps to each face of the mesh either a vector of $\mathbb{R}^2$ or a complex number representing the gradient vector inside this face in local base.

    Args:
        mesh (SurfaceMesh): The input mesh
        conn (SurfaceConnectionFaces, optional): Connection objects specifying local bases of faces. Will be computed if not provided. Defaults to None.
        as_complex (bool, optional): whether the output is |F| complex values of 2|F| float values

    Raises:
        Exception: Fails if the mesh is not a triangulation
    
    Returns:
        scipy.sparse.csc_matrix: Gradient operator
    """
    if not mesh.is_triangular():
        raise Exception("Mesh is not a triangulation")
    if conn is None : conn = SurfaceConnectionFaces(mesh)
    area = face_area(mesh)

    N = len(mesh.vertices)
    M = len(mesh.faces)

    if as_complex:
        cols, rows, vals = np.zeros(3*M, dtype=int), np.zeros(3*M, dtype=int), np.zeros(3*M, dtype=complex)

        for iT,(A,B,C) in enumerate(mesh.faces):
            aT = 2*area[iT]
            xA,yA = conn.project(mesh.vertices[A], iT)
            xB,yB = conn.project(mesh.vertices[B], iT)
            xC,yC = conn.project(mesh.vertices[C], iT)

            rows[3*iT],   cols[3*iT],   vals[3*iT]   = iT, A, complex(yB-yC, xC-xB) / aT
            rows[3*iT+1], cols[3*iT+1], vals[3*iT+1] = iT, B, complex(yC-yA, xA-xC) / aT
            rows[3*iT+2], cols[3*iT+2], vals[3*iT+2] = iT, C, complex(yA-yB, xB-xA) / aT

    else:
        cols, rows, vals = np.zeros(6*M, dtype=int), np.zeros(6*M, dtype=int), np.zeros(6*M, dtype=float)
        for iT,(A,B,C) in enumerate(mesh.faces):
            aT = 2*area[iT]
            xA,yA = conn.project(mesh.vertices[A], iT)
            xB,yB = conn.project(mesh.vertices[B], iT)
            xC,yC = conn.project(mesh.vertices[C], iT)

            rows[6*iT],   cols[6*iT],   vals[6*iT]   = 2*iT,   A, (yB-yC) / aT
            rows[6*iT+1], cols[6*iT+1], vals[6*iT+1] = 2*iT+1, A, (xC-xB) / aT
            rows[6*iT+2], cols[6*iT+2], vals[6*iT+2] = 2*iT,   B, (yC-yA) / aT
            rows[6*iT+3], cols[6*iT+3], vals[6*iT+3] = 2*iT+1, B, (xA-xC) / aT
            rows[6*iT+4], cols[6*iT+4], vals[6*iT+4] = 2*iT,   C, (yA-yB) / aT
            rows[6*iT+5], cols[6*iT+5], vals[6*iT+5] = 2*iT+1, C, (xB-xA) / aT
        M *= 2
    return sp.csc_matrix((vals, (rows,cols)), shape=(M,N))