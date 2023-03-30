from ..mesh.datatypes import *
from ..attributes.misc_vertices import degree
from ..operators.laplacian import laplacian
from ..geometry import distance
from .combinatorics import SurfaceSubdivision

import numpy as np
import scipy.sparse as sp

@allowed_mesh_types(SurfaceMesh)
def split_double_boundary_edges_triangles(mesh : SurfaceMesh) -> SurfaceMesh:
    """
    A triangle with double edge on the boundary can occur on the border in a case like :

        /\\ 
    ___/__\\___

    This function detects every occurrences of such a configuration and split the problematic
    triangle in three by adding a new vertex in the middle

    Parameters:
        mesh (Mesh): the mesh (modified in place)

    Raises:
        Exception: Isolated vertex
        Raised when a vertex of the mesh has degree < 2, which mean that the mesh is not manifold
    Returns:
        SurfaceMesh: the modified input mesh
    """
        
    deg = degree(mesh, "_degree", persistent=False)
    pb_faces = []
    for i,f in enumerate(mesh.faces):
        for v in f:
            if deg[v]<2: 
                raise Exception("Isolated vertex")
            if deg[v]==2:
                pb_faces.append(i)
                break
    if pb_faces:
        with SurfaceSubdivision(mesh) as subdv:
            for f in pb_faces: # Triangulate face with a vertex in the middle
                subdv.split_face_as_fan(f)
    return mesh

@allowed_mesh_types(SurfaceMesh)
def smooth(mesh : SurfaceMesh, n_iter=100, mode="cotan", keep_border=False, damping=0.0) -> SurfaceMesh:
    """
    Applies Laplacian smoothing to a mesh

    Parameters:
        mesh (Mesh): 
            The mesh to smooth. Operation is done in place
        
        n_iter (int, optional): 
            Number of steps. Defaults to 100.
        
        mode (str, optional): 
            if "cotan": uses cotan expression of Laplace-Beltrami operator. 
            if "connectivity": uses simple connectivity Laplacian.
            Defaults to "cotan".
        
        keep_border (bool, optional):
            Fix the boundary. Defaults to False.

        damping (float, optional): 
            Damping coefficient. Will consider matrix (L+ + damping*Id). 
            Defaults to 0.0.

    Returns:
        (SurfaceMesh): the smoothed input mesh
    """
    
    A = laplacian(mesh, cotan=(mode=="cotan"))

    if damping>0.:
        A = damping*A + (1-damping)*sp.eye(A.shape[0])
    V = np.array(mesh.vertices)
    dist_begin = distance(V[0], V[1])

    if keep_border:
        mask = np.array([not mesh.is_vertex_on_border(i) for i in mesh.id_vertices]).astype(bool)

    for _ in range(n_iter):
        if keep_border:
            V[mask] = A.dot(V)[mask]
        else:
            V = A.dot(V)

    dist_end = distance(V[0], V[1])
    vA =  V[0]
    V = (V-vA)*(dist_begin/dist_end) + vA

    for i in mesh.id_vertices:
        mesh.vertices[i] = V[i,:]
    return mesh