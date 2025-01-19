from ..mesh.datatypes import *
from ..operators.laplacian_op import laplacian
from ..geometry import distance

import numpy as np
import scipy.sparse as sp

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