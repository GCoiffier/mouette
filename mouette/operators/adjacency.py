from ..mesh.datatypes import *
from .. import geometry as geom
import scipy.sparse as sp
import numpy as np

@forbidden_mesh_types(PointCloud)
def adjacency_matrix(mesh : Mesh, weights="one") -> sp.coo_matrix:
    """
    Computes and returns the adjacency matrix of a mesh, that is a (sparse) matrix M such that
        
        M[i,j] = M[j,i] = weights[i,j] if (i,j) is an edge of the mesh
        
        M[i,j] = 0 otherwise


    Args:
        mesh (Mesh): the input mesh
        weights (str, optional): How to weight the edges of the matrix. Options are:
            - "one" : every edge is 1
            - "length" : every edge has weight corresponding to its length
            - a custom dict edge_id:int -> weight:float
            Defaults to "one".

    Returns:
        scipy.sparse.coo_matrix
    """
    if not ( (isinstance(weights, str) and weights in ("one", "length")) or isinstance(weights, dict)):
        raise Exception("weights should be 'ones', 'length' or a custom dict")

    n = len(mesh.vertices)
    m = len(mesh.edges)

    # build matrix coefficients
    if weights == "one":
        vals = np.ones(2*m)
    elif weights == "length":
        vals = np.zeros(2*m)
        for e,(a,b) in enumerate(mesh.edges):
            d = geom.distance(mesh.vertices[a], mesh.vertices[b])
            vals[2*e] = d
            vals[2*e+1] = d
    else:
        vals = np.zeros(2*m)
        for e in mesh.id_edges:
            vals[2*e] = weights[e]
            vals[2*e+1] = weights[e]
    
    # build row and cols indices
    rows, cols = np.zeros(2*m), np.zeros(2*m)
    for e,(a,b) in enumerate(mesh.edges):
        rows[2*e] = a
        rows[2*e+1] = b
        cols[2*e] = b
        cols[2*e+1] = a

    return sp.coo_matrix( (vals, (rows,cols)), shape=(n,n))

@forbidden_mesh_types(PointCloud)
def vertex_to_edge_operator(mesh : Mesh, oriented:bool = False) -> sp.csc_matrix:
    """Vertices to edges operator. Matrix M of size |V|x|E| where:
        
        M[v,e] = 1 if and only if v is one extremity of edge e.

        if oriented is set, M[v,e] = -1 if v is the origin of the edge and 1 if it is the arrival.

    Args:
        mesh (Mesh): the input mesh
        oriented (bool): whether to consider oriented edge and signed values or not. Defaults to False.

    Returns:
        scipy.sparse.csc_matrix
    """
    n,m = len(mesh.vertices), len(mesh.edges)
    mat = sp.lil_matrix((n,m))
    orig_coeff = -1 if oriented else 1
    for e,(A,B) in enumerate(mesh.edges):
        mat[A,e] = orig_coeff
        mat[B,e] = 1
    return mat.tocsc()

@allowed_mesh_types(SurfaceMesh)
def vertex_to_face_operator(mesh : SurfaceMesh) -> sp.csc_matrix:
    """
    Vertices to face operator. Matrix M of size |V| x |F| where:

        M[v,f] = 1/len(f) if and only if v is one of the vertices of f

    Args:
        mesh (SurfaceMesh): input surface mesh

    Returns:
        scipy.sparse.csc_matrix
    
    See also:
        mouette.attributes.interpolate_vertices_to_faces
    """
    n,m = len(mesh.vertices), len(mesh.faces)
    mat = sp.lil_matrix((m,n))
    for iT,T in enumerate(mesh.faces):
        aT = 1/len(T)
        for V in T:
            mat[iT,V] = aT
    return mat.tocsc()