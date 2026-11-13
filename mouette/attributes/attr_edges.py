from typing import Union

from ..mesh.mesh_attributes import SparseAttribute, Attribute
from ..mesh.datatypes import *
from .attr_corners import cotangent
from .. import geometry as geom
from ..geometry import Vec

import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True)
def _edge_length(V, E):
    n = E.shape[0]
    lengths = np.empty(n, dtype=float)
    for i in prange(n):
        lengths[i] = np.linalg.norm(V[E[i,0]] - V[E[i,1]])
    return lengths

@forbidden_mesh_types(PointCloud)
def edge_length(mesh : Mesh, name = "length", persistent:bool=True) -> Union[Attribute, np.ndarray]:
    """Compute edge lengths across the mesh.

    Parameters:
        mesh (Mesh): the input mesh. PointClouds are forbidden (no edges).
        name (str, optional): name of the attribute. Defaults to "length".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
    """
    lengths = _edge_length(mesh.vertices._data, mesh.edges._data)
    if persistent:
        return mesh.edges.register_array_as_attribute(name, lengths)
    return lengths


@njit(cache=True, parallel=True)
def _edge_middle_point(V, E):
    n = E.shape[0]
    mid = np.empty((n,3), dtype=float)
    for i in prange(n):
        mid[i] = (V[E[i,0]] + V[E[i,1]])/2
    return mid

@forbidden_mesh_types(PointCloud)
def edge_middle_point(mesh : SurfaceMesh, name="middle", persistent: bool = True) -> Union[Attribute, np.ndarray]:
    """Compute the middle point of each edge across the mesh.

    Parameters:
        mesh (Mesh): the input mesh. PointClouds are forbidden (no edges).
        name (str, optional): name of the attribute. Defaults to "middle".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
    """
    middle = _edge_length(mesh.vertices._data, mesh.edges._data)
    if persistent:
        return mesh.edges.register_array_as_attribute(name, middle)
    return middle


@allowed_mesh_types(SurfaceMesh)
def curvature_matrices(mesh : SurfaceMesh) -> np.ndarray:
    """
    Curvature matrix for each edge on the mesh
    
    References:
        _Restricted Delaunay Triangulations and Normal Cycle_, David Cohen-Steiner and Jean-Marie Morvan (2003)

    Note:
        See Curvature Frame Fields for their aggregation on triangles or vertices

    Returns:
        np.ndarray: an array of shape (V,3,3) containing all the matrices
    """
    mats = np.zeros((len(mesh.edges), 3,3), dtype=np.float64)
    for e, (A,B) in enumerate(mesh.edges):
        T1,T2 = mesh.connectivity.edge_to_faces(A,B)
        if T1 is not None and T2 is not None:
            pA, pB = mesh.vertices[A], mesh.vertices[B]
            _,_,n1 = geom.face_basis(*(mesh.vertices[u] for u in mesh.faces[T1]))
            _,_,n2 = geom.face_basis(*(mesh.vertices[u] for u in mesh.faces[T2]))
            # angle = geom.signed_angle_2vec3D(n1,n2,n2-n1)
            angle = geom.angle_2vec3D(n1,n2)
            edge = Vec.normalized(pB-pA)
            mats[e,:,:] = angle * geom.distance(pA,pB) * np.outer(edge,edge)
    return mats


@allowed_mesh_types(SurfaceMesh)
def cotan_weights(mesh : SurfaceMesh, name="cotan_weight", persistent: bool = True)-> Union[Attribute, np.ndarray]:
    """ Compute the cotan weights of edges.
    The weight of an edge separating T1 and T2 is the sum of cotangent of opposite angles in T1 and T2

    Parameters:
        mesh (Mesh): the input mesh.
        name (str, optional): name of the attribute. Defaults to "cotan_weight".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
    
    Raises:
        Exception: fails if the mesh is not a triangular mesh.
    """
    if not mesh.is_triangular():
        raise Exception("Tried to compute cotan weights on a non-triangulated mesh")

    if mesh.face_corners.has_attribute("cotan"):
        cot = mesh.face_corners.get_attribute("cotan").as_array()
    elif persistent:
        cot = cotangent(mesh, persistent=True).as_array()
    else:
        cot = cotangent(mesh, persistent=False)
    
    cw = np.zeros(len(mesh.edges), dtype=float)
    for e,(A,B) in enumerate(mesh.edges):
        T,iA,iB = mesh.connectivity.direct_face(A,B,True)
        if T is not None:
            cnr = mesh.connectivity.face_to_first_corner(T)+3-iA-iB
            cw[e] += cot[cnr]/2
        T,iB,iA = mesh.connectivity.direct_face(B,A,True)
        if T is not None:
            cnr = mesh.connectivity.face_to_first_corner(T)+3-iA-iB
            cw[e] += cot[cnr]/2
    if persistent:
        return mesh.edges.register_array_as_attribute(name, cw)
    return cw 