import numpy as np
import cmath
from numba import njit, prange
from typing import Union

from ..mesh.mesh_attributes import Attribute, SparseAttribute
from ..mesh.datatypes import *
from .. import geometry as geom
from .attribute_utils import _mean_per_element

@njit(cache=True, parallel=True)
def _triangle_area(V,F):
    n = F.shape[0]
    area = np.empty(n, dtype=float)
    for i in prange(n):
        area[i] = geom.triangle_area(V[F[i,0]], V[F[i,1]], V[F[i,2]])
    return area

@allowed_mesh_types(SurfaceMesh, VolumeMesh)
def face_area(mesh : Mesh, name="area", persistent:bool=True) -> Union[Attribute, np.ndarray]:
    """
    Computes the area of each triangular face.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "area".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Warning:
        The result of this function is not accurate if the considered face is not planar.

    Returns:
       Union[Attribute, np.ndarray]: array containing a float per face.
    """
    if mesh.faces._data.shape[1]==3:
        areas = _triangle_area(mesh.vertices._data, mesh.faces._data)
    else:
        VF = mesh.vertices._data[mesh.faces._data]
        areas = np.linalg.norm(np.cross(VF, np.roll(VF, -1, axis=1)).sum(axis=1), axis=1) / 2.
    if persistent:
        return mesh.faces.register_array_as_attribute(name, areas)
    return areas


@njit(cache=True, parallel=True)
def _face_normals(V,F):
    n = F.shape[0]
    normals = np.empty((n,3), dtype=float)
    for i in prange(n):
        nrm = np.cross(V[F[i,1]]-V[F[i,0]], V[F[i,2]]-V[F[i,0]])
        normals[i] = nrm / np.linalg.norm(nrm)
    return normals

@allowed_mesh_types(SurfaceMesh)
def face_normals(mesh : SurfaceMesh, name="normals", persistent:bool=True) -> Union[Attribute, np.ndarray]:
    """
    Computes the normal vector of each face.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "normals".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Warning:
        This function computes the normal by considering the first three vertices of a face. That means that if the face is not a triangle, the result is valid only if the face is planar.
    
    Returns:
        Union[Attribute, np.ndarray]: 3 floats per faces
    """    
    normals = _face_normals(mesh.vertices._data, mesh.faces._data)
    if persistent :
        return mesh.faces.register_array_as_attribute(name, normals)
    return normals


@allowed_mesh_types(SurfaceMesh, VolumeMesh)
def face_barycenter(mesh : SurfaceMesh, name="barycenter", persistent:bool = True) -> Union[Attribute, np.ndarray]:
    """
    Computes the barycenter point of each face.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "barycenter".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Returns:
        Union[Attribute, np.ndarray]: one 3D vector per face
    """
    bary = _mean_per_element(mesh.faces._data, mesh.vertices._data)
    if persistent:
        return mesh.faces.register_array_as_attribute(name, bary)
    return bary

@njit(cache=True, parallel=True)
def _face_circumcenter(V, F):
    n = F.shape[0]
    centers = np.empty((n, 3), dtype=float)
    for i in prange(n):
        centers[i] = geom.circumcenter(V[F[i,0]], V[F[i,1]], V[F[i,2]])
    return centers

@allowed_mesh_types(SurfaceMesh, VolumeMesh)
def face_circumcenter(mesh : SurfaceMesh, name="circumcenter", persistent:bool=True) -> Union[Attribute, np.ndarray]:
    """
    Computes the circumcenter point of each triangular face.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "barycenter".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Union[Attribute, np.ndarray]: one float per face

    Raises:
        Exception: fails if the mesh is not triangular
    """
    if mesh.faces._data.shape[1] != 3:
         raise Exception("circumcenters can be computed only for triangular faces")
    centers = _face_circumcenter(mesh.vertices._data, mesh.faces._data)
    if persistent :
        return mesh.faces.register_array_as_attribute(name, centers)
    return centers


@allowed_mesh_types(SurfaceMesh)
def face_near_border(mesh : SurfaceMesh, dist:int = 2, name = "near_border", persistent:bool = True, sparse:bool = True) -> SparseAttribute:
    """Returns a boolean attribute that is True if the face is at most at 'dist' neighbours from the boundary.
    Proceeds by region growing, starting from all faces touching the boundary and going inwards.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        dist (int, optional): Extend to which we flag the faces. All faces with a path of length < dist will be set to True. Defaults to 2.
        name (str, optional): Name given to the attribute. Defaults to "near_border".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        sparse (bool, optional): Is the attribute sparse (dict based) or not (array based). Defaults to True

    Returns:
        Attribute: one bool per face.
    """
    
    near = set()
    for e in mesh.boundary_edges:
        T1,T2 = mesh.connectivity.edge_to_faces(*mesh.edges[e])
        if T1 is None: near.add(T2)
        else: near.add(T1)
    for _ in range(dist-1):
        new_near = set()
        for T in mesh.id_faces:
            if T in near : continue
            for T2 in mesh.connectivity.face_to_faces(T):
                if T2 in near: new_near.add(T)
        near = near | new_near
    
    if persistent:
        near_attr = mesh.faces.create_attribute(name, bool, sparse=sparse)
    else:
        near_attr = SparseAttribute(bool, len(mesh.faces)) if sparse else Attribute(bool, len(mesh.faces))
    for f in near: near_attr[f] = True
    return near_attr


@njit(cache=True, parallel=True)
def _triangle_aspect_ratio(V,F):
    n = F.shape[0]
    aspect = np.empty((n, 3), dtype=float)
    for i in prange(n):
        aspect[i] = geom.aspect_ratio(V[F[i,0]], V[F[i,1]], V[F[i,2]])
    return aspect

@allowed_mesh_types(SurfaceMesh)
def triangle_aspect_ratio(mesh : SurfaceMesh, name : str="aspect_ratio", persistent : bool = True) -> Union[Attribute,np.ndarray]:
    """Computes the aspect ratio of every triangular faces. Sets aspect ratio to -1 for every other faces

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "aspect_ratio".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Raises:
        Exception: Fails if the mesh is not triangular

    Returns:
        Union[Attribute,np.ndarray]: One float per face.
    """
    if not mesh.is_triangular():
        raise Exception("Aspect ratios can only be computed for triangular faces.")
    ratio = _triangle_aspect_ratio(mesh.vertices._data, mesh.faces._data)
    if persistent :
        return mesh.faces.register_array_as_attribute(name, ratio)
    return ratio


@allowed_mesh_types(SurfaceMesh)
def parallel_transport_curvature(mesh : SurfaceMesh, PT:"SurfaceConnectionVertices", name : str="curvature", persistent : bool=True) -> Union[Attribute, np.ndarray]:
    """
    Compute the curvature of each face associated to a given parallel transport pT

    Args:
        mesh (SurfaceMesh): input mesh
        PT (SurfaceConnectionVertices): local bases and parallel transport for vertices.
        name (str, optional): Name given to the attribute.. Defaults to "curvature".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
    
    Returns:
        Attribute: One float per face.
    """
    if persistent :
        curv = mesh.faces.create_attribute(name, float)
    else:
        curv = np.empty(len(mesh.faces), dtype=float)
    for iF, (A,B,C) in enumerate(mesh.faces):
        v = 1+0j
        for a,b in [(A,B) ,(B,C), (C,A)]:
            v *= cmath.rect(1., PT.transport(b,a) - PT.transport(a,b) - np.pi)
        curv[iF] = cmath.phase(v)
    return curv