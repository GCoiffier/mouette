from ..mesh.mesh_attributes import SparseAttribute, SparseAttribute, Attribute
from ..mesh.datatypes import *
from .. import geometry as geom
import numpy as np
from numba import njit, prange
from typing import Union


@njit(cache=True, parallel=True)
def _corner_angles(V, F):
    n = F.shape[0]
    nf = F.shape[1]
    angles = np.empty(n*nf, dtype=float)
    for i in prange(n):
        for j in prange(nf):
            iPrev,iV,iNext = F[i, (j-1)%nf], F[i, j], F[i, (j+1)%nf]
            pPrev, pV, pNext = V[iPrev], V[iV], V[iNext]
            angles[i*nf+j] = geom.angle_3pts(pPrev, pV, pNext)
    return angles

@allowed_mesh_types(SurfaceMesh)
def corner_angles(mesh : SurfaceMesh, name = "angles", persistent:bool=True) -> Union[Attribute, np.ndarray]:
    """Attribute storing the angles of a face at a vertex

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "angles".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Union[Attribute, np.ndarray]: one float per face corner
    """
    angles = _corner_angles(mesh.vertices._data, mesh.faces._data)
    if persistent:
        return mesh.face_corners.register_array_as_attribute(name, angles)
    return angles



@allowed_mesh_types(SurfaceMesh)
def cotangent(mesh : SurfaceMesh, name = "cotan", persistent:bool = True) -> Union[Attribute, np.ndarray]:
    """Attribute storing the cotangents of each face corner angle.
    
    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "cotan".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Raises:
        Exception: fails if the mesh is not triangulated
    
    Returns:
       Union[Attribute, np.ndarray]: one float per face corner
    """
    if not mesh.is_triangular():
        raise Exception("Tried to compute cotangents on a non-triangulated mesh")

    if mesh.face_corners.has_attribute("angles"):
        angles = mesh.face_corners.get_attribute("angles").as_array()
    elif persistent:
        angles = corner_angles(mesh, persistent=True).as_array()
    else:
        angles = corner_angles(mesh, persistent=False)

    cot = -np.tan(angles + np.pi/2)
    if persistent:
        return mesh.face_corners.register_array_as_attribute(name, cot)
    return cot