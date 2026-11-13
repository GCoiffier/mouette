import numpy as np
from numba import njit, prange
from typing import Union

from ..mesh.mesh_attributes import SparseAttribute, Attribute
from ..mesh.datatypes import *
from ..geometry.geometry import det_3x3
from .attribute_utils import _mean_per_element

@njit(cache=True, parallel=True)
def _tetrahedra_volume(V,tets):
    n = tets.shape[0]
    volume = np.empty(n, dtype=float)
    for i in prange(n):
        A,B,C,D = V[tets[i,0]], V[tets[i,1]], V[tets[i,2]] ,V[tets[i,3]]
        volume[i] = abs(det_3x3(A-D, B-D, C-D)/6)
    return volume


@njit(cache=True, parallel=True)
def _hexahedra_volume(V, hexs, faces):
    n = hexs.shape[0]
    volume = np.empty(n, dtype=float)
    for i in prange(n):
        bary = np.zeros(3)
        for k in range(8): bary += V[hexs[i,k]]
        bary /= 8
        vol = 0.
        for f in range(6):
            for j in range(4):
                P1 = V[hexs[i, faces[f, j]]]
                P2 = V[hexs[i, faces[f, (j+1)%4]]]
                P3 = V[hexs[i, faces[f, (j+2)%4]]]
                vol += det_3x3(P1-bary, P2-bary, P3-bary)/12
        volume[i] = abs(vol)
    return volume

@allowed_mesh_types(VolumeMesh)
def cell_volume(mesh : VolumeMesh, name="volume", persistent=True) -> Union[Attribute, np.ndarray]: 
    """Computes the volume of each cell.

    Parameters:
        mesh (VolumeMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "volume".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Returns:
        Union[Attribute, np.ndarray]: one float per cell
    """
    if mesh.cells._data.shape[1] == 4:
        volumes = _tetrahedra_volume(mesh.vertices._data, mesh.cells._data)
    elif mesh.cells._data.shape[1] == 8:
        # Outward-oriented quad faces of an hexahedron (bottom face 0-1-2-3, top face 4-5-6-7)
        HEX_FACES = np.array([[0,3,2,1], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [4,5,6,7]], dtype=np.int64)
        volumes = _hexahedra_volume(mesh.vertices._data, mesh.cells._data, HEX_FACES)
    else:
        raise Exception("Unsupported cell type in cell_volume")
    if persistent :
        return mesh.cells.register_array_as_attribute(name, volumes)
    return volumes


@allowed_mesh_types(VolumeMesh)
def cell_barycenter(mesh : VolumeMesh, name="barycenter", persistent=True)-> Union[Attribute, np.ndarray]:
    """
    Computes the barycenter point of each cell.

    Parameters:
        mesh (VolumeMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "barycenter".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Returns:
       Union[Attribute, np.ndarray]: 3D vector per cell
    """
    bary = _mean_per_element(mesh.cells._data, mesh.vertices._data)
    if persistent :
        return mesh.cells.register_array_as_attribute(name, bary)
    return bary


@allowed_mesh_types(VolumeMesh)
def cell_faces_on_boundary(mesh : VolumeMesh, name="boundary", persistent=True, sparse=True)-> SparseAttribute:
    """
    Integer flag on cells. For each cell, computes the number of its faces that lay on the boundary

    Parameters:
        mesh (VolumeMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "boundary".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        sparse (bool, optional): Is the attribute sparse (dict based) or not (array based). Defaults to True.
    
    Returns:
        Attribute: bool per cell
    """
    if persistent:
        bndf = mesh.cells.create_attribute(name, int, sparse=sparse)
    else:
        bndf = SparseAttribute(int, len(mesh.cells)) if sparse else Attribute(int, len(mesh.cells))

    for iF in mesh.id_faces:
        if len(mesh.connectivity.face_to_cells(iF))==1 : 
            iC = mesh.connectivity.face_to_cells(iF)[0]
            bndf[iC] += 1
    return bndf