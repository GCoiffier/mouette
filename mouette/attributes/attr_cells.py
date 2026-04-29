import numpy as np

from ..mesh.mesh_attributes import Attribute, ArrayAttribute
from ..mesh.datatypes import *
from ..geometry.geometry import det_3x3

@allowed_mesh_types(VolumeMesh)
def cell_volume(mesh : VolumeMesh, name="volume", persistent=True , dense=True) -> Attribute:
    """
    Computes the volume of each cell.

    Parameters:
        mesh (VolumeMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "volume".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute: one float per cell
    """
    if persistent :
        volume = mesh.cells.create_attribute(name, float, dense=dense)
    else:
        volume = ArrayAttribute(float, len(mesh.cells)) if dense else Attribute(float)
    if not mesh.is_tetrahedral():
        raise NotImplementedError
    for ic,(A,B,C,D) in enumerate(mesh.cells):
        pA,pB,pC,pD = (mesh.vertices[_v] for _v in (A,B,C,D))
        volume[ic]= abs(det_3x3(pA-pD,pB-pD,pC-pD))/6
    return volume

@allowed_mesh_types(VolumeMesh)
def cell_barycenter(mesh : VolumeMesh, name="barycenter", persistent=True, dense=True)-> Attribute:
    """
    Computes the barycenter point of each cell.

    Parameters:
        mesh (VolumeMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "barycenter".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute: 3D vector per cell
    """
    if persistent :
        bary = mesh.cells.create_attribute(name, float, 3, dense=dense)
    else:
        bary = ArrayAttribute(float, len(mesh.cells), 3) if dense else Attribute(float, 3)
    for iC,C in enumerate(mesh.cells):
        bary[iC] = sum(mesh.vertices[u] for u in C) / len(C)
    return bary

@allowed_mesh_types(VolumeMesh)
def cell_faces_on_boundary(mesh : VolumeMesh, name="boundary", persistent=True, dense=False)-> Attribute:
    """
    Integer flag on cells. For each cell, computes the number of its faces that lay on the boundary

    Parameters:
        mesh (VolumeMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "boundary".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to False
    
    Returns:
        Attribute: bool per cell
    """
    if persistent:
        bndf = mesh.cells.create_attribute(name, int, dense=dense)
    else:
        bndf = ArrayAttribute(int, len(mesh.cells)) if dense else Attribute(int)

    for iF in mesh.id_faces:
        if len(mesh.connectivity.face_to_cells(iF))==1 : 
            iC = mesh.connectivity.face_to_cells(iF)[0]
            bndf[iC] += 1
    return bndf