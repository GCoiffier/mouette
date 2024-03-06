from ..mesh.mesh_attributes import Attribute, Attribute, ArrayAttribute
from ..mesh.datatypes import *
from .. import geometry as geom
from ..geometry import Vec
import numpy as np
import math

@allowed_mesh_types(SurfaceMesh)
def corner_angles(mesh : SurfaceMesh, name = "angles", persistent:bool=True, dense:bool = True) -> Attribute:
    """Attribute storing the angles of a face at a vertex

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "angles".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute: one float per face corner
    """
    if persistent:
        angles = mesh.face_corners.create_attribute(name, float, dense=True)
    else:
        angles = ArrayAttribute(float, len(mesh.face_corners)) if dense else Attribute(float)
    c = 0
    for face in mesh.faces:
        n = len(face)
        for i in range(n):
            iPrev,iV,iNext = face[(i-1)%n], face[i], face[(i+1)%n]
            pPrev, pV, pNext = mesh.vertices[iPrev], mesh.vertices[iV], mesh.vertices[iNext]
            angles[c] = geom.angle_3pts(pPrev, pV, pNext)
            c += 1
    return angles

@allowed_mesh_types(SurfaceMesh)
def cotangent(mesh : SurfaceMesh, name = "cotan", persistent=True, dense=True) -> Attribute:
    """Attribute storing the cotangents of each face at each vertex.
    WARNING: only works if the mesh is triangulated (ie every faces are triangles)

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "cotan".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute(: one float per face corner

    Raises:
        Exception: fails if the mesh is not triangulated
    """

    if not mesh.is_triangular():
        raise Exception("Tried to compute cotangents on a non-triangulated mesh")

    if persistent:
        if mesh.face_corners.has_attribute(name):
            cot = mesh.face_corners.get_attribute(name)
        else:
            cot = mesh.face_corners.create_attribute(name, float, dense=True)
    else:
        cot = ArrayAttribute(float, len(mesh.face_corners)) if dense else Attribute(float)

    if mesh.face_corners.has_attribute("angles"):
        # Compute cotan from angles
        angles = mesh.face_corners.get_attribute("angles")
        for c in mesh.id_corners:
            cot[c] = -np.tan(angles[c] + np.pi/2)
    else:
        for i, (iA,iB,iC) in enumerate(mesh.faces):
            pA, pB, pC = (mesh.vertices[_i] for _i in (iA,iB,iC))
            cot[3*i] = geom.cotan(pC, pA, pB)
            cot[3*i+1] = geom.cotan(pA, pB, pC)
            cot[3*i+2] = geom.cotan(pB, pC, pA)
    return cot