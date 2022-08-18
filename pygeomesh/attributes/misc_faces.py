import numpy as np

from ..mesh.mesh_attributes import ArrayAttribute, Attribute
from ..mesh.datatypes import *
from .. import geometry as geom
from ..geometry import Vec

@allowed_mesh_types(SurfaceMesh, VolumeMesh)
def face_area(mesh : Mesh, name="area", persistent:bool=True, dense:bool=True):
    """
    Computes the barycenter point of each face.

    Args:
        mesh (SurfaceMesh)
        name (str, optional): Name given to the attribute. Defaults to "area".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute[float] on faces
    """
    if persistent:
        area = mesh.faces.create_attribute(name, float, dense=dense)
    else:
        area = ArrayAttribute(float, len(mesh.faces)) if dense else Attribute(float)
    for T in mesh.id_faces:
        pts = [mesh.vertices[u] for u in mesh.faces[T]]
        npt = len(pts)
        if npt==3:
            area[T] = geom.triangle_area(*pts)
        elif npt==4:
            area[T] = geom.quad_area(*pts)
        else:
            bary = sum(pts)/npt
            for i in range(npt):
                A = pts[i]
                B = pts[(i+1)%npt]
                area[T] += geom.triangle_area(A,B,bary)            
    return area

@allowed_mesh_types(SurfaceMesh)
def face_normals(mesh : SurfaceMesh, name="normals", persistent:bool=True, dense:bool=True):
    """
    Computes the barycenter point of each face.

    Args:
        mesh (SurfaceMesh)
        name (str, optional): Name given to the attribute. Defaults to "normals".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute[float, 3] on faces
    """
    if persistent :
        normals = mesh.faces.create_attribute(name, float, 3)
    else:
        normals = ArrayAttribute(float, len(mesh.faces), 3) if dense else Attribute(float)
    for iT,T in enumerate(mesh.faces):
        pA,pB,pC = (mesh.vertices[u] for u in T[:3])
        normals[iT] = Vec.normalized(geom.cross(pB-pA, pC-pA))
    return normals

@allowed_mesh_types(SurfaceMesh, VolumeMesh)
def face_barycenter(mesh : SurfaceMesh, name="barycenter", persistent:bool = True, dense:bool = True):
    """
    Computes the barycenter point of each face.

    Args:
        mesh (SurfaceMesh)
        name (str, optional): Name given to the attribute. Defaults to "barycenter".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute[float, 3] on faces
    """
    if persistent :
        bary = mesh.faces.create_attribute(name, float, 3)
    else:
        bary = ArrayAttribute(float, len(mesh.faces), 3) if dense else Attribute(float, 3)
    for iT,T in enumerate(mesh.faces):
        bary[iT] = sum(mesh.vertices[u] for u in T) / len(T)
    return bary

@allowed_mesh_types(SurfaceMesh, VolumeMesh)
def face_circumcenter(mesh : SurfaceMesh, name="circumcenter", persistent:bool=True, dense:bool=True):
    """
    Computes the circumcenter point of each face.

    Args:
        mesh (SurfaceMesh)
        name (str, optional): Name given to the attribute. Defaults to "barycenter".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute[float, 3] on faces
    """
    if persistent :
        circum = mesh.faces.create_attribute(name, float, 3)
    else:
        circum = ArrayAttribute(float, len(mesh.faces), 3) if dense else Attribute(float, 3)
    for iF,F in enumerate(mesh.faces):
        if len(F)!=3:
            raise Exception("circumcenters can be computed only for triangular faces. Received face n°{} containing {}!=3 vertices".format(iF,len(F)))
        circum[iF] = geom.circumcenter(*(mesh.vertices[u] for u in F))

@allowed_mesh_types(SurfaceMesh)
def faces_near_border(mesh : SurfaceMesh, dist:int = 2, name = "near_border", persistent:bool = True, dense:bool = False):
    """Returns the faces that are at most at 'dist' neighbours from the boundary.
    Proceeds by region growing, starting from all faces touching the boundary and going inwards.

    Args:
        mesh (SurfaceMesh): input mesh
        dist (int, optional): Extend to which we flag the faces. All faces with a path of length < dist to the boundary will be returned. Defaults to 2.
        name (str, optional): Name given to the attribute. Defaults to "barycenter".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to False

    Returns:
        Attribute[bool] on faces. If not persistent, returns a set.
    """
    
    near = set()
    for e in mesh.boundary_edges:
        T1,T2 = mesh.half_edges.edge_to_triangles(*mesh.edges[e])
        if T1 is None: near.add(T2)
        else: near.add(T1)
    for _ in range(dist-1):
        new_near = set()
        for T in mesh.id_faces:
            if T in near : continue
            for T2 in mesh.connectivity.face_to_face(T):
                if T2 in near: new_near.add(T)
        near = near | new_near
    
    if persistent:
        near_attr = mesh.faces.create_attribute(name, bool, dense=dense)
    else:
        near_attr = ArrayAttribute(bool, len(mesh.faces)) if dense else Attribute(bool)
    for f in near: near_attr[f] = True
    return near_attr
