import numpy as np
import cmath

from ..mesh.mesh_attributes import ArrayAttribute, Attribute
from ..mesh.datatypes import *
from .. import geometry as geom
from ..geometry import Vec

@allowed_mesh_types(SurfaceMesh, VolumeMesh)
def face_area(mesh : Mesh, name="area", persistent:bool=True, dense:bool=True) -> Attribute:
    """
    Computes the barycenter point of each face.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "area".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute: a float per faces
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
def face_normals(mesh : SurfaceMesh, name="normals", persistent:bool=True, dense:bool=True) -> Attribute:
    """
    Computes the barycenter point of each face.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "normals".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute: 3 floats per faces
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
def face_barycenter(mesh : SurfaceMesh, name="barycenter", persistent:bool = True, dense:bool = True) -> Attribute:
    """
    Computes the barycenter point of each face.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "barycenter".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute: one 3D vector per face
    """
    if persistent :
        bary = mesh.faces.create_attribute(name, float, 3)
    else:
        bary = ArrayAttribute(float, len(mesh.faces), 3) if dense else Attribute(float, 3)
    for iT,T in enumerate(mesh.faces):
        bary[iT] = sum(mesh.vertices[u] for u in T) / len(T)
    return bary

@allowed_mesh_types(SurfaceMesh, VolumeMesh)
def face_circumcenter(mesh : SurfaceMesh, name="circumcenter", persistent:bool=True, dense:bool=True) -> Attribute:
    """
    Computes the circumcenter point of each triangular face.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "barycenter".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute: one float per face

    Raises:
        Exception: fails if a face of the mesh is not a triangle
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
def faces_near_border(mesh : SurfaceMesh, dist:int = 2, name = "near_border", persistent:bool = True, dense:bool = False) -> Attribute:
    """Returns the faces that are at most at 'dist' neighbours from the boundary.
    Proceeds by region growing, starting from all faces touching the boundary and going inwards.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        dist (int, optional): Extend to which we flag the faces. All faces with a path of length < dist to the boundary will be returned. Defaults to 2.
        name (str, optional): Name given to the attribute. Defaults to "barycenter".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to False

    Returns:
        Attribute: one bool per faces. If not persistent, returns a set.
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
        near_attr = mesh.faces.create_attribute(name, bool, dense=dense)
    else:
        near_attr = ArrayAttribute(bool, len(mesh.faces)) if dense else Attribute(bool)
    for f in near: near_attr[f] = True
    return near_attr

@allowed_mesh_types(SurfaceMesh)
def triangle_aspect_ratio(mesh : SurfaceMesh, name : str="aspect_ratio", persistent : bool = True, dense : bool =True) -> Attribute:
    """Computes the aspect ratio of every triangular faces. Sets aspect ratio to -1 for every other faces

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "aspect_ratio".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to False.

    Returns:
        Attribute: One float per face.
    """
    if persistent :
        ratio = mesh.faces.create_attribute(name, float, 3)
    else:
        ratio = ArrayAttribute(float, len(mesh.faces), 3) if dense else Attribute(float, 3)
    for iF,F in enumerate(mesh.faces):
        if len(F)!=3: ratio[iF] = -1 
        ratio[iF] = geom.aspect_ratio(*(mesh.vertices[u] for u in F))

@allowed_mesh_types(SurfaceMesh)
def parallel_transport_curvature(mesh : SurfaceMesh, PT:"SurfaceConnectionFaces", name : str="curvature", persistent : bool=True, dense : bool = True):
    """
    Compute the curvature of each face associated to a given parallel transport pT

    Args:
        mesh (SurfaceMesh): _description_
        PT (dict): parallel transport. Dictionnary of keys (A,B) -> direction (angle) of edge (A,B) in local basis of A
        name (str, optional): Name given to the attribute.. Defaults to "curvature".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to False.
    
    Returns:
        Attribute: One float per face.
    """
    if persistent :
        curv = mesh.faces.create_attribute(name, float)
    else:
        curv = ArrayAttribute(float, len(mesh.faces)) if dense else Attribute(float)
    for iF, (A,B,C) in enumerate(mesh.faces):
        v = 1+0j
        for a,b in [(A,B) ,(B,C), (C,A)]:
            v *= cmath.rect(1., PT.transport(b,a) - PT.transport(a,b) - np.pi)
        curv[iF] = cmath.phase(v)
    return curv