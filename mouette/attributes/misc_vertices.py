import numpy as np
from math import pi, atan2

from ..mesh.datatypes import *
from ..mesh.mesh_attributes import Attribute, ArrayAttribute, Attribute
from .misc_faces import face_normals
from .misc_corners import corner_angles
from .interpolate import interpolate_faces_to_vertices
from .. import geometry as geom
from ..geometry import Vec

@forbidden_mesh_types(PointCloud)
def degree(mesh : Mesh, name : str = "degree", persistent:bool = True, dense:bool = True) -> Attribute :
    """Computes the degree of each vertex, that is the number of vertex that are adjacent.

    Parameters:
        mesh (Mesh): the input mesh. Pointclouds are forbidden
        name (str, optional): _description_. Defaults to "degree".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute(int) on vertices 
    """
    if persistent:
        deg = mesh.vertices.create_attribute(name, int, dense=dense)
    else:
        deg = ArrayAttribute(int, len(mesh.vertices)) if dense else Attribute(int)
    for (a,b) in mesh.edges:
        deg[a] = deg[a] + 1
        deg[b] = deg[b] + 1
    return deg

@allowed_mesh_types(SurfaceMesh)
def angle_defects(mesh : SurfaceMesh, zero_border=False, name = "angleDefect", persistent=True, dense=True) -> Attribute :
    """Computes the angle defect at each vertex, defined as 2*pi minus the sum of angles around the vertex.

    Angle defect is an discrete approximation of the gaussian curvature.
    This function relies on the computation of cotangents on the mesh.
    WARNING: only works for triangulated meshes

    Parameters:
        mesh (SurfaceMesh): the mesh
        zero_border (bool): if set to true, ignores the defect on the boundary of the mesh. Defaults to False.
        name (str, optional): Name given to the attribute. Defaults to "angleDefect".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute(float) on vertices

    Raises:
        Exception: fails if the mesh is not triangulated (ie if a face is not a triangle)
    """
    if not mesh.is_triangular():
        raise Exception("Tried to compute angle defects on a non-triangulated mesh")

    if persistent:
        defects = mesh.vertices.create_attribute(name, float, dense=dense, default_value=2*pi)
    else:
        defects = ArrayAttribute(float, len(mesh.vertices), default_value=2*pi) if dense else Attribute(float, default_value=2*pi)
    
    for i in mesh.boundary_vertices:
        defects[i] = 0 if zero_border else pi

    if mesh.face_corners.has_attribute("angles"):
        ang = mesh.face_corners.get_attribute("angles")
    else:
        ang = corner_angles(mesh, persistent=persistent)

    for C,V in enumerate(mesh.face_corners):
        if mesh.is_vertex_on_border(V) and zero_border: continue
        defects[V] -= ang[C]
    return defects

@allowed_mesh_types(SurfaceMesh)
def vertex_normals(mesh : SurfaceMesh, name="normals", persistent=True, interpolation="area", dense=True, custom_fnormals : Attribute = None) -> Attribute:
    """Computes normal directions as 3d vectors for each vertex.
    Normal at a vertex is a weighted sum of normals of adjacent faces. This function essentially interpolates the face normals.

    Parameters:
        mesh (SurfaceMesh): the input mesh
        name (str, optional): Name given to the attribute. Defaults to "normals".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        interpolation (str, optional): Interpolation weighting mode.
            'uniform' : computes the mean for adjacent facet
            'area' : the mean is weighted by facet area. Default value.
            'angle' : the mean is weighted by internal angle at the vertex 
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True
        custom_fnormals (Attribute, optional): custom values for the face normals to be interpolated. Defaults to None.

    Raises:
        Exception: if 'interpolation' is not one of {'uniform', 'area', 'angles'}

    Returns:
        Attribute(float, 3) on vertices
    """

    if interpolation not in {'uniform', 'area', 'angle'}:
        raise Exception(f"vertex_normals: interpolation argument '{interpolation}' not recognized.\nPossibilites are {'uniform', 'area', 'angle'}.")

    if custom_fnormals is not None:
        fnormals = custom_fnormals
    elif mesh.faces.has_attribute("normals"):
        fnormals = mesh.faces.get_attribute("normals")
    else:
        fnormals = face_normals(mesh, persistent=persistent)
    
    if persistent:
        normals = mesh.vertices.create_attribute(name, float, 3, dense=dense)
    else:
        normals = ArrayAttribute(float, len(mesh.vertices), 3) if dense else Attribute(float)

    normals = interpolate_faces_to_vertices(mesh, fnormals, normals, weight=interpolation)
    for v in mesh.id_vertices:
        normals[v] = Vec.normalized(normals[v])
    return normals
    
@allowed_mesh_types(SurfaceMesh)
def border_normals(mesh : SurfaceMesh, name="borderNormals", persistent: bool = True, dense:bool = False) -> Attribute :
    """Computes the normal direction of the boundary curve.

    Parameters:
        mesh (SurfaceMesh): the mesh
        name (str, optional): Name given to the attribute. Defaults to "borderNormals".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Attribute(float,3) on vertices
    """
    if persistent:
        bnormals = mesh.vertices.create_attribute(name, float, 3, dense=dense)
    else:
        bnormals = ArrayAttribute(float, len(mesh.vertices), 3) if dense else Attribute(float, 3)

    for v in mesh.boundary_vertices:
        v0 = mesh.connectivity.vertex_to_vertices(v)[0]
        v1 = mesh.connectivity.vertex_to_vertices(v)[-1]
        P,P0,P1 = (mesh.vertices[u] for u in (v,v0,v1))
        E0 = P - P0
        E1 = P1 - P
        T0 = mesh.connectivity.direct_face(v0,v)
        if T0 is None: T0 = mesh.connectivity.direct_face(v,v0)
        T1 = mesh.connectivity.direct_face(v,v1)
        if T1 is None: T1 = mesh.connectivity.direct_face(v1,v)
        _,_,N0 = geom.face_basis(*mesh.pt_of_face(T0))
        _,_,N1 = geom.face_basis(*mesh.pt_of_face(T1))
        bnormals[v] = geom.cross(E0,N0) + geom.cross(E1,N1)
        bnormals[v] = Vec.normalized(bnormals[v])
    return bnormals