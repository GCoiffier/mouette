from math import pi
import numpy as np
from numba import njit, prange
from typing import Union

from ..mesh.datatypes import *
from ..mesh.mesh_attributes import SparseAttribute, Attribute, SparseAttribute
from .attr_faces import face_normals
from .attr_corners import corner_angles
from .interpolate import interpolate_faces_to_vertices
from .. import geometry as geom
from ..geometry import Vec

@njit(cache=True)
def _degree(V,E):
    deg = np.zeros(V.shape[0], dtype=int)
    for i in range(E.shape[0]):
        deg[E[i,0]] += 1
        deg[E[i,1]] += 1
    return deg

@forbidden_mesh_types(PointCloud)
def degree(mesh : Mesh, name : str = "degree", persistent:bool = True) -> Attribute:
    """Computes the degree of each vertex, that is the number of vertex that are adjacent.

    Parameters:
        mesh (Mesh): the input mesh. Pointclouds are forbidden
        name (str, optional): _description_. Defaults to "degree".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Returns:
        Attribute(int)|np.ndarray:  degree of each vertices 
    """
    deg = _degree(mesh.vertices._data, mesh.edges._data)
    if persistent:
        return mesh.vertices.register_array_as_attribute(name, deg)
    else:
        return deg

@njit(cache=True)
def _angle_defects(C, angles, defects):
    for i in range(C.shape[0]):
        defects[C[i]] -= angles[i]
    return defects

@allowed_mesh_types(SurfaceMesh)
def angle_defects(mesh : SurfaceMesh, zero_border=False, name = "angleDefect", persistent=True) -> Union[Attribute,np.ndarray]:
    """Computes the angle defect at each vertex, defined as 2*pi minus the sum of angles around the vertex.

    Angle defect is an discrete approximation of the gaussian curvature.
    This function relies on the computation of cotangents on the mesh.
    
    Warning: 
        Only works for triangulated meshes.

    Parameters:
        mesh (SurfaceMesh): the mesh
        zero_border (bool): if set to true, ignores the defect on the boundary of the mesh. Defaults to False.
        name (str, optional): Name given to the attribute. Defaults to "angleDefect".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Returns:
        Attribute(float) on vertices
    """
    if mesh.face_corners.has_attribute("angles"):
        angles = mesh.face_corners.get_attribute("angles").as_array()
    elif persistent:
        angles = corner_angles(mesh, persistent=True).as_array()
    else:
        angles = corner_angles(mesh, persistent=False)

    defects = np.full(len(mesh.vertices), 2*pi)
    defects[mesh.boundary_vertices] = 0 if zero_border else pi
    defects = _angle_defects(mesh.face_corners._elem, angles, defects)

    if persistent:
        return mesh.vertices.register_array_as_attribute(name, defects)
    else:
        return defects


@allowed_mesh_types(SurfaceMesh)
def vertex_normals(mesh : SurfaceMesh, name="normals", persistent=True, interpolation="area", custom_fnormals : np.ndarray = None) -> Union[Attribute,np.ndarray]:
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
        custom_fnormals (np.ndarray, optional): custom values for the face normals to be interpolated. Defaults to None.

    Raises:
        Exception: if 'interpolation' is not one of {'uniform', 'area', 'angles'}

    Returns:
        Attribute(float, 3) on vertices
    """

    if interpolation not in {'uniform', 'area', 'angle'}:
        raise Exception(f"vertex_normals: interpolation argument '{interpolation}' not recognized.\nPossibilites are {'uniform', 'area', 'angle'}.")

    if custom_fnormals is not None:
        fnormals = np.asarray(custom_fnormals)
    elif mesh.faces.has_attribute("normals"):
        fnormals = mesh.faces.get_attribute("normals").as_array()
    elif persistent:
        fnormals = face_normals(mesh, persistent=True).as_array()
    else:
        fnormals = face_normals(mesh, persistent=False)
    
    normals = interpolate_faces_to_vertices(mesh, fnormals, "normals", weight=interpolation, persistent=False)
    normals /= np.linalg.norm(normals, axis=1).reshape((-1,1))
    if persistent:
        return mesh.vertices.register_array_as_attribute(name, normals)
    return normals

    
@allowed_mesh_types(SurfaceMesh)
def border_normals(mesh : SurfaceMesh, name="borderNormals", persistent: bool = True, dense:bool = False) -> Union[Attribute, SparseAttribute] :
    """Computes the normal direction of the boundary curve.

    Parameters:
        mesh (SurfaceMesh): the mesh
        name (str, optional): Name given to the attribute. Defaults to "borderNormals".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True

    Returns:
        Union[Attribute, SparseAttribute,np.ndarray]: The border normals on vertices (zero for non-boundary vertices)
    """
    if persistent:
        bnormals = mesh.vertices.create_attribute(name, float, 3, dense=dense)
    else:
        bnormals = np.zeros((len(mesh.vertices), 3)) if dense else SparseAttribute(float, 3)

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
        VT0 = mesh.vertices[mesh.faces[T0]]
        VT1 = mesh.vertices[mesh.faces[T1]]
        _,_,N0 = geom.face_basis(VT0[0,:], VT0[1,:], VT0[2,:])
        _,_,N1 = geom.face_basis(VT1[0,:], VT1[1,:], VT1[2,:])
        bnormals[v] = geom.cross(E0,N0) + geom.cross(E1,N1)
        bnormals[v] = Vec.normalized(bnormals[v])
    return bnormals