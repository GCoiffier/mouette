import numpy as np
import scipy.sparse as sp
from ..mesh.datatypes import *
from ..attributes import face_area, cell_volume, cotangent

@allowed_mesh_types(SurfaceMesh)
def area_weight_matrix(mesh : SurfaceMesh, inverse:bool = False, sqrt:bool = False, format: str = "csc") -> sp.csc_matrix:
    """
    Returns the diagonal matrix A of area weights on vertices.
    
    Args:
        mesh (SurfaceMesh): input mesh
        inverse (bool, optional): whether to return A or A^-1. Defaults to False.
        sqrt (bool, optional): whether to return A or A^{1/2}. Can be combined with `inverse` to return A^{-1/2}. Defaults to False.
        format (str, optional): one of the sparse matrix format of scipy (csc, csr, coo, lil, ...). Defaults to csc.

    Returns:
        sp.csc_matrix: diagonal matrix of vertex areas
    """
    A = np.zeros(len(mesh.vertices))
    if mesh.faces.has_attribute("area"):
        area = mesh.faces.get_attribute("area")
    else:
        area = face_area(mesh)
    for iT,T in enumerate(mesh.faces):
        for u in T:
            A[u] += area[iT]
    if sqrt: A = np.sqrt(A)
    if inverse : A = 1/A # /!\ perform inverse after sqrt
    return sp.diags(A, format=format)


@allowed_mesh_types(SurfaceMesh)
def cotan_weight_matrix(mesh : SurfaceMesh, inverse:bool = False, sqrt:bool = False, format: str = "csc") -> sp.csc_matrix:
    """
    Returns the diagonal matrix A of cotan weights on vertices. The weight of vertex v is defined as the sum of cotangents around v.
    
    Args:
        mesh (SurfaceMesh): input mesh
        inverse (bool, optional): whether to return A or A^-1. Defaults to False.
        sqrt (bool, optional): whether to return A or A^{1/2}. Can be combined with `inverse` to return A^{-1/2}. Defaults to False.
        format (str, optional): one of the sparse matrix format of scipy (csc, csr, coo, lil, ...). Defaults to csc.

    Returns:
        sp.csc_matrix: diagonal matrix of vertex areas
    """
    A = np.zeros(len(mesh.vertices))
    if mesh.face_corners.has_attribute("cotan"):
        cotan = mesh.face_corners.get_attribute("cotan")
    else:
        cotan = cotangent(mesh)
    for c,v in enumerate(mesh.face_corners):
        A[v] += cotan[c]
    if sqrt: A = np.sqrt(A)
    if inverse : A = 1/A # /!\ perform inverse after sqrt
    return sp.diags(A, format=format)


@allowed_mesh_types(SurfaceMesh)
def area_weight_matrix_faces(mesh : SurfaceMesh, inverse : bool=False, format: str = "csc") -> sp.csc_matrix:
    """
    Returns the diagonal matrix A of area weights on faces
    Laplace-beltrami operator for a 2D manifold is (A^-1)L where A is the area weight and L is the cotan matrix

    Args:
        mesh (SurfaceMesh): the input mesh
        inverse (bool, optional): whether to return A or A^-1. Defaults to False.

    Returns:
        sp.csc_matrix
    """
    if mesh.faces.has_attribute("area"):
        area = mesh.faces.get_attribute("area")
    else:
        area = face_area(mesh)
    area = area.as_array(len(mesh.faces))
    if inverse:
        area = 1/area
    return sp.diags(area, format=format)


@allowed_mesh_types(SurfaceMesh)
def area_weight_matrix_edges(mesh : SurfaceMesh, inverse : bool=False) -> sp.csc_matrix:
    """
    Returns the diagonal matrix A of area weights on edges

    Args:
        mesh (SurfaceMesh): the input mesh
        inverse (bool, optional): whether to return A or A^-1. Defaults to False.

    Returns:
        sp.csc_matrix
    """
    if mesh.faces.has_attribute("area"):
        area = mesh.faces.get_attribute("area")
    else:
        area = face_area(mesh)

    area_edges = np.zeros(len(mesh.edges))
    for e,(A,B) in enumerate(mesh.edges):
        for T in mesh.connectivity.edge_to_faces(A,B):
            if T is None: continue
            area_edges[e] += area[T]/3
    if inverse:
        area_edges = 1/area_edges
    return sp.diags(area_edges, format="csc")


@allowed_mesh_types(VolumeMesh)
def volume_weight_matrix(mesh: VolumeMesh, inverse: bool = False, sqrt: bool = False, format: str = "csc") -> sp.csc_matrix:
    """
    Mass diagonal matrix for volume Laplacian.
    Args:
        mesh (VolumeMesh): input mesh
        inverse (bool, optional): whether to return A or A^-1. Defaults to False.
        sqrt (bool, optional): whether to return A or A^{1/2}. Can be combined with `inverse` to return A^{-1/2}. Defaults to False.
        format (str, optional): one of the sparse matrix format of scipy (csc, csr, coo, lil, ...). Defaults to dia.

    Returns:
        sp.csc_matrix: diagonal matrix of vertices area
    """
    if mesh.cells.has_attribute("volume"):
        volume = mesh.cells.get_attribute("volume")
    else:
        volume = cell_volume(mesh)
    V = np.zeros(len(mesh.vertices))
    for iC,C in enumerate(mesh.cells):
        for u in C:
            V[u] += volume[iC] 
    if sqrt: V = np.sqrt(V)
    if inverse: V = 1/V
    return sp.diags(V, format=format)


@allowed_mesh_types(VolumeMesh)
def volume_weight_matrix_cells(mesh: VolumeMesh, inverse: bool = False, sqrt: bool = False, format: str = "csc") -> sp.csc_matrix:
    """
    Mass diagonal matrix for volume Laplacian on cells
    Args:
        mesh (VolumeMesh): input mesh
        inverse (bool, optional): whether to return A or A^-1. Defaults to False.
        sqrt (bool, optional): whether to return A or A^{1/2}. Can be combined with `inverse` to return A^{-1/2}. Defaults to False.
        format (str, optional): one of the sparse matrix format of scipy (csc, csr, coo, lil, ...). Defaults to dia.

    Returns:
        sp.csc_matrix: diagonal matrix of vertices area
    """
    if mesh.cells.has_attribute("volume"):
        volume = mesh.cells.get_attribute("volume")
    else:
        volume = cell_volume(mesh)
    V = volume.as_array(len(mesh.cells))
    if sqrt: V = np.sqrt(V)
    if inverse: V = 1/V
    return sp.diags(V, format=format)