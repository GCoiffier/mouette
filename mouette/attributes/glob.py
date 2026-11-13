import numpy as np

from ..mesh.datatypes import *
from ..geometry import Vec
from .attr_edges import edge_length
from .attr_faces import face_area
from .attr_cells import cell_volume

@allowed_mesh_types(SurfaceMesh)
def euler_characteristic(mesh : SurfaceMesh) -> int:
    """
    Computes the Euler characteristic of a surface mesh, given as V-E+F
    
    See https://en.wikipedia.org/wiki/Euler_characteristic

    Args:
        mesh (SurfaceMesh): the mesh

    Returns:
        (int): the Euler characteristic
    """
    
    v = len(mesh.vertices)
    e = len(mesh.edges)
    f = len(mesh.faces)
    return v-e+f


@forbidden_mesh_types(PointCloud)
def mean_edge_length(mesh : Mesh) -> float:
    """
    Estimation of mean edge length

    Parameters:
        mesh (Mesh): input mesh

    Returns:
        float: the computed mean edge length
    """
    if mesh.edges.has_attribute("length"):
        length = mesh.edges.get_attribute("length").as_array()
    else:
        length = edge_length(mesh,persistent=False)
    return np.mean(length)


@allowed_mesh_types(SurfaceMesh,VolumeMesh)
def mean_face_area(mesh : SurfaceMesh) -> float:
    """Estimation of mean face area

    Parameters:
        mesh (Mesh): input mesh

    Returns:
        float: the computed mean face area
    """
    if mesh.faces.has_attribute("area"):
        farea = mesh.faces.get_attribute("area").as_array()
    else:
        farea = face_area(mesh,persistent=False)
    return np.mean(farea)

@allowed_mesh_types(VolumeMesh)
def mean_cell_volume(mesh : VolumeMesh) -> float:
    """Estimation of mean cell volume

    Parameters:
        mesh (VolumeMesh): input mesh

    Returns:
        float: the computed mean cell volume
    """
    if mesh.cells.has_attribute("volume"):
        cvol = mesh.cells.get_attribute("volume").as_array()
    else:
        cvol = tetrahedra_volume(mesh, persistent=False)
    return np.mean(cvol)

@allowed_mesh_types(SurfaceMesh)
def total_area(mesh : SurfaceMesh) -> float:
    """Sum of face areas

    Parameters:
        mesh (Mesh): input mesh

    Returns:
        float: the computed mean face area
    """
    if mesh.faces.has_attribute("area"):
        farea = mesh.faces.get_attribute("area").as_array()
    else:
        farea = face_area(mesh,persistent=False)
    return np.sum(farea)

def barycenter(mesh : Mesh) -> Vec:
    """Barycenter of the vertices of the mesh

    Args:
        mesh (Mesh): input mesh

    Returns:
        Vec: coordinates of the barycenter
    """
    return np.mean(mesh.vertices._data)
