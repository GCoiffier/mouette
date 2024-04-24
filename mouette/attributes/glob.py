from ..mesh.datatypes import *
from ..geometry import Vec
from .misc_faces import face_area
from .misc_cells import cell_volume

@allowed_mesh_types(SurfaceMesh)
def euler_characteristic(mesh : SurfaceMesh) -> int:
    """
    Computes the Euler characteristic of a surface mesh, given as V-E+F
    
    See https://en.wikipedia.org/wiki/Euler_characteristic

    Parameters:
        mesh (SurfaceMesh): the mesh

    Returns:
        (int): the Euler characteristic
    """
    
    v = len(mesh.vertices)
    e = len(mesh.edges)
    f = len(mesh.faces)
    return v-e+f

# @allowed_mesh_types(SurfaceMesh)
# def genus(mesh : SurfaceMesh) -> int:
#     """Computes the genus g of a mesh.
#     The genus is the number of "holes" a surface contains. It is linked to the euler characteristic X by
#     X = 2 - 2g

#     Parameters:
#         mesh (Mesh): the mesh

#     Returns:
#         (int): the Euler characteristic
#     """
#     X = euler_characteristic(mesh)
#     return 1 - X//2

@forbidden_mesh_types(PointCloud)
def mean_edge_length(mesh : Mesh, n : int = None) -> float:
    """
    Estimation of mean edge length

    Parameters:
        mesh (Mesh): input mesh
        n (int, optional): 
            Early stopping for number of edges to consider in mean computation. 
            If set to None, considers all the edges.
            Defaults to None.

    Returns:
        float: the computed mean edge length
    """
    l = 0
    if n is None: n = len(mesh.edges)
    for k in range(min(n, len(mesh.edges))):
        a,b = (Vec(mesh.vertices[u]) for u in mesh.edges[k])
        l += (b-a).norm()
    return l/n

@allowed_mesh_types(SurfaceMesh,VolumeMesh)
def mean_face_area(mesh : SurfaceMesh, n : int = None) -> float:
    """Estimation of mean face area

    Parameters:
        mesh (Mesh): input mesh
        n (int, optional): 
            Early stopping for number of faces to consider in mean computation. 
            If set to None, considers all the faces.
            Defaults to None.

    Returns:
        float: the computed mean face area
    """
    if mesh.faces.has_attribute("area"):
        farea = mesh.faces.get_attribute("area")
    else:
        farea = face_area(mesh)
    if n is None: n = len(mesh.faces)
    res = 0
    for k in range(min(n, len(mesh.faces))):
        res += farea[k]
    return res/n

@allowed_mesh_types(VolumeMesh)
def mean_cell_volume(mesh : VolumeMesh, n : int = None) -> float:
    """Estimation of mean cell volume

    Parameters:
        mesh (VolumeMesh): input mesh
        n (int, optional): 
            Early stopping for number of cells to consider in mean computation. 
            If set to None, considers all the cells.
            Defaults to None.

    Returns:
        float: the computed mean cell volume
    """
    if mesh.cells.has_attribute("volume"):
        cvol = mesh.cells.get_attribute("volume")
    else:
        cvol = cell_volume(mesh)
    if n is None: n = len(mesh.cells)
    res = 0
    for k in range(min(n, len(mesh.cells))):
        res += cvol[k]
    return res/n

@allowed_mesh_types(SurfaceMesh)
def total_area(mesh : SurfaceMesh) -> float:
    """Sum of face areas

    Parameters:
        mesh (Mesh): input mesh

    Returns:
        float: the computed mean face area
    """
    if mesh.faces.has_attribute("area"):
        farea = mesh.faces.get_attribute("area")
    else:
        farea = face_area(mesh, persistent=False)
    return sum([farea[iF] for iF in mesh.id_faces])

def barycenter(mesh : Mesh) -> Vec:
    """Barycenter of the vertices of the mesh

    Args:
        mesh (Mesh): input mesh

    Returns:
        Vec: coordinates of the barycenter
    """
    return sum(mesh.vertices)/len(mesh.vertices)