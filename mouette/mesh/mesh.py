from .datatypes.raw import RawMeshData
from .data_container import DataContainer, CornerDataContainer
from .datatypes import Mesh, PointCloud, PolyLine, SurfaceMesh, VolumeMesh
from .io.io import read_by_extension, write_by_extension
from typing import Iterable
import numpy as np

##### Mesh loading #####
def load(filename: str, dim: int = None, raw: bool = False) -> Mesh:
    """
    Loads a Mesh object from a file on the disk.

    Supported file formats are:
    
    - wavefront (.obj)  
    - medit (.mesh)  
    - geogram (.geogram_ascii)
    - ply  
    - stl  
    - vtk
    - msh
    - off  
    - tet  
    - xyz

    Parameters:
        filename (str):  the path of the file
        dim (int, optional): 
            Override for the dimensionality of the resulting mesh.
            Depending on this value, returns a PointCloud, LineMesh, SurfaceMesh or VolumeMesh. 
            If not specified, computed from the data read in the file.
        raw (bool, optional): Returns the RawMeshData object instead of a fully prepared Mesh object. Defaults to False

    Returns:
        Mesh:  A new mesh object (PointCloud, LineMesh, SurfaceMesh or VolumeMesh)
    """
    rawdata : RawMeshData = read_by_extension(filename)
    if raw: return rawdata
    return rawdata.instantiate()


def from_arrays(
        V: np.ndarray, 
        E: np.ndarray = None, 
        F: np.ndarray = None, 
        C: np.ndarray = None
    ) -> Mesh:
    """Creates a mesh object from numpy arrays.

    Args:
        V (np.ndarray): vertex coordinates. SHould have shape |V|*3 or |V|*2. If shape is |V|*2, the array is padded so that z=0.
        E (np.ndarray, optional): Edge indices (shape |E|*2). Defaults to None.
        F (np.ndarray, optional): Face indices (shape |F|*n for n-regular faces). Defaults to None.
        C (np.ndarray, optional): Cell indices (shape |C|*n for n-regular cells). Defaults to None.

    Note:
        The function also accepts list of lists or any 2D iterable as input, though they will first be converted to numpy arrays.

    Returns:
        Mesh: a mesh object (PointCloud to VolumeMesh depending on the data provided)
    """
    vertices = np.ascontiguousarray(V, dtype=np.float64)
    edges = np.ascontiguousarray(E, dtype=np.int64) if E is not None else None
    faces = np.ascontiguousarray(F, dtype=np.int64) if F is not None else None
    cells = np.ascontiguousarray(C, dtype=np.int64) if C is not None else None
    rawmesh = RawMeshData()

    if vertices.shape[1]<3:
        vertices = np.pad(vertices, ((0,0),(0,3-vertices.shape[1])))
    elif vertices.shape[1]!=3: 
        raise Exception("Vertex array should have shape (n,3)")
    n_vert = vertices.shape[0]
    rawmesh.vertices = vertices

    if edges is not None:
        if np.any(edges>=n_vert): raise Exception("Edges indices should be between 0 and n_vertices")
        if edges.shape[1]!=2: raise Exception("Edge array should have shape (n,2)")
        rawmesh.edges = edges

    if faces is not None:
        if np.any(faces>=n_vert): raise Exception("Face indices should be between 0 and n_vertices")
        rawmesh.faces = faces

    if cells is not None:
        if np.any(cells>=n_vert): raise Exception("Cell indices should be between 0 and n_vertices")
        rawmesh.cells = cells
    return rawmesh.instantiate()


##### Mesh saving #####
def save(mesh : Mesh, filename: str) -> None:
    """ Saves the mesh data into a file.

    Supported file formats are:
    
    - wavefront (.obj)  
    - medit (.mesh)  
    - geogram (.geogram_ascii)
    - ply  
    - stl  
    - vtk
    - msh
    - off  
    - tet  
    - xyz

    Parameters:
        filename (str): The output file path

    Raises:
        Exception: Unsupported file extension
    """
    write_by_extension(mesh, filename)



##### Other mesh manipulations ######

def copy(mesh : Mesh, copy_attributes=False, copy_connectivity=False) -> Mesh:
    """Makes a copy of the input mesh

    Parameters:
        mesh (Mesh): input mesh
        copy_attributes (bool, optional): whether to also copy attributes data. Defaults to False.
        copy_connectivity (bool, optional): whether to also copy the connectivity data. Defaults to False.
    Returns:
        Mesh: a hard copy of the given mesh
    """
    from copy import deepcopy
    def make_copy(container, id):
        if isinstance(container, CornerDataContainer):
            return deepcopy(container) if copy_attributes else CornerDataContainer(container._elem.copy(), container._adj.copy(), id=id)
        elif isinstance(container, DataContainer):
            return deepcopy(container) if copy_attributes else DataContainer(container._data.copy(), id=id)
        else:
            raise Exception("Should not happen")
        
    match mesh:
        case PointCloud():
            copy_mesh = PointCloud(make_copy(mesh.vertices, "vertices"))

        case PolyLine():
            copy_mesh = PolyLine(
                vertices = make_copy(mesh.vertices, "vertices"),
                edges = make_copy(mesh.edges, "edges")
            )
        case SurfaceMesh():
            copy_mesh = SurfaceMesh(
                vertices = make_copy(mesh.vertices, "vertices"),
                edges = make_copy(mesh.edges, "edges"),
                faces = make_copy(mesh.faces, "faces"),
                face_corners= make_copy(mesh.face_corners, "face_corners")
            )
        case VolumeMesh():
            copy_mesh = SurfaceMesh(
                vertices = make_copy(mesh.vertices, "vertices"),
                edges = make_copy(mesh.edges, "edges"),
                faces = make_copy(mesh.faces, "faces"),
                face_corners= make_copy(mesh.face_corners, "face_corners"),
                cells = make_copy(mesh.cells, "cells"),
                cell_corners = make_copy(mesh.cell_corners, "cell_corners"),
                cell_faces = make_copy(mesh.cell_faces, "cell_faces")
            )
    if copy_connectivity and hasattr(mesh, "connectivity"):
        copy_mesh.connectivity = deepcopy(mesh.connectivity)
    return copy_mesh


def merge(mesh_list : list) -> Mesh:
    """Merges a list of independents meshes as a unique mesh.

    Parameters:
        mesh_list (list): a list of meshes. If empty, this function returns None

    Raises:
        Exception: fails if provided meshes are not of the same type (i.e. cannot merge a PointCloud with a SurfaceMesh).

    Returns:
        Mesh: the merged mesh.
    """
    if len(mesh_list)==0: return None
    if not np.all([isinstance(m, type(mesh_list[0])) for m in mesh_list]):
        raise Exception("`merge` received meshes of different types.")
    vertices = [m.vertices for m in mesh_list]
    vertex_offset = np.zeros(len(mesh_list)+1)
    vertex_offset[1:] = np.cumsum([len(v) for v in vertices])
    vertices = np.concatenate(vertices)

    edges, faces, cells = None, None, None
    if hasattr(mesh_list[0], "edges"):
        edges = np.concatenate([m.edges._data+vertex_offset[i] for i,m in enumerate(mesh_list)])
    if hasattr(mesh_list[0], "faces"):
        faces =  np.concatenate([m.faces._data+vertex_offset[i] for i,m in enumerate(mesh_list)])
    if hasattr(mesh_list[0], "cells"):
        cells = np.concatenate([m.cells._data+vertex_offset[i] for i,m in enumerate(mesh_list)])
    return from_arrays(vertices, edges, faces, cells)


def reorder_vertices(mesh : Mesh, new_indices: Iterable) -> Mesh:
    """Reorders vertex indices in a mesh object.

    Args:
        mesh (Mesh): input mesh
        new_indices (Iterable): a container with a permutation of vertices. New index of vertex L[i] will be i.

    Returns:
        Mesh: a copy of the mesh with permuted vertices
    """
    new_indices = np.asarray(new_indices)
    assert new_indices.shape == (len(mesh.vertices),)
    ind = np.argsort(new_indices) # "inverted list"
    perm = np.vectorize(lambda i: ind[i])

    V, E, F, C = None, None, None, None
    
    V = mesh.vertices._data[new_indices,:]
    if hasattr(mesh, "edges"): E = perm(mesh.edges._data.copy())
    if hasattr(mesh, "faces"): F = perm(mesh.faces._data.copy())
    if hasattr(mesh, "cells"): C = perm(mesh.cells._data.copy())
    return from_arrays(V, E, F, C)