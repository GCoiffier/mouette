from .mesh_data import RawMeshData
from .datatypes import Mesh, PointCloud, PolyLine, SurfaceMesh, VolumeMesh
from .io.io import read_by_extension, write_by_extension

def new_point_cloud() -> PointCloud: 
    """
    Create a new empty point cloud

    Returns:
        PointCloud
    """
    return PointCloud()
    
def new_polyline() -> PolyLine:
    """
    Creates a new empty poly line

    Returns:
        PolyLine
    """
    return PolyLine()

def new_surface() -> SurfaceMesh:
    """
    Creates a new empty surface mesh

    Returns:
        SurfaceMesh
    """
    return SurfaceMesh()

def new_volume() -> VolumeMesh:
    """
    Creates a new empty volume mesh

    Returns:
        VolumeMesh
    """
    return VolumeMesh()


def _instanciate_raw_mesh_data(mesh_data : RawMeshData, dim : int = None) -> Mesh:
    mesh_data.prepare()
    if dim is None: dim = -1
    dim = max(dim, mesh_data.dimensionality)
    if dim==0: return PointCloud(mesh_data)
    if dim==1: return PolyLine(mesh_data)
    if dim==2: return SurfaceMesh(mesh_data)
    if dim==3: return VolumeMesh(mesh_data)

def load(filename : str, dim : int = None, raw : bool = False) -> Mesh:
    """
    Loads a Mesh object from a file on the disk

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
    data = read_by_extension(filename)
    if raw: return data
    return _instanciate_raw_mesh_data(data, dim)
    

def save(mesh : Mesh, filename: str, ignore_elements:set = None) -> None:
    """ Saves the mesh data into a file

    Parameters:
        filename (str): The output file path

    Raises:
        Exception: Unsupported file extension
    """
    if isinstance(mesh,VolumeMesh) and ".geogram" in filename:
        mesh.connectivity._compute_adjacent_cell()
    raw_mesh = RawMeshData(mesh) # get rid of connectivity and additional attributes depending on dimension
    if ignore_elements is not None:
        if "edges" in ignore_elements: raw_mesh.edges.clear()
        if "faces" in ignore_elements:
            raw_mesh.faces.clear()
            raw_mesh.face_corners.clear()
        if "cells" in ignore_elements: 
            raw_mesh.cells.clear()
            raw_mesh.cell_corners.clear()
            raw_mesh.cell_faces.clear()
    write_by_extension(raw_mesh, filename)

def copy(mesh : Mesh, copy_attributes=False, copy_connectivity=False, copy_half_edges=False) -> Mesh:
    """[summary]

    Parameters:
        mesh (Mesh): [description]
        copy_attributes (bool, optional): [description]. Defaults to False.

    Returns:
        Mesh: a hard copy of the given mesh
    """
    from copy import deepcopy
    copy_mesh = type(mesh)() # instanciate a new mesh of the same type

    if copy_attributes:
        copy_mesh.vertices = deepcopy(mesh.vertices)
        if hasattr(mesh, "edges") : copy_mesh.edges = deepcopy(mesh.edges)
        if hasattr(mesh, "faces") :
            copy_mesh.faces = deepcopy(mesh.faces)
            copy_mesh.face_corners = deepcopy(mesh.face_corners)
        if hasattr(mesh, "cells"):
            copy_mesh.cells = deepcopy(mesh.cells)
            copy_mesh.cell_corners = deepcopy(mesh.cell_corners)
    else:
        # copy only _cont data of each container (connectivity data and not attribute data)
        copy_mesh.vertices._data = deepcopy(mesh.vertices._data)
        if hasattr(mesh, "edges"): copy_mesh.edges._data = deepcopy(mesh.edges._data)
        if hasattr(mesh, "faces") : 
            copy_mesh.faces._data = deepcopy(mesh.faces._data)
            copy_mesh.face_corners._data = deepcopy(mesh.face_corners._data)
        if hasattr(mesh, "cells") : 
            copy_mesh.cells._data = deepcopy(mesh.cells._data)
            copy_mesh.cell_corners._data = deepcopy(mesh.cell_corners._data)
        # _cont of face_corners and cell_corners are always empty
    if copy_connectivity and hasattr(mesh, "connectivity"):
        copy_mesh.connectivity = mesh.connectivity
    if copy_half_edges and hasattr(mesh, "half_edges"):
        copy_mesh.half_edges = mesh.half_edges
    return copy_mesh

def merge(mesh_list : list) -> Mesh:
    """Merges a list of independents meshes as a unique mesh.

    Parameters:
        mesh_list (list): the list of meshes. If empty, this function returns None

    Returns:
        Mesh: the merged mesh. Takes the type of the mesh with the largest dimensionality in the list (ie merging a Polyline with a Volume mesh returns a Volume mesh)
    """
    if len(mesh_list)==0: return None
    merged = RawMeshData()
    vertex_offset = 0
    for to_merge in mesh_list:
        merged.vertices += to_merge.vertices
        if hasattr(to_merge, "edges") : 
            merged.edges += [tuple((vertex_offset+u for u in e)) for e in to_merge.edges]
        if hasattr(to_merge, "faces") : 
            merged.faces += [tuple((vertex_offset+u for u in f)) for f in to_merge.faces]
        if hasattr(to_merge, "cells") :
            merged.cells += [tuple((vertex_offset+u for u in c)) for c in to_merge.cells]
        vertex_offset += len(to_merge.vertices)
    return _instanciate_raw_mesh_data(merged)