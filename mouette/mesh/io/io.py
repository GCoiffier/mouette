import meshio
import numpy as np
import warnings
from ..mesh_data import RawMeshData

from ...utils import get_extension
from .geogram_ascii import import_geogram_ascii, export_geogram_ascii
from .obj import import_obj, export_obj
# from .off import import_off, export_off
# from .medit import import medit, export_medit
from .xyz import import_xyz, export_xyz
from .tet import import_tet, export_tet
from .ply import import_ply



def convert_meshio_to_mouette_mesh(meshio_data : meshio.Mesh) -> RawMeshData:
    output_data = RawMeshData()
    
    V = meshio_data.points
    if V.shape[1]<3:
        V = np.pad(V, ((0,0),(0,3-V.shape[1])))
    output_data.vertices += list(V)
    
    filtered_point_attributes = {"medit:ref"}
    for attr_name, attr_data in meshio_data.point_data.items():
        if attr_name in filtered_point_attributes: continue
        output_data.vertices.register_array_as_attribute(attr_name, attr_data)
    
    if "line" in meshio_data.cells_dict:
        output_data.edges += list(meshio_data.cells_dict["line"])

    for face_type in ("triangle", "quad"):
        if face_type in meshio_data.cells_dict:
            output_data.faces += list(meshio_data.cells_dict[face_type])
   
    for cell_type in ("tetra", "hexahedron"):
        if cell_type in meshio_data.cells_dict:
            output_data.cells += list(meshio_data.cells_dict[cell_type])
    
    filtered_cell_attributes = {"medit:ref"}
    for attr_name, attr_data in meshio_data.cell_data_dict.items():
        if attr_name in filtered_cell_attributes: continue
            
    return output_data


def _attribute_as_array(attr, n : int) -> np.ndarray:
    """Dense (n,) or (n,k) array of an attribute. Returns None if the type cannot be exported."""
    arr = np.asarray(attr.as_array(n)).reshape((n,-1))
    if arr.shape[1]==1: arr = arr[:,0]
    if arr.dtype==bool: arr = arr.astype(np.uint8)
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        return None
    return arr


def convert_to_meshio_data(mesh : RawMeshData, ext : str = None) -> meshio.Mesh:
    """Converts a mouette mesh into a meshio mesh. The conversion depends on the target format:
    - medit (.mesh, .meshb): same output as `export_medit` (hard edges only if defined, triangles/quads, tetrahedra/hexahedra, no attributes)
    - .vtk, .vtu: all elements, vertex attributes as point data and edge/face/cell attributes as cell data
    - other formats: all elements (no polygons for .msh), attributes are dropped
    """
    ext = (ext or "").lower()
    is_medit, is_msh, is_vtk = ext in ("mesh", "meshb"), ext=="msh", ext in ("vtk", "vtu")
    allow_polygons = not (is_medit or is_msh)  # medit and gmsh files have no polygon element type
    containers = {name : getattr(mesh, name) for name in ("vertices", "edges", "faces", "cells") if hasattr(mesh, name)}
    _ELEMENT_TYPES = {
        "edges" : {2: "line"},
        "faces" : {3: "triangle", 4: "quad"},
        "cells" : {4: "tetra", 8: "hexahedron"},
    }

    blocks = [] # (meshio cell type, container name, element indices)
    for name, types in _ELEMENT_TYPES.items():
        container = containers.get(name)
        if container is None or container.empty(): continue
        ids = range(len(container))
        if is_medit and name=="edges" and container.has_attribute("hard_edges"):
            ids = list(container.get_attribute("hard_edges"))
        by_size = dict()
        for i in ids:
            by_size.setdefault(len(container[i]), []).append(i)
        for size, size_ids in sorted(by_size.items()):
            cell_type = types.get(size, "polygon" if (name=="faces" and allow_polygons) else None)
            if cell_type is None:
                warnings.warn(f"{len(size_ids)} {name} with {size} vertices cannot be exported in .{ext} format. Skipping.")
                continue
            blocks.append((cell_type, name, np.array(size_ids)))

    point_data, cell_data = dict(), dict()
    attributes = [(cname, attr) for cname, c in containers.items() for attr in c.attributes if not c.get_attribute(attr).empty()]
    if is_vtk:
        names = [attr for _, attr in attributes]
        for cname, attr in attributes:
            container = containers[cname]
            arr = _attribute_as_array(container.get_attribute(attr), len(container))
            if arr is None:
                warnings.warn(f"Attribute '{attr}' on {cname} has a type that cannot be exported. Skipping.")
                continue
            if cname=="vertices":
                point_data[attr] = arr
            elif any(bc==cname for _, bc, _ in blocks):
                key = attr if names.count(attr)==1 else f"{cname}:{attr}"
                cell_data[key] = [arr[ids] if bc==cname else np.zeros((len(ids),) + arr.shape[1:], dtype=arr.dtype) for _, bc, ids in blocks]
    elif attributes and not is_medit:
        warnings.warn(f"Attributes are not exported in .{ext} format. Dropping {', '.join(f'{c}:{a}' for c,a in attributes)}.")

    cells = [(cell_type, np.array([containers[cname][i] for i in ids], dtype=np.int64)) for cell_type, cname, ids in blocks]
    V = np.asarray(mesh.vertices, dtype=np.float64).reshape((-1,3))
    return meshio.Mesh(V, cells=cells, point_data=point_data, cell_data=cell_data)



def read_by_extension(filename : str) -> RawMeshData:
    ext = get_extension(filename).lower()
    match ext:
        case "geogram_ascii":
            return import_geogram_ascii(filename)
        
        case "xyz":
            return import_xyz(filename)
        
        case "tet":
            return import_tet(filename)
        
        case "obj":
            try:
                data = meshio.read(filename)
                return convert_meshio_to_mouette_mesh(data)
            except (meshio.ReadError, ValueError) as e:
                return import_obj(filename)
            
        case "ply":
            try:
                data = meshio.read(filename)
                return convert_meshio_to_mouette_mesh(data)
            except meshio.ReadError as e:
                return import_ply(filename)
                   
        case _:
            # .msh, .stl, .mesh, .meshb, .off are supported
            data = meshio.read(filename)
            return convert_meshio_to_mouette_mesh(data)



def write_by_extension(mesh : RawMeshData, filename : str):
    ext = get_extension(filename).lower()
    match ext:
        case "geogram_ascii":
            export_geogram_ascii(mesh, filename)
        case "xyz":
            export_xyz(mesh, filename)
        case "tet":
            export_tet(mesh, filename)
        case "obj":
            export_obj(mesh, filename)
        case _:
            meshio_data = convert_to_meshio_data(mesh, ext)
            # .msh is ambiguous in meshio (ansys or gmsh): use gmsh 2.2 which handles mixed cell types without entity tags
            file_format = "gmsh22" if ext=="msh" else None
            meshio.write(filename, meshio_data, file_format=file_format)