from ...utils import get_extension
from .geogram_ascii import import_geogram_ascii, export_geogram_ascii
from .obj import import_obj, export_obj
from .medit import import_medit, export_medit
from .off import import_off, export_off
from .xyz import import_xyz, export_xyz
from .tet import import_tet, export_tet
from .stl import import_stl, export_stl
from .ply import import_ply, export_ply

def read_by_extension(filename : str):
    ext = get_extension(filename)
    import_fun = {
        "geogram_ascii" : import_geogram_ascii,
        "obj" : import_obj,
        "mesh" : import_medit,
        "off" : import_off,
        "xyz" : import_xyz,
        "tet" : import_tet,
        "stl" : import_stl,
        "ply" : import_ply
    }.get(ext.lower(), None)
    if import_fun is None:
        raise Exception("Unsupported file extension '{}'".format(ext))
    data = import_fun(filename)
    return data

def write_by_extension(mesh : "RawMeshData", filename : str):
    ext = get_extension(filename)
    export_fun = {
        "geogram_ascii" : export_geogram_ascii,
        "obj" : export_obj,
        "mesh" : export_medit,
        "off" : export_off,
        "xyz" : export_xyz,
        "tet" : export_tet,
        "stl" : export_stl,
        "ply" : export_ply
    }.get(ext.lower(), None)
    if export_fun is None:
        raise Exception("Unsupported file extension '{}'".format(ext))
    export_fun(mesh, filename)