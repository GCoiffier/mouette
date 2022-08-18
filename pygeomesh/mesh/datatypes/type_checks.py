from .base import Mesh
from .pointcloud import PointCloud
from .linear import PolyLine
from .surface import SurfaceMesh
from .volume import VolumeMesh

class BadMeshTypeException(Exception):
    def __init__(self, typ, message):
        self.type = typ
        super().__init__(message)

def type_to_str(typ):
    if typ==PointCloud: return "GEO.PointCloud"    
    if typ==PolyLine: return "GEO.PolyLine"    
    if typ==SurfaceMesh: return "GEO.SurfaceMesh"    
    if typ==VolumeMesh: return "GEO.VolumeMesh"
    raise BadMeshTypeException(typ, "[Pygeomesh] Unknown mesh type")

def allowed_mesh_types(*allowed_types : list):
    def decorator(function):
        def wrapper(*args, **kwargs):
            for arg in args:
                if isinstance(arg, Mesh) and type(arg) not in allowed_types:
                    raise BadMeshTypeException(type(arg), "[Pygeomesh] Mesh type '{}' is not allowed for this function. Allowed types : {}".format( type_to_str(type(arg)), [type_to_str(u) for u in allowed_types]))
            result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator

def forbidden_mesh_types(*forbidden_types : list):
    def decorator(function):
        def wrapper(*args, **kwargs):
            for arg in args:
                if isinstance(arg, Mesh) and type(arg) in forbidden_types:
                    raise BadMeshTypeException("[Pygeomesh] Mesh type '{}' is forbidden for this function. Forbidden types are {}".format( type_to_str(type(arg)), [type_to_str(u) for u in forbidden_types]))
            result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator