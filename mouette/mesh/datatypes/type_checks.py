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
    if typ==PointCloud: return "mouette.PointCloud"    
    if typ==PolyLine: return "mouette.PolyLine"    
    if typ==SurfaceMesh: return "mouette.SurfaceMesh"    
    if typ==VolumeMesh: return "mouette.VolumeMesh"
    raise BadMeshTypeException(typ, "[Mouette] Unknown mesh type")

def allowed_mesh_types(*allowed_types : list):
    def decorator(function):
        def wrapper(*args, **kwargs):
            for arg in args:
                if isinstance(arg, Mesh) and type(arg) not in filter(lambda t : issubclass(t,Mesh), allowed_types):
                    raise BadMeshTypeException(type(arg), "[Mouette] Mesh type '{}' is not allowed for this function. Allowed types : {}".format( type_to_str(type(arg)), [type_to_str(u) for u in allowed_types]))
            result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator

def forbidden_mesh_types(*forbidden_types : list):
    def decorator(function):
        def wrapper(*args, **kwargs):
            for arg in args:
                if isinstance(arg, Mesh) and type(arg) in filter(lambda t : issubclass(t,Mesh), forbidden_types):
                    raise BadMeshTypeException("[Mouette] Mesh type '{}' is forbidden for this function. Forbidden types are {}".format( type_to_str(type(arg)), [type_to_str(u) for u in forbidden_types]))
            result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator