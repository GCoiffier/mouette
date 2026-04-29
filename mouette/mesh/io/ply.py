from ..mesh_data import RawMeshData
import pyminiply
import numpy as np

def import_ply(path : str):
    vertices, faces, normals, uv, color = pyminiply.read(path)
    out = RawMeshData()
    out.vertices += list(vertices)
    if np.any(np.asarray(faces)>=len(out.vertices)): raise Exception("Face indices should be between 0 and n_vertices")
    out.faces += list(faces)
    if normals.size>0:
        out.vertices.register_array_as_attribute("normals", normals)
    if uv.size>0:
        out.vertices.register_array_as_attribute("uv_coords", uv)
    if color.size>0:
        out.vertices.register_array_as_attribute("color", color)
    return out


def export_ply(mesh, path : str):
    raise NotImplementedError