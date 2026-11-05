from ..mesh import PointCloud, RawMeshData
import numpy as np

def import_xyz(filepath : str) -> RawMeshData:
    """Imports a point cloud from a .xyz file on the disk

    Parameters:
        filepath (str): path to the .xyz file

    Returns:
        RawMeshData: parsed file
    """
    vertices = []
    nrmls = []
    with open(filepath, 'r') as f:
        for v in f.readlines():
            data = [float(x) for x in v.strip().split()]
            if len(data)==1: continue
            vertices.append(data[:3])
            if len(data)>=6:
                nrmls.append(data[3:6])
    obj = RawMeshData()
    obj.vertices.set_data(np.ascontiguousarray(vertices))
    if len(nrmls)==len(vertices):
        obj.vertices.register_array_as_attribute("normals", np.ascontiguousarray(nrmls))
    return obj

def export_xyz(mesh : PointCloud, filepath : str) -> None:
    """Exports a Point Cloud to a .xyz file

    Parameters:
        mesh (PointCloud): the object to be exported
        filepath (str): path to the .xyz file
    """
    with open(filepath, 'w') as f:
        if mesh.vertices.has_attribute("normals"):
            normals = mesh.vertices.get_attribute("normals")
            for iv,v in enumerate(mesh.vertices):
                nv = normals[iv]
                f.write("{} {} {} {} {} {}\n".format(v[0], v[1], v[2], nv[0], nv[1], nv[2]))
        else:
            for v in mesh.vertices:
                f.write("{} {} {}\n".format(v[0], v[1], v[2]))