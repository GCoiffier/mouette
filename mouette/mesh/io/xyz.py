from ..mesh_data import RawMeshData
from ...geometry import Vec
from ..mesh import PointCloud

def import_xyz(filepath : str) -> RawMeshData:
    """Imports a point cloud from a .xyz file on the disk

    Parameters:
        filepath (str): path to the .xyz file

    Returns:
        RawMeshData: parsed file
    """
    obj = RawMeshData()
    with open(filepath, 'r') as f:
        for v in f.readlines():
            v = [float(x) for x in v.strip().split()[:3]]
            obj.vertices.append(v)
    return obj

def export_xyz(mesh : PointCloud, filepath : str) -> None:
    """Exports a Point Cloud to a .xyz file

    Parameters:
        mesh (PointCloud): the object to be exported
        filepath (str): path to the .xyz file
    """
    with open(filepath, 'w') as f:
        for v in mesh.vertices:
            f.write("{} {} {}\n".format(v[0], v[1], v[2]))