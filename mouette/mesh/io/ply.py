from ..mesh_data import RawMeshData
from plyfile import PlyData, PlyElement
import numpy as np

def import_ply(path : str):
    plydata = PlyData.read(path)
    out = RawMeshData()
    for element in plydata.elements:
        if element.name == "vertex":
            pos = np.vstack([element["x"], element["y"], element["z"]])
            out.vertices += list(pos.T)
        elif element.name == "face":
            for face in element["vertex_indices"]:
                out.faces.append(face)
        elif element.name == "edge":
            ind = np.vstack([element["v1"], element["v2"]])
            out.edges += list(ind.T)
        elif element.name == "cell":
            for cell in element["vertex_indices"]:
                out.cells.append(cell)
    return out


def export_ply(mesh, path : str):
    V = np.array([tuple(p) for p in mesh.vertices], dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    elV = PlyElement.describe(V,"vertex")

    E = np.array([tuple(e) for e in mesh.edges], dtype=[('v1','i4'), ('v2','i4')])
    elE = PlyElement.describe(E,"edge")

    F = np.array([(face,) for face in mesh.faces],dtype=[('vertex_indices', 'i4', (len(mesh.faces[0]),))])
    elF = PlyElement.describe(F,"face")

    PlyData([elV,elE, elF]).write(path)
