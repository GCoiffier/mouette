from ..mesh_data import RawMeshData
from ..datatypes.type_checks import *
from collections import deque

def import_tet(path : str):
    """
    Read a .tet file : 
		/////////////////
        n vertices
        m tets
        v1_x v1_y v1_z
        v2_x v2_y v2_z
        ...
        vn_x vn_y vn_z
        4 v1_1 v1_2 v1_3 v1_4
        4 v2_1 v2_2 v2_3 v2_4
        ...
        4 vm_1 vm_2 vm_3 vm_3					
        /////////////////
    Parameters:
        path (str): the input file path
    """

    with open(path, 'r' ) as file:
        output = parse_tet_data(file.readlines())
    return output

def parse_tet_data(data):
    data = deque(data)
    get_line = lambda : data.popleft().strip().split()
    output = RawMeshData()
    nvert = int(get_line()[0])
    ntet  = int(get_line()[0])
    for _ in range(nvert):
        v = [float(u) for u in get_line()]
        output.vertices.append(v)
    for _ in range(ntet):
        c = tuple((int(u) for u in get_line()[1:]))
        output.cells.append(c)
    return output

def export_tet(mesh, path):
    with open( path, 'w' ) as ofile:
        ofile.write("{} vertices\n".format(len(mesh.vertices)))
        ofile.write("{} tets\n".format(len(mesh.cells)))

        for vtx in mesh.vertices:
            ofile.write(' '.join(['{}'.format(v) for v in vtx])+'\n')
        for cell in mesh.cells:
            str_face = "{} ".format(len(cell))
            str_face += " ".join([str(v) for v in cell])
            ofile.write(str_face + "\n")