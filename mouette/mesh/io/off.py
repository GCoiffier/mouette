import numpy as np
from ..datatypes.raw import RawMeshData
from collections import deque

def import_off(path : str):
    """
    Read a .off file : 
		/////////////////
        OFF
        #v #f #e
        v1_x v1_y v1_z
        v2_x v2_y v2_z
        ...

        vn_x vn_y vn_z
        3 v1_1 v1_2 v1_3
        3 v2_1 v2_2 v2_3
        ...
        3 vm_1 vm_2 vm_3
        /////////////////

        if the file represents a surface, 3 is given at beginning of each simplex line (triangles), otherwise 4 (tetrahedra)

    Parameters:
        path (str): the input file path
    """
    with open(path, 'r' ) as file:
        output = parse_off_data(file.readlines())
    return output

def parse_off_data(data): 
    data = [x.strip().split() for x in data]

    # remove empty lines and comments from data
    data = deque([x for x in data if x and not x[0]=="#"])

    header = data.popleft()[0]
    if (header != "OFF"): # file always starts with OFF
        raise Exception("Import OFF file : OFF header missing.")

    output = RawMeshData()
    nv,nf,ne = (int(u) for u in data.popleft())

    for _ in range(nv):
        output.vertices.append([float(u) for u in data.popleft()])

    for _ in range(nf):
        simplex = data.popleft()
        nvi = int(simplex[0])
        simplex = [int(u) for u in simplex[1:nvi+1]]
        if   nvi==3: output.faces.append(simplex)
        elif nvi==4: output.cells.append(simplex)
    return output

def export_off(mesh, path):
    with open( path, 'w' ) as ofile:
        ofile.write("OFF\n")
        nums = "{} {} {}\n".format(len(mesh.vertices), len(mesh.faces), len(mesh.edges))
        ofile.write(nums)
        for vtx in mesh.vertices:
            ofile.write(' '.join(['{}'.format(v) for v in vtx])+'\n')
        for face in mesh.faces:
            str_face = "{} ".format(len(face))
            str_face += " ".join([str(v) for v in face])
            ofile.write(str_face + "\n")