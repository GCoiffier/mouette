from ..mesh_data import RawMeshData
from ...geometry import Vec
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

        if surfacic, 3 is given at beginning of each simplex line (triangle), otherwise 4 (tetrahedron)
    Parameters:
        path (str): the input file path
    """

    with open(path, 'r' ) as file:
        output = parse_off_data(file.readlines())
    return output

def parse_off_data(data):
    output = RawMeshData()
    
    data = [x.strip().split() for x in data]
    # remove empty lines from data
    data = deque([x for x in data if x])

    header = data.popleft()[0]
    if (header != "OFF"): # file always starts with OFF
        raise Exception("Import OFF file : OFF header missing.")

    nv,nf,ne = (int(u) for u in data.popleft())

    for _ in range(nv):
        vertex = [float(u) for u in data.popleft()]
        output.vertices.append(Vec(vertex))

    i_f, i_c = 0, 0
    for _ in range(nf):
        simplex = data.popleft()
        nvi = int(simplex[0])
        if nvi==3:
            face = [int(u) for u in simplex[1:nvi+1]]
            output.faces.append(face)
            output.face_corners += [(x,i_f) for x in face]
            i_f += 1
        elif nvi==2:
            a,b = simplex[1], simplex[2]
            output.edges.append((min(a,b), max(a,b)))
        elif nvi==4:
            cell = [int(u) for u in simplex[1:nvi+1]]
            output.cells.append(cell)
            output.cell_corners += [(x,i_c) for x in cell]
            i_c += 1
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