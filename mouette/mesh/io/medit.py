from ..datatypes import *
from ..mesh_data import RawMeshData
from collections import deque

def parse_field(data : deque, container, nlines, nelem):
    for _ in range(nlines):
        line = data.popleft().split()
        d = [int(u.strip())-1 for u in line][:nelem]
        container.append(d)

def import_medit(path):
    obj = RawMeshData()
    data = deque()
    with open(path, 'r' ) as meditf:
        data = deque([x.strip() for x in meditf.readlines()])
                
    while data:
        line = data.popleft()

        if line=="End": break # end of file

        elif line=="Vertices":
            nv = int(data.popleft())
            for _ in range(nv):
                line = data.popleft().split()
                vertex = [float(u.strip()) for u in line[:3]]
                obj.vertices.append(vertex)

        elif line=="Edges":
            ne = int(data.popleft())
            parse_field(data, obj.edges, ne, 2)

        elif line=="Triangles":
            nt = int(data.popleft())
            parse_field(data, obj.faces, nt, 3)

        elif line=="Quadrilaterals":
            nq = int(data.popleft())
            parse_field(data, obj.faces,  nq, 4)

        elif line=="Tetrahedra":
            nc = int(data.popleft())
            parse_field(data, obj.cells, nc, 4)
        
        elif line=="Hexahedra":
            nc = int(data.popleft())
            parse_field(data, obj.cells, nc, 6)

    return obj

def count_cells(mesh : RawMeshData):
    cell_count = [0,0,0]
    # 0 = hexahedra
    # 1 = tetrahedra
    # 2 = else
    for c in mesh.cells:
        N = len(c)
        if N==8: cell_count[0] += 1
        elif N==4 : cell_count[1] += 1
        else : cell_count[2] += 1
    return cell_count

def count_faces(mesh : RawMeshData):
    face_count = [0, 0, 0]
    # 0 = quad
    # 1 = tri
    # 2 = else
    for f in mesh.faces:
        if len(f)==4 : face_count[0] +=1
        elif len(f)==3 : face_count[1] +=1
        else: face_count[2] += 1
    return face_count

def export_medit(mesh : RawMeshData, path):
    with open(path, 'w') as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n") # does not matter -> 3 as default

        if not mesh.vertices.empty():
            f.write("Vertices\n")
            f.write("{}\n".format(len(mesh.vertices)))
            for v in mesh.vertices:
                f.write("{} {} {} 1\n".format(v[0], v[1], v[2]))
            f.write("\n")

        if hasattr(mesh, "edges") and not mesh.edges.empty():
            f.write("Edges\n")
            if mesh.edges.has_attribute("hard_edges"):
                f.write("{}\n".format(len(mesh.edges.get_attribute("hard_edges"))))
                for e in mesh.edges.get_attribute("hard_edges"):
                    a,b = mesh.edges[e]
                    f.write("{} {} 1\n".format(a+1,b+1))
            else:
                f.write("{}\n".format(len(mesh.edges)))
                for a,b in mesh.edges:
                    f.write("{} {} 1\n".format(a+1,b+1))
            f.write("\n")
        
        if hasattr(mesh, "faces") and not mesh.faces.empty():
            nquad, ntri, nother = count_faces(mesh)
            if nother>0:
                print("[Warning] faces that are not triangle or quad detected.\nExporting as .mesh will result in a loss of data.")

            # export triangles
            if ntri>0:
                f.write("Triangles\n{}\n".format(ntri))
                for face in mesh.faces:
                    if len(face)==3:
                        f.write("{} {} {} 1\n".format(*(i+1 for i in face)))
                f.write("\n")

            # export quads
            if nquad>0:
                f.write("Quadrilaterals\n{}\n".format(nquad))
                for face in mesh.faces:
                    if len(face)==4:
                        f.write("{} {} {} {} 1\n".format(*(i+1 for i in face)))
                f.write("\n")
            
        if hasattr(mesh, "cells") and not mesh.cells.empty():
            nhex,ntet, nother = count_cells(mesh)
            if nother>0:
                print("[Warning] Cells that are not tetrahedron or hexahedron detected.\nExporting as .mesh will result in a loss of data.")
            
            # export hexahedra
            if nhex>0:
                f.write("Hexahedra\n{}\n".format(nhex))
                for c in mesh.cells:
                    if len(c)==8:
                        f.write("{} {} {} {} {} {} {} {} 1\n".format(*(i+1 for i in c)))
                f.write("\n")
            
            # export tetrahedra
            if ntet>0:
                f.write("Tetrahedra\n{}\n".format(ntet))
                for c in mesh.cells:
                    if len(c)==4:
                        f.write("{} {} {} {} 1\n".format(*(i+1 for i in c)))
                f.write("\n")
    