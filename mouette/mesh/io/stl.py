import stl_reader
import struct
from ..mesh_data import RawMeshData
from collections import deque

def is_stl_ascii(path : str):
    try:
        with open(path, 'r', encoding="utf-8") as f:
            return "solid" in f.readline().strip().split()
    except Exception as e:
        return False

def import_stl(path : str):
    if is_stl_ascii(path):
        return _import_stl_ascii(path)
    else:
        vertices, faces =  stl_reader.read(path)
        out = RawMeshData()
        out.vertices += list(vertices)
        out.faces += list(faces)
        return out

def _import_stl_ascii(path : str):
    data = deque()
    with open(path, "r", encoding="utf-8") as stlf:
        data = deque([x.strip().split() for x in stlf.readlines()])
    
    out = RawMeshData()
    normals = out.faces.create_attribute("normals", float, 3)
    iF = 0
    while data:
        line = data.popleft()
        if line[0]=="solid": continue
        
        elif line[0]=="facet":
            assert line[1] == "normal"
            normals[iF] = [float(x) for x in line[2:]]
            out_loop = data.popleft()
            assert out_loop[0] == "outer"
            vertices = []
            for _ in range(3):
                v = data.popleft()
                assert v[0] == "vertex"
                vertices.append([float(x) for x in v[1:]])
            end_loop = data.popleft()
            assert end_loop[0] == "endloop"
            endfacet = data.popleft()
            assert endfacet[0] == "endfacet"

            out.vertices += vertices
            out.faces.append((3*iF,3*iF+1,3*iF+2))
            iF += 1

        elif line[0] == "endsolid": continue
    return out

# Exporter adapted from https://gist.github.com/ryansturmer/9329299
# License: MIT License 

def export_stl(mesh, path :str):
    if not hasattr(mesh, "faces"):
        print("[Warning] no faces detected in the object. Nothing to write. Returning.")
        return
    with open(path, 'wb') as fp:
        writer = Binary_STL_Writer(fp)
        writer.write(mesh)

class Binary_STL_Writer:
    BINARY_HEADER ="80sI"
    BINARY_FACET = "12fH"

    def __init__(self, stream):
        self.counter = 0
        self.fp = stream
        self._write_header()

    def _write_header(self):
        self.fp.seek(0)
        self.fp.write(struct.pack(Binary_STL_Writer.BINARY_HEADER, b'Python Binary STL Writer', self.counter))

    def _write_triangle(self, p1, p2, p3):
        self.counter += 1
        data = [
            0., 0., 0.,
            p1[0], p1[1], p1[2],
            p2[0], p2[1], p2[2],
            p3[0], p3[1], p3[2],
            0
        ]
        self.fp.write(struct.pack(Binary_STL_Writer.BINARY_FACET, *data))

    def write(self, mesh):
        self._write_header()
        for face in mesh.faces:
            pts = [mesh.vertices[v] for v in face]
            if len(face) == 3:
                self._write_triangle(*pts)
            elif len(face) == 4:
                self._write_triangle(pts[0], pts[1], pts[2])
                self._write_triangle(pts[2], pts[3], pts[0])
            else:
                raise ValueError('Only triangular and quad faces are supported.')
        self._write_header() # rewrite the header with the correct value of self.counter