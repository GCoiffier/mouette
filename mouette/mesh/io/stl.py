import stl_reader
import struct
from ..mesh_data import RawMeshData

def import_stl(path : str):
    vertices, faces =  stl_reader.read(path)
    data = RawMeshData()
    data.vertices += list(vertices)
    data.faces += list(faces)
    return data

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