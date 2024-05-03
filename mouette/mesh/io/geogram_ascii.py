from ..mesh_data import RawMeshData
from ..mesh_attributes import Attribute
from ...geometry import Vec
import numpy as np
from enum import Enum
from ...config import NOT_AN_ID

class Chunk:

    class Type(Enum):
        HEAD = 0
        ATTR = 1
        ATTS = 2

        @classmethod
        def from_string(cls, txt : str):
            if txt=="[HEAD]": return cls.HEAD
            if txt=="[ATTR]": return cls.ATTR
            if txt=="[ATTS]": return cls.ATTS

    class Container(Enum):
        VERTICES = 0
        EDGES = 1
        FACES = 2
        FACE_CORNERS = 3
        CELLS = 4
        CELL_CORNERS = 5
        CELL_FACETS = 6

        @classmethod
        def from_string(cls, txt : str):
            if "vertices" in txt: return cls.VERTICES
            if "edges" in txt : return cls.EDGES
            if "facet_corners" in txt : return cls.FACE_CORNERS
            if "facets" in txt : return cls.FACES
            if "cell_corners" in txt : return cls.CELL_CORNERS
            if "cell_facets" in txt : return cls.CELL_FACETS
            if "cells" in txt : return cls.CELLS

        def to_string(self):
            return { 
                0 : "\"GEO::Mesh::vertices\"",
                1 : "\"GEO::Mesh::edges\"",
                2 : "\"GEO::Mesh::facets\"",
                3 : "\"GEO::Mesh::facet_corners\"",
                4 : "\"GEO::Mesh::cells\"",
                5 : "\"GEO::Mesh::cell_corners\"",
                6 : "\"GEO::Mesh::cell_facets\""
            }.get(self.value, None)

    def __init__(self, chunk_data):
        self.type : Chunk.Type = Chunk.Type.from_string(chunk_data[0])
        self.container : Chunk.Container = Chunk.Container.from_string(chunk_data[1]) # like GEO::Mesh::vertices
        
        if self.type == Chunk.Type.ATTR:
            self.name : str = chunk_data[2]
            self.data_type : Attribute.Type = Attribute.Type.from_string(chunk_data[3])
            self.data_size : int = int(chunk_data[4]) # number of data point per element
            self.n_data : int = int(chunk_data[5]) # number of elements
            if self.data_type == Attribute.Type.Float:
                self.data = [np.float64(x) for x in chunk_data[6:]]
            elif self.data_type == Attribute.Type.Int:
                self.data = [int(x) for x in chunk_data[6:]]
            elif self.data_type == Attribute.Type.Bool:
                self.data = [bool(int(x)) for x in chunk_data[6:]]
            else:
                # Attribute type cannot be exported
                self.data = chunk_data[6:]

        elif self.type == Chunk.Type.ATTS:
            self.n : int = int(chunk_data[2])

def is_chunk_header(line : str) -> bool:
    return "[HEAD]" in line or "[ATTS]" in line or "[ATTR]" in line
    # ignore other type of blocks (parameters for geogram / graphite)

def import_attribute(chk : Chunk, attr: Attribute):
    for i in range(len(chk.data)//chk.n_data):
        val = []
        for j in range(chk.n_data):
            val.append(chk.data[chk.n_data*i + j])
        if chk.n_data==1 and val[0]!= attr.default_value:
            attr[i] = val[0]
        elif chk.n_data>1:
            attr[i] = val

def import_geogram_ascii(path):

    data = None
    with open(path, 'r') as f:
        data = [x.split("#")[0].strip() for x in f.readlines()] # split to remove comments

    # Detect chunk separators
    chunk_sep = [] # int values for header lines -> separators of data
    for i,line in enumerate(data):
        if is_chunk_header(line):
            chunk_sep.append(i)
    chunk_sep.append(len(data))

    # build chunk objects
    chunks = []
    for i in range(len(chunk_sep)-1):
        start =  chunk_sep[i]
        end = chunk_sep[i+1]
        chunk_data = []
        for k in range(start,end):
            chunk_data.append(data[k])
        chunks.append(chunk_data)
    chunks = [Chunk(data) for data in chunks]

    outmesh = RawMeshData()

    # First read ATTS chunks
    container_sizes = dict([(a,0) for a in Chunk.Container])
    for chk in chunks:
        if chk.type != Chunk.Type.ATTS: continue
        container_sizes[chk.container] = chk.n

    # Build facet index
    n_corner_in_facet = []
    facet_ptr = []

    n_corner_in_cell = []
    cell_ptr = []
    for chk in chunks:
        if chk.type == Chunk.Type.ATTR and chk.name == "\"GEO::Mesh::facets::facet_ptr\"":
            # facet sizes are provided : the mesh is not triangular
            for i in range(container_sizes[Chunk.Container.FACES]-1):
                n_corner_in_facet.append(chk.data[i+1] - chk.data[i])
            n_corner_in_facet.append(container_sizes[Chunk.Container.FACE_CORNERS] - chk.data[-1])
            facet_ptr = chk.data

        elif chk.type == Chunk.Type.ATTR and chk.name == "\"GEO::Mesh::cells::cell_ptr\"":
            # cell sizrs are provided : the mesh is not tetrahedral
            for i in range(container_sizes[Chunk.Container.CELLS]-1):
                n_corner_in_cell.append(chk.data[i+1] - chk.data[i])
            n_corner_in_facet.append(container_sizes[Chunk.Container.CELL_CORNERS] - chk.data[-1])
            cell_ptr = chk.data

    if len(n_corner_in_facet)==0 and container_sizes[Chunk.Container.FACES]>0:
        # We have faces but no facet_ptr chunk. By convention, all faces are triangles
        n_corner_in_facet = [3]*container_sizes[Chunk.Container.FACES]
        ptr = 0
        facet_ptr = []
        for c in n_corner_in_facet:
            facet_ptr.append(ptr)
            ptr += c

    if len(n_corner_in_cell)==0 and container_sizes[Chunk.Container.CELLS]>0:
        # We have cells but no cell_ptr chunk. By convention, all cells are tetrahedra
        n_corner_in_cell = [4]*container_sizes[Chunk.Container.CELLS]
        ptr = 0
        cell_ptr = []
        for c in n_corner_in_cell:
            cell_ptr.append(ptr)
            ptr += c
    
    # Build mesh data from chunk objects
    for chk in chunks:

        if chk.type == Chunk.Type.ATTS or chk.type == Chunk.Type.HEAD: continue # already treated and HEAD is ignored
        # chk.type is now [ATTR]

        # first handle special cases : vertices coordinates, edges, faces and cells connectivity
        if chk.container == Chunk.Container.VERTICES and chk.name == "\"point\"":
            assert chk.n_data==3 # points have three coordinates
            for i in range(len(chk.data)//chk.n_data):
                outmesh.vertices.append(Vec([chk.data[3*i], chk.data[3*i+1], chk.data[3*i+2] ]))
        
        elif chk.container == Chunk.Container.EDGES and chk.name == "\"GEO::Mesh::edges::edge_vertex\"":
            assert chk.n_data==2 # edges should have 2 pointers to vertices id
            for i in range(container_sizes[Chunk.Container.EDGES]):
                outmesh.edges.append([chk.data[2*i], chk.data[2*i+1]])
        
        elif chk.container == Chunk.Container.FACE_CORNERS and chk.name == "\"GEO::Mesh::facet_corners::corner_vertex\"":
            assert chk.n_data==1
            for i in range(container_sizes[Chunk.Container.FACES]):
                ncf = n_corner_in_facet[i]
                ptr = facet_ptr[i]
                face = [chk.data[ptr+_i] for _i in range(ncf)]
                outmesh.faces.append(face)
                outmesh.face_corners += [(x,i) for x in face]

        elif chk.container == Chunk.Container.FACE_CORNERS and chk.name == "\"GEO::Mesh::facet_corners::corner_adjacent_facet\"":
            assert chk.n_data==1
            adj_corner = outmesh.face_corners.create_attribute("opposite_face", int, default_value=NOT_AN_ID)
            import_attribute(chk, adj_corner)

        elif chk.container == Chunk.Container.CELL_CORNERS and chk.name == "\"GEO::Mesh::cell_corners::corner_vertex\"":
            assert chk.n_data==1
            for i in range(container_sizes[Chunk.Container.CELLS]):
                ncc = n_corner_in_cell[i]
                ptr = cell_ptr[i]
                cell = [chk.data[ptr+_i] for _i in range(ncc)]
                outmesh.cells.append(cell)
                outmesh.cell_corners += [(x,i) for x in cell]

        elif chk.container == Chunk.Container.CELL_FACETS and chk.name == "\"GEO::Mesh::cell_facets::adjacent_cell\"":
            assert chk.n_data==1
            # chk.name = "adjacent_cell"
            adj_cell = outmesh.cell_faces.create_attribute("opposite_cell", int, default_value=NOT_AN_ID)
            adj_cell._expand(container_sizes[Chunk.Container.CELL_FACETS])
            import_attribute(chk, adj_cell)

        else: # user defined attribute
            container = {
                Chunk.Container.VERTICES : outmesh.vertices,
                Chunk.Container.EDGES : outmesh.edges,
                Chunk.Container.FACES : outmesh.faces,
                Chunk.Container.FACE_CORNERS : outmesh.face_corners,
                Chunk.Container.CELLS : outmesh.cells,
                Chunk.Container.CELL_CORNERS : outmesh.cell_corners,
                Chunk.Container.CELL_FACETS : outmesh.cell_faces,
            }.get(chk.container, None)
            if container is None:
                err_msg = f"In import_geogram_ascii : Container {chk.container} is not recognized"
                raise Exception(err_msg)
            attr = container.create_attribute(chk.name.split("\"")[1], chk.data_type, chk.n_data) # remove first and last "
            if chk.container == Chunk.Container.CELL_FACETS:
                attr._expand(container_sizes[Chunk.Container.CELL_FACETS])
            import_attribute(chk, attr)
    return outmesh

def export_attribute(f, size, container, attr, attr_name):
    f.write(f"[ATTR]\n\"{container}\"\n\"{attr_name}\"\n\"{attr.type.to_string()}\"\n{attr.type.byte_size()}\n{attr.elemsize}\n")
    for i in range(size):
        if attr.elemsize==1:
            if attr.type==Attribute.Type.Bool: # should be written as 0 or 1 and not as "true" or "false"
                f.write(f"{int(attr[i])}\n")
            else:
                f.write("{}\n".format(attr[i]))
        else:
            for j in range(attr.elemsize):
                if attr.type==Attribute.Type.Bool:
                    f.write(f"{int(attr[i][j])}\n")
                else:
                    f.write(f"{attr[i][j]}\n")

def export_geogram_ascii(mesh : RawMeshData, path):
    with open(path, "w", newline="\n") as f:
        f.write("[HEAD]\n\"GEOGRAM\"\n\"1.0\"\n")
        
        # Vertices
        n_vert = len(mesh.vertices)
        f.write("[ATTS]\n\"GEO::Mesh::vertices\"\n{}\n".format(n_vert))
        f.write("[ATTR]\n\"GEO::Mesh::vertices\"\n\"point\"\n\"double\"\n8\n3\n")
        for i in range(n_vert):
            f.write("{}\n{}\n{}\n".format(*mesh.vertices[i]))
        for attr_key in mesh.vertices.attributes:
            attr = mesh.vertices.get_attribute(attr_key)
            export_attribute(f, n_vert, "GEO::Mesh::vertices", attr, attr_key)
 
        # Edges
        if hasattr(mesh, "edges") and not mesh.edges.empty():
            n_edges = len(mesh.edges)
            f.write("[ATTS]\n\"GEO::Mesh::edges\"\n{}\n".format(n_edges))
            f.write("[ATTR]\n\"GEO::Mesh::edges\"\n\"GEO::Mesh::edges::edge_vertex\"\n\"index_t\"\n4\n2\n")
            for edge in mesh.edges:
                f.write(f"{edge[0]}\n{edge[1]}\n")
            for attr_key in mesh.edges.attributes:
                attr = mesh.edges.get_attribute(attr_key)
                export_attribute(f, n_edges, "GEO::Mesh::edges", attr, attr_key)

        # faces
        if hasattr(mesh, "faces") and not mesh.faces.empty():
            n_face = len(mesh.faces)
            f.write(f"[ATTS]\n\"GEO::Mesh::facets\"\n{n_face}\n")
            for attr_key in mesh.faces.attributes:
                attr = mesh.faces.get_attribute(attr_key)
                export_attribute(f, n_face, "GEO::Mesh::facets", attr, attr_key)

            # face_corners
            n_corners = len(mesh.face_corners)
            f.write("[ATTS]\n\"GEO::Mesh::facet_corners\"\n{}\n".format(n_corners))
            f.write("[ATTR]\n\"GEO::Mesh::facet_corners\"\n\"GEO::Mesh::facet_corners::corner_vertex\"\n\"index_t\"\n4\n1\n")
            for c in mesh.face_corners:
                f.write(f"{c}\n")

            if mesh.face_corners.has_attribute("corner_adjacent_facet"):
                f.write("[ATTR]\n\"GEO::Mesh::facet_corners\"\n\"GEO::Mesh::facet_corners::corner_adjacent_facet\"\n\"index_t\"\n4\n1\n")
                corner_adjacent_face = mesh.face_corners.get_attribute('corner_adjacent_facet')
                for i in range(n_corners):
                    f.write(f"{corner_adjacent_face[i]}\n")
                
            for attr_key in mesh.face_corners.attributes:
                if attr_key == "corner_adjacent_facet" : continue # already handled
                attr = mesh.face_corners.get_attribute(attr_key)
                export_attribute(f, n_corners, "GEO::Mesh::facet_corners", attr, attr_key)

        # Cells
        if hasattr(mesh, "cells") and not mesh.cells.empty():
            n_cells = len(mesh.cells)
            f.write("[ATTS]\n\"GEO::Mesh::cells\"\n{}\n".format(n_cells))
            for attr_key in mesh.cells.attributes:
                attr = mesh.cells.get_attribute(attr_key)
                export_attribute(f, n_cells, "GEO::Mesh::cells", attr, attr_key)

            # Cell Corners
            n_corners = sum([len(cell) for cell in mesh.cells])
            f.write("[ATTS]\n\"GEO::Mesh::cell_corners\"\n{}\n".format(n_corners))
            f.write("[ATTR]\n\"GEO::Mesh::cell_corners\"\n\"GEO::Mesh::cell_corners::corner_vertex\"\n\"index_t\"\n4\n1\n")
            for cell in mesh.cells:
                for x in cell:
                    f.write(f"{x}\n")
            for attr_key in mesh.cell_corners.attributes:
                attr = mesh.cell_corners.get_attribute(attr_key)
                export_attribute(f, n_corners, "GEO::Mesh::cell_corners", attr, attr_key)
                   
            # Cell faces
            n_cell_faces = sum([len(c) for c in mesh.cells])
            cell_adj = mesh.cell_faces.get_attribute("adjacent_cell")
            f.write("[ATTR]\n\"GEO::Mesh::cell_corners\"\n\"GEO::Mesh::cell_faces::adjacent_cell\"\n\"index_t\"\n4\n1\n")
            for x in cell_adj:
                f.write(f"{x}\n")

            for attr_key in mesh.cell_faces.attributes:
                if attr_key=="adjacent_cell" : continue
                attr = mesh.cell_faces.get_attribute(attr_key)
                export_attribute(f, n_cell_faces, "GEO::Mesh::cell_faces", attr, attr_key)