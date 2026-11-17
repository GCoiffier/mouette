import numpy as np
import warnings
from ..data_container import DataContainer, CornerDataContainer

from .base import Mesh
from .pointcloud import PointCloud
from .linear import PolyLine
from .surface import SurfaceMesh
from .volume import VolumeMesh

from ... import utils
from ... import config

def _is_empty(container):
    if container is None:
        return True
    elif isinstance(container, list) or isinstance(container, dict):
        return len(container)==0
    elif isinstance(container, np.ndarray):
        return container.size==0
    raise Exception("invalid container type")

class RawMeshData:
    """
    A base container class to store all the data relative to a mesh.
    Serves an intermediate between io parsers or manual inputs and instantiated (and typed) mesh classes.
    
    Contrary to other mesh classes, RawMeshData have editable containers to build a mesh incrementally. A Mesh object can then be built correctly by using the `instantiate` method
    """

    def __init__(self):
        self.vertices = []
        self.vertices_attr = dict()

        self.edges = []
        self.edges_attr = dict()

        self.faces = []
        self.faces_attr = dict()

        self.face_corners = []
        self.face_corners_attr = dict()

        self.cells = []   
        self.cells_attr = dict()
        
        self.cell_corners = []
        self.cell_corners_attr = dict()

        self.cell_faces = []
        self.cell_faces_attr = dict()
    
    @property
    def id_vertices(self): return range(len(self.vertices))

    @property
    def id_edges(self): return range(len(self.edges))

    @property
    def id_faces(self): return range(len(self.faces))

    @property
    def id_cells(self): return range(len(self.cells))

    @property
    def id_facecorners(self): return range(len(self.face_corners))

    @property
    def id_cellcorners(self): return range(len(self.cell_corners))

    @property
    def id_cellfaces(self): return range(len(self.cell_faces))

    def instantiate(self, dim:int=None) -> Mesh:
        """
        Prepares the data into datacontainer and returns a Mesh object that wraps these containers.  
        Applies various data generation before returning, like computing edges from faces or faces from cells. This behavior can be controlled via the `config.complete_faces_from_cells` and the `config.complete_edges_from_faces` global config variables.
        
        Args:
            dim (int, optional): dimensionality of the instantiated mesh. 0=PointCloud, 1=PolyLine, 2=SurfaceMesh, 3=VolumeMesh. If not provided, will infer the dimensionality from the data. Setting a low dimensionality may result in a loss of data. Defaults to None.
        """

        if config.complete_faces_from_cells:
            self._complete_faces_from_cells()

        # call edge completion ***after*** face completion
        if config.complete_edges_from_faces : 
            self._complete_edges_from_faces()

        def register_attributes(container, attr_dict):
            for attr_name, attr_data in attr_dict.items():
                if isinstance(attr_data, np.ndarray) or isinstance(attr_data, list):
                    if _is_empty(attr_data): continue
                    container.register_array_as_attribute(attr_name, np.ascontiguousarray(attr_data))
                elif isinstance(attr_data, dict):
                    if _is_empty(attr_data): continue
                    container.register_dict_as_sparse_attribute(attr_name, attr_data)
                else:
                    warnings.warn(f"attribute name '{attr_name}' in RawMeshData is associated with an object of type {type(attr_data)}. Only numpy arrays, lists and dict are supported. This attribute will be ignored.")
        
        edges, faces, face_corners, cells, cell_corners, cell_faces = None, None, None, None, None, None

        ### Vertices
        vertices = DataContainer(np.ascontiguousarray(self.vertices, dtype=np.float64), self.vertices_attr, id="vertices")
        register_attributes(vertices, self.vertices_attr)
        data_dim = 0
        
        ### Edges
        if not _is_empty(self.edges):
            edges = DataContainer(np.ascontiguousarray(self.edges, dtype=np.int64), self.edges_attr, id="edges")
            register_attributes(edges, self.edges_attr)
            data_dim = 1

        ### Faces
        if not _is_empty(self.faces):
            faces = DataContainer(np.ascontiguousarray(self.faces, dtype=np.int64), self.faces_attr, id="faces")
            register_attributes(faces, self.faces_attr)
            face_corners = self._generate_face_corners(faces._data)
            register_attributes(face_corners, self.face_corners_attr)
            data_dim = 2

        ### Cells
        if not _is_empty(self.cells):
            cells = DataContainer(np.ascontiguousarray(self.cells, dtype=np.int64), self.cells_attr, id="cells")
            register_attributes(cells, self.cells_attr)
            cell_corners = self._generate_cell_corners(cells._data)
            register_attributes(cell_corners, self.cell_corners_attr)
            cell_faces = self._generate_cell_faces()
            register_attributes(cell_faces, self.cell_faces_attr)
            data_dim = 3

        dim = data_dim if dim is None else dim
        match dim:
            case 3: # VolumeMesh
                return VolumeMesh(vertices, edges, faces, face_corners, cells, cell_corners, cell_faces)
            case 2: #SurfaceMesh
                return SurfaceMesh(vertices, edges, faces, face_corners)
            case 1: #Polyline
                return PolyLine(vertices, edges)
            case 0:
                return PointCloud(vertices)
            case _:
                raise Exception("Dimensionality of raw mesh data is not an integer between 0 and 3. This should not happen.")


    def _complete_edges_from_faces(self):
        if _is_empty(self.faces): return # no faces => nothing to do
        
        # Create hard edges attribute for already existing edges
        if not _is_empty(self.edges):
            self.edges_attr["hard_edges"] = dict()
            for e in range(len(self.edges)):
                self.edges_attr["hard_edges"][e] = True

        new_edges = []
        edge_set = set([utils.keyify(e) for e in self.edges])
        for f in self.faces:
            nf = len(f)
            for i in range(nf):
                edge = utils.keyify(f[i], f[(i+1)%nf])
                if edge not in edge_set:
                    edge_set.add(edge)
                    new_edges.append(edge)
        if not new_edges: return
        if isinstance(self.edges, list):
            self.edges += new_edges
        elif isinstance(self.edges, np.ndarray):
            new_edges = np.array(new_edges, dtype=np.int64)
            self.edges = new_edges if _is_empty(self.edges) else np.concatenate((self.edges, new_edges))

    def _complete_faces_from_cells(self):
        if _is_empty(self.cells) : return # no cells => nothing to do
        face_set = set([utils.keyify(*f) for f in self.faces])
        new_faces = []
        for C in self.cells:
            if len(C)==8:
                # hexahedron
                v1,v2,v3,v4,v5,v6,v7,v8 = C
                faces_C = [
                    (v1,v2,v3,v4),
                    (v5,v6,v7,v8),
                    (v1,v4,v8,v5),
                    (v1,v2,v6,v5),
                    (v2,v3,v7,v6),
                    (v3,v4,v8,v7)
                ]
            elif len(C)==4:
                # tetrahedron
                v0,v1,v2,v3 = C
                # convention: face fi does not contain vertex vi
                faces_C = [
                    (v1,v3,v2), 
                    (v0,v2,v3), 
                    (v3,v1,v0), 
                    (v0,v1,v2)
                ]
            else:
                raise Exception(f"Unrecognized cell format with {len(C)} vertices.")
            for face in faces_C:
                face_key = utils.keyify(face)
                if face_key not in face_set:
                    face_set.add(face_key)
                    new_faces.append(face)
        if not new_faces : return
        elif isinstance(self.faces, list): 
            self.faces += new_faces
        elif isinstance(self.faces, np.ndarray):
            new_faces = np.array(new_faces, dtype=np.int64)
            self.faces = new_faces if _is_empty(self.faces) else np.concatenate((self.faces, new_faces))

    def _generate_face_corners(self, F:np.ndarray):
        if not _is_empty(self.face_corners): # already generated : trust the values
            corners_elem = np.array([x[0] for x in self.face_corners])
            corners_adj = np.array([x[1] for x in self.face_corners])
        else:
            corners_elem = F.copy().ravel()
            corners_adj = np.repeat(np.arange(F.shape[0]), F.shape[1])
        return CornerDataContainer(corners_elem, corners_adj, id="face_corners")


    def _generate_cell_corners(self, C:np.ndarray):
        if not _is_empty(self.cell_corners): # already generated : trust the values
            corners_elem = np.array([x[0] for x in self.cell_corners])
            corners_adj = np.array([x[1] for x in self.cell_corners])
        else:
            corners_elem = C.copy().ravel()
            corners_adj = np.repeat(np.arange(C.shape[0]), C.shape[1])
        return CornerDataContainer(corners_elem, corners_adj, id="cell_corners")


    def _generate_cell_faces(self):
        if not _is_empty(self.cell_faces): 
            elem = np.array([x[0] for x in self.cell_faces])
            adj = np.array([x[1] for x in self.cell_faces])
        else:
            face_id = dict() # map a face tuple to a face id
            for iF,F in enumerate(self.faces):
                key = utils.keyify(F)
                face_id[key] = iF

            elem, adj = [], []
            for iC,C in enumerate(self.cells):
                if len(C)==4:
                    # cell is tetrahedron
                    v0,v1,v2,v3 = C
                    # convention: face fi does not contain vertex vi
                    faces_C = [(v1,v3,v2), (v0,v2,v3), (v3,v1,v0), (v0,v1,v2)]
                elif len(C)==8:
                    v1,v2,v3,v4,v5,v6,v7,v8 = C
                    faces_C = [
                        (v1,v2,v3,v4),
                        (v5,v6,v7,v8),
                        (v1,v4,v8,v5),
                        (v1,v2,v6,v5),
                        (v2,v3,v7,v6),
                        (v3,v4,v8,v7)
                    ]
                for face in faces_C:
                    elem.append(face_id[utils.keyify(face)])
                    adj.append(iC)
            elem = np.array(elem, dtype=np.int64)
            adj = np.array(adj, dtype=np.int64)
        return CornerDataContainer(elem, adj, id="cell_faces")

