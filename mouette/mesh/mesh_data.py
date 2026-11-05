import numpy as np
from .data_container import DataContainer, CornerDataContainer
from .. import utils
from .. import config

class RawMeshData:
    """
    A base container class to store all the data relative to a mesh.
    Serves an intermediate between io parsers or manual inputs and instantiated (and typed) mesh classes.
    """

    def __init__(self, mesh=None):
        self.vertices = DataContainer(id="vertices") if mesh is None else mesh.vertices
        self.edges = DataContainer(id="edges") if (mesh is None or not hasattr(mesh,"edges")) else mesh.edges

        self.faces = DataContainer(id="faces") if (mesh is None or not hasattr(mesh,"faces")) else mesh.faces
        self.face_corners = CornerDataContainer(id="face_corners") if (mesh is None or not hasattr(mesh,"face_corners")) else mesh.face_corners
       
        self.cells = DataContainer(id="cells") if (mesh is None or not hasattr(mesh,"cells")) else mesh.cells
        self.cell_corners = CornerDataContainer(id="cell_corners") if (mesh is None or not hasattr(mesh,"cell_corners")) else mesh.cell_corners
        self.cell_faces = CornerDataContainer(id="cell_faces") if (mesh is None or not hasattr(mesh,"cell_faces")) else mesh.cell_faces

        self._dimensionality : int = None
        self._prepared : bool = False
    
    @property
    def id_vertices(self):
        """
        Shortcut for `range(len(self.vertices))`
        """
        return range(len(self.vertices))

    @property
    def id_edges(self):
        """
        Shortcut for `range(len(self.edges))`
        """
        return range(len(self.edges))

    @property
    def id_faces(self):
        """
        Shortcut for `range(len(self.faces))`
        """
        return range(len(self.faces))

    @property
    def id_cells(self):
        """
        Shortcut for `range(len(self.cells))`
        """
        return range(len(self.cells))

    @property
    def id_facecorners(self):
        """
        Shortcut for `range(len(self.face_corners))`
        """
        return range(len(self.face_corners))

    @property
    def id_cellcorners(self):
        """
        Shortcut for `range(len(self.cell_corners))`
        """
        return range(len(self.cell_corners))

    @property
    def dimensionality(self) -> int:
        """
        Dimensionality of the manifold represented by the data.

        - If only vertices -> 0

        - If vertices and edges (embedded graph) -> 1

        - If faces (surface manifold) -> 2

        - If cells (volume manifold) -> 3

        Returns:
            int: 0,1,2 or 3
        """
        if self._dimensionality is None:
            self._compute_dimensionality()
        return self._dimensionality

    def _compute_dimensionality(self):
        if not self.cells.empty():
            self._dimensionality = 3
        elif not self.faces.empty():
            self._dimensionality = 2
        elif not self.edges.empty():
            self._dimensionality = 1
        else:
            self._dimensionality = 0

    def prepare(self) -> None:
        """
        Prepares the data to have the correct format. This method is called by the constructors of data structures.
        
        First generates explicitly the set of faces from cells and the set of edges from faces. 
        This behavior can be controlled via the `config.complete_faces_from_cells` and 
        the `config.complete_edges_from_faces` global config variables.
        
        Then, corner data containers are generated if needed.
        """
        if self._prepared : return

        # call completion before preparation
        if config.complete_faces_from_cells:
            self._complete_faces_from_cells()

        # call edge completion ***after*** face completion
        if config.complete_edges_from_faces : 
            self._complete_edges_from_faces() 

        self._compute_dimensionality()
        self._generate_face_corners()
        self._generate_cell_corners()
        self._generate_cell_faces()
        self._prepared = True

    def _generate_face_corners(self):
        if self.faces.empty(): return
        if not self.face_corners.empty(): return
        elem, adj = [], []
        for iF,F in enumerate(self.faces):
            for v in F:
                elem.append(v)
                adj.append(iF)
        self.face_corners.set_data(elem, adj)

    def _generate_cell_corners(self):
        if self.cells.empty(): return
        if not self.cell_corners.empty(): return
        elem, adj = [], []
        for iC,C in enumerate(self.cells):
            for v in C:
                elem.append(v)
                adj.append(iC)
        self.cell_corners.set_data(elem, adj)


    def _generate_cell_faces(self):
        if self.cells.empty(): return
        if not self.cell_faces.empty(): return

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
        self.cell_faces.set_data(elem, adj)


    def _complete_edges_from_faces(self):
        if self.faces.empty() : return # nothing to do
        
        # Create hard edges attribute for already existing edges
        if not self.edges.empty():
            hard_edges = self.edges.create_attribute("hard_edges", bool)
            for e in self.id_edges: hard_edges[e] = True

        new_edges = []
        edge_set = set([utils.keyify(e) for e in self.edges])
        for f in self.faces:
            nf = len(f)
            for i in range(nf):
                edge = utils.keyify(f[i], f[(i+1)%nf])
                if edge not in edge_set:
                    edge_set.add(edge)
                    new_edges.append(edge)
        self.edges.add_data(new_edges)

    def _complete_faces_from_cells(self):
        if self.cells.empty() : return # nothing to do
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
        self.faces.add_data(new_faces)

