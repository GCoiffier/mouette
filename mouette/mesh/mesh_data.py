from .data_container import DataContainer, CornerDataContainer
from ..geometry import Vec
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
        
        First generates explicitely the set of faces from cells and the set of edges from faces. 
        This behavior can be controlled via the [`config.complete_faces_from_cells`][mouette.config.complete_faces_from_cells] and 
        the [`config.complete_edges_from_faces`][mouette.config.complete_edges_from_faces] global config variables.
        
        Then, treatments are applied on each DataContainer:

        - On vertices : casts 3D vectors to [`mouette.Vec`](mouette.geometry.vector.Vec)
        
        - On edges : sorts edge tuples to satisfy edge convention (smallest index first)
        
        - On faces : nothing

        - On face corners : generates the face_corners container if empty
        
        - On cells : nothing

        - On cell corners : generates the cell_corners container if empty

        """
        if self._prepared : return

        # call completion before preparation
        if config.complete_faces_from_cells:
            self._complete_faces_from_cells()

        # call edge completion ***after*** face completion
        if config.complete_edges_from_faces : 
            self._complete_edges_from_faces() 

        self._prepare_vertices()
        self._prepare_edges()
        self._prepare_faces()
        self._generate_face_corners()
        self._prepare_cells()
        self._generate_cell_corners()
        self._generate_cell_faces()
        self._compute_dimensionality()
        self._prepared = True

    def _prepare_vertices(self):
        for iv in self.id_vertices:
            self.vertices[iv] = Vec(self.vertices[iv])

    def _prepare_edges(self):
        N = len(self.vertices)

        def is_valid(a,b):
            return a!=b and 0<=a<N and 0<=b<N

        # Filter invalid edges (like (x,x)) and make the other immutable
        edges_invalid = any((not is_valid(a,b) for a,b in self.edges))
        if edges_invalid:
            # Rebuild the edge container
            new_edges = DataContainer(id="edges")
            new_attrs = dict()
            old_attrs = dict()
            for attr_name in self.edges.attributes:
                old_attrs[attr_name] = self.edges.get_attribute(attr_name)
                new_attrs[attr_name] = new_edges.create_attribute(attr_name, old_attrs[attr_name].type, old_attrs[attr_name].elemsize)
            n = 0
            for ie in self.id_edges:
                a,b = self.edges[ie]
                if is_valid(a,b):
                    new_edges.append(utils.keyify(a,b))
                    for name in new_attrs:
                        if ie in old_attrs[name]: # keep sparsity of the attribute
                            new_attrs[name][n] = old_attrs[name][ie]
                    n+=1
            self.edges = new_edges
        else:
            for ie in self.id_edges:
                self.edges[ie] = utils.keyify(self.edges[ie])

    def _prepare_faces(self):
        pass

    def _generate_face_corners(self):
        nc = len(self.face_corners)
        nf = sum([len(f) for f in self.faces])
        if nc == 0 or nc!= nf :
            # corners were not or badly generated
            self.face_corners._elem = []
            self.face_corners._adj = []
            for iF,F in enumerate(self.faces):
                for v in F:
                    self.face_corners.append(v,iF)

    def _prepare_cells(self):
        pass

    def _generate_cell_corners(self):
        nce = len(self.cell_corners._elem)
        nca = len(self.cell_corners._adj)
        if nce==0 or nca==0:
            # corners were not or badly generated
            if nca==0 and nce>0:
                # build only adjacency
                self.cell_corners._adj = []
                for iC,C in enumerate(self.cells):
                    self.cell_corners._elem += [iC]*len(C)
            else:
                # build both containers
                self.cell_corners._elem = []
                self.cell_corners._adj = []
                for iC,C in enumerate(self.cells):
                    for v in C:
                        self.cell_corners.append(v,iC)

    def _generate_cell_faces(self):
        nce = len(self.cell_faces._elem)
        nca = len(self.cell_faces._adj)
        if nca==0 or nce==0:
            # cell faces were not generated completely

            face_id = dict() # first invert face indirection
            for iF,F in enumerate(self.faces):
                key = utils.keyify(F)
                face_id[key] = iF

            for iC,C in enumerate(self.cells):
                if len(C)==4:
                    # cell is tetrahedron
                    v0,v1,v2,v3 = C
                    # convention: face fi does not contain vertex vi
                    faces_C = [(v1,v3,v2), (v0,v2,v3), (v3,v1,v0), (v0,v1,v2)]
                elif len(C)==8:
                    # cell is hexahedron : TODO
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
                    self.cell_faces._elem.append(face_id[utils.keyify(face)])
                    if nca!=0: 
                        self.cell_faces._adj.append(iC)

    def _complete_edges_from_faces(self):
        if self.faces.empty() : return # nothing to do
        
        # Create hard edges attribute for already existing edges
        hard_edges = self.edges.create_attribute("hard_edges", bool)
        for e in self.id_edges: hard_edges[e] = True
        
        edge_set = set([utils.keyify(e) for e in self.edges])
        for f in self.faces:
            nf = len(f)
            for i in range(nf):
                edge = utils.keyify(f[i], f[(i+1)%nf])
                if edge not in edge_set:
                    edge_set.add(edge)
                    self.edges.append(edge)

    def _complete_faces_from_cells(self):
        if self.cells.empty() : return # nothing to do
        face_set = set([utils.keyify(f) for f in self.faces])
        for C in self.cells:
            faces_C = []
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
            
            for face in faces_C:
                face_key = utils.keyify(face)
                if face_key not in face_set:
                    face_set.add(face_key)
                    self.faces.append(face)
