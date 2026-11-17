from ..data_container import DataContainer, CornerDataContainer
from .base import Mesh
from .linear import PolyLine
from .surface import SurfaceMesh
from ... import geometry as geom
from ... import config
from ... import utils

import numpy as np

class VolumeMesh(Mesh):

    def __init__(self,
        vertices : DataContainer,
        edges : DataContainer,
        faces : DataContainer,
        face_corners : CornerDataContainer,
        cells : DataContainer,
        cell_corners : CornerDataContainer,
        cell_faces : CornerDataContainer
    ):
        vertices = vertices or DataContainer(id="vertices")
        edges = edges or DataContainer(id="edges")
        faces = faces or DataContainer(id="faces")
        face_corners = face_corners or CornerDataContainer(id="face_corners")
        cells = cells or DataContainer(id="cells")
        cell_corners = cell_corners or CornerDataContainer(id="cell_corners")
        cell_faces = cell_faces or CornerDataContainer(id="cell_faces")

        Mesh.__init__(self, 
            vertices=vertices, 
            edges=edges, 
            faces=faces, 
            face_corners=face_corners, 
            cells=cells, 
            cell_corners=cell_corners, 
            cell_faces=cell_faces)
        
        self.connectivity = VolumeMesh._Connectivity(self)

        # Boundary data
        self._faces_on_border : np.ndarray[bool] = None
        self._edge_on_border : np.ndarray[bool] = None
        self._vertex_on_border : np.ndarray[bool] = None

        self.boundary_connectivity : VolumeMesh._BoundaryConnectivity = None

    def enable_boundary_connectivity(self):
        """
        Builds the indirection maps to the boundary surface mesh, which then can be accessed via the `self.boundary_connectivity` attribute.

        See documentation for VolumeMesh._BoundaryConnectivity for more details.
        """
        self.boundary_connectivity = VolumeMesh._BoundaryConnectivity(self)

    def __str__(self):
        out = "mouette.mesh.VolumeMesh object\n"
        out += "| {} vertices\n".format(len(self.vertices))
        out += "| {} edges\n".format(len(self.edges))
        out += "| {} faces\n".format(len(self.faces))
        out += "| {} cells\n".format(len(self.cells))
        return out

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
    def id_corners(self):
        """
        Shortcut for `range(len(self.face_corners))`
        """
        return range(len(self.face_corners))

    @property
    def boundary_mesh(self):
        """The boundary of the VolumeMesh as a SurfaceMesh. Shortcut for `self.boundary_connectivity.mesh
        
        Returns:
            None: if the boundary connectivity is not enabled (mesh was not built). See `enable_connectivity`
            SurfaceMesh: the boundary mesh otherwise

        """
        try:
            return self.boundary_connectivity.mesh
        except Exception as _:
            return None

    def _compute_interior_boundary_faces(self):
        self._faces_on_border = np.empty(len(self.faces), dtype=bool)
        for F in self.id_faces:
            self._faces_on_border[F] = self.is_face_on_border(F)

    def _compute_interior_boundary_vertices(self):
        self._vertex_on_border = np.zeros(len(self.vertices), dtype=bool)
        for iF in self.boundary_faces:
            self._vertex_on_border[self.faces[iF]] = True

    def _compute_interior_boundary_edges(self):
        self._edge_on_border = np.zeros(len(self.edges), dtype=bool)
        for iF in self.boundary_faces:
            for e in self.connectivity.face_to_edges(iF):
                self._edge_on_border[e] = True

    def is_face_on_border(self, *args) -> bool:
        """Simple test to determine if a given face is on the boundary of the mesh.
        
        Parameters:
            *args: Either an integer index representing a face, or n integer indices representing the vertices
            
        Returns:
            bool: Returns True is the given face exists and is on the boundary of the mesh
        """
        if len(args)==1:
            n = len(self.connectivity.face_to_cellfaces(args[0]))
        else:
            n = len(self.connectivity.face_to_cellfaces(self.connectivity.face_id(*args)))
        return n<2
    
    def is_vertex_on_border(self, u) -> bool:
        """
        whether vertex `u` is a boundary vertex or not.

        Args:
            u (int): vertex id

        Returns:
            bool: whether vertex `u` is a boundary vertex or not.
        """
        if self._is_vertex_on_border is None:
            self._compute_interior_boundary_vertices()
        return self._is_vertex_on_border[u]
    
    def is_edge_on_border(self, *args) -> bool:
        """Simple test to determine if a given edge is on the boundary of the mesh.

        Parameters:
            *args: Either an integer index representing an edge, or two integer indices representing two (adjacent) vertices

        Returns:
            bool: Returns True if the given edge is on the boundary of the mesh.
        """
        if self._edge_on_border is None:
            self._compute_interior_boundary_edges()
        if len(args)==1:
            return self._edge_on_border[args[0]]
        return self._edge_on_border[self.connectivity.edge_id(args[0],args[1])]

    def is_tetrahedral(self) -> bool:
        """
        Returns:
            bool: True if the mesh is tetrahedral (all cells are tetrahedra)
        """
        return self.cells._data.shape[1]==4


    @property
    def boundary_faces(self) -> np.ndarray:
        """Array of indices of all faces that are on the boundary
        """
        if self._faces_on_border is None:
            self._compute_interior_boundary_faces() 
        return np.arange(len(self.faces))[self._faces_on_border]

    @property
    def interior_faces(self) -> np.ndarray:
        """Array of indices of all faces that are **not** on the boundary
        """
        if self._faces_on_border is None :
            self._compute_interior_boundary_faces() 
        return np.arange(len(self.faces))[~self._faces_on_border]

    @property
    def interior_edges(self) -> np.ndarray:
        """Array of indices of all edges that are **not** on the boundary
        """
        if self._edge_on_border is None:
            self._compute_interior_boundary_edges()
        return np.arange(len(self.edges))[~self._edge_on_border]

    @property
    def boundary_edges(self) -> np.ndarray:
        """Array of indices of all edges that are on the boundary
        """
        if self._edge_on_border is None:
            self._compute_interior_boundary_edges()
        return np.arange(len(self.edges))[self._edge_on_border]

    @property
    def boundary_vertices(self) -> np.ndarray:
        """Array of indices of all vertices that are on the boundary
        """
        if self._vertex_on_border is None:
            self._compute_interior_boundary_vertices()
        return np.arange(len(self.vertices))[self._vertex_on_border]
    
    @property
    def interior_vertices(self) -> np.ndarray:
        """Array of indices of all vertices that are **not** on the boundary
        """
        if self._vertex_on_border is None:
            self._compute_interior_boundary_vertices()
        return np.arange(len(self.vertices))[~self._vertex_on_border]



    class _Connectivity(PolyLine._Connectivity): 

        def __init__(self, master):
            super().__init__(master)
            self._face_id : dict = None
            self._cell_id : dict = None
            #self._edge_id is inherited
            
            self._adjC2C : list = None # cell -> cell

            self._adjV2C : list = None # vertex -> cells
            self._adjV2F : list = None # vertex -> faces
            # self._adjV2V is inherited

            self._adjE2F : list = None # edge -> faces 
            self._adjE2C : list = None # edge -> cells

            self._adjF2Cfn : list = None # face -> cell facets

        
        def clear(self):
            """
            Resets connectivity. 
            The next query in the code will regenerate internal arrays.
            """
            super().clear()
            self._face_id : dict = None
            self._cell_id : dict = None            
            self._adjC2C : list = None
            self._adjV2C : list = None
            self._adjV2F : list = None
            self._adjE2F : list = None
            self._adjE2C : list = None
            self._adjF2Cfn : list = None

        ##### Ids #####

        def _compute_cell_ids(self):
            self._cell_id = dict()
            for iC, C in enumerate(self.mesh.cells):
                self._cell_id[utils.keyify(C)] = iC

        def cell_id(self, *args) -> int|None:
            """The identifier of a cell.

            Args:
                int*: integers representing indices of the vertices of the cell (not necessarily in the correct order)

            Returns:
                int|None: A cell index or None if the given indices do not correspond to a valid cell
            """
            if self._cell_id is None:
                self._compute_cell_ids()
            key = utils.keyify(*args)
            return self._cell_id.get(key, None)
        
        def _compute_face_ids(self):
            self._face_id = dict()
            for iF, F in enumerate(self.mesh.faces):
                self._face_id[utils.keyify(F)] = iF

        def face_id(self, *args) -> int|None:
            """ The identifier of a face.
            
            Args:
                int*: integers representing indices of vertices of the face (not necessarily in the correct order)

            Returns:
                int: A face index or None if the given indices do not correspond to a valid face
            """
            if self._face_id is None:
                self._compute_face_ids()
            key = utils.keyify(*args)
            return self._face_id.get(key, None)

        # inherits edge_id from Polyline._Connectivity

        ##### Connectivity #####
        def _compute_vertex_adj(self):
            self._adjV2C = [[] for _ in self.mesh.id_vertices]
            self._adjV2F = [[] for _ in self.mesh.id_vertices]
            for iC,C in enumerate(self.mesh.cells):
                for V in C: self._adjV2C[V].append(iC)
            for iF,F in enumerate(self.mesh.cells):
                for V in F: self._adjV2F[V].append(iF)

        def _compute_face_adj(self):
            self._adjF2Cfn = [[] for _ in self.mesh.id_faces]
            for iCn,F in enumerate(self.mesh.cell_faces):
                self._adjF2Cfn[F].append(iCn)

        def _compute_cell_adj(self):
            self._adjC2C = [[] for _ in self.mesh.id_cells]
            for iF in self.mesh.id_faces:
                cell_faces_iF = self.face_to_cellfaces(iF)
                if len(cell_faces_iF)!=2: continue
                cf1,cf2 = cell_faces_iF
                adjcf1, adjcf2 = self.mesh.cell_faces.adj(cf1), self.mesh.cell_faces.adj(cf2)
                self._adjC2C[adjcf1].append(adjcf2)
                self._adjC2C[adjcf2].append(adjcf1)

        def _compute_edge_adj(self):
            if self._edge_id is None:
                self._compute_edge_id()
            self._adjE2F = [[] for _ in self.mesh.id_edges]
            self._adjE2C = [[] for _ in self.mesh.id_edges]

            if config.sort_neighborhoods:
                self._sort_edge_neighborhoods()

        def _sort_edge_neighborhoods(self):
            """
            Sorts cells adjacent to an edge so that they are in direct order
            """
            if not self.mesh.is_tetrahedral(): return # TODO
            for e,(A,B) in enumerate(self.mesh.edges):
                keys_cell = dict()
                keys_face = dict()
                kc = 0
                kf = 0
                iC = self._adjE2C[e][0]
                keys_cell[iC] = kc
                p1,p2 = (x for x in self.mesh.cells[iC] if x not in (A,B)) # pivots
                while True:
                    face = self.face_id(A,B,p1)
                    kf += 1
                    keys_face[face] = kf
                    nextC = self.other_face_side(iC,face)
                    if nextC is None or nextC in keys_cell: break # we have gone full circle
                    kc += 1
                    keys_cell[nextC] = kc
                    iC = nextC
                    p1 = [x for x in self.mesh.cells[iC] if x not in (A,B,p1)][0]
                
                kc = 0
                kf = 0
                iC = self._adjE2C[e][0]
                while True:
                    face = self.face_id(A,B,p2)
                    kf -= 1
                    keys_face[face] = kf
                    nextC = self.other_face_side(iC,face)
                    if nextC is None or nextC in keys_cell: break # we have gone full circle
                    kc -= 1
                    keys_cell[nextC] = kc
                    iC = nextC
                    p2 = [x for x in self.mesh.cells[iC] if x not in (A,B,p2)][0]
                self._adjE2C[e].sort(key= lambda c : keys_cell[c])
                self._adjE2F[e].sort(key= lambda f : keys_face[f])


        ##### Cells to elements #####

        def cell_to_cells(self, iC:int) -> list:
            """
            Lists all tje cells that share a common face with face `iC`.

            Args:
                iC (int): cell id

            Returns:
                list: list of adjacent cells.
            """
            if self._adjC2C is None:
                self._compute_cell_adj()
            return [int(x) for x in self._adjC2C[iC,:] if x!=-1]

        def cell_to_faces(self, iC:int) -> list:
            """
            Lists all the faces of cell `iC`.

            Args:
                iC (int): cell id

            Returns:
                list: face ids
            """
            cell_size = 4 if self.mesh.is_tetrahedral() else 6
            return [self.mesh.cell_faces[c] for c in range(iC*cell_size, (iC+1)*cell_size)]

        def cell_to_edges(self, iC):
            raise NotImplementedError


        ##### Face to elements #####
        def face_to_cells(self, iF: int) -> list:
            """Returns the list of cells that are on either side of face `iF` (length 1 if F is a boundary face and 2 otherwise)

            Args:
                iF (int): face id

            Returns:
                list: cell ids
            """
            if self._adjF2Cfn is None:
                self._compute_cell_adj()
            return [int(self.mesh.cell_faces.adj(x)) for x in self._adjF2Cfn[iF]]

        def face_to_cellfaces(self, iF: int) -> list:
            """Returns the list of cell_faces the face `iF` is associated to (length 1 if F is a boundary face and 2 otherwise)

            Args:
                iF (int): face id

            Returns:
                list: cell_faces ids
            """
            if self._adjF2Cfn is None:
                self._compute_face_adj()
            return self._adjF2Cfn[iF]
        
        def face_to_edges(self, iF: int) -> list:
            """Lists the edges adjacent to face `iF`.

            Args:
                iF (int): cell id

            Returns:
                list: edge ids
            """
            F = self.mesh.faces[iF].tolist()
            return [self.edge_id(A,B) for A,B in utils.cyclic_pairs(F)]

        ##### Edge to elements #####
        def edge_to_cells(self, iE: int) -> list:
            """
            Lists the cells that are adjacent to an edge `iE`.  
            If `config.sort_neighborhoods` is set to True, the list is in consecutive order, i.e. consecutive cells are adjacent.

            Args:
                iE (int): edge id

            Returns:
                list: cell ids
            """
            if self._adjE2C is None:
                self._compute_edge_adj()
            return self._adjE2C[iE]

        def edge_to_faces(self, iE: int) -> list:
            """Lists the faces that are adjacent to an edge `iE`.
            If `config.sort_neighborhoods` is set to True, the list is in consecutive order consistent with the cell ordering (see `edges_to_cells`).

            Args:
                iE (int): edge id

            Returns:
                list: list of face ids
            """
            if self._adjE2F is None:
                self._compute_edge_adj()
            return self._adjE2F[iE]


        ##### Vertex to elements #####
        def vertex_to_cells(self, iV: int) -> list:
            """
            Neighborhood of vertex `V` in terms of cells.

            Args:
                V (int): vertex id

            Returns:
                list: list of cells C such that V belongs to C.
            """
            if self._adjV2C is None:
                self._compute_vertex_adj()
            return self._adjV2C[iV]

        def vertex_to_faces(self, iV: int) -> list:
            """
            Neighborhood of vertex `V` in terms of faces.

            Args:
                V (int): vertex id

            Returns:
                list: list of faces F such that V belongs to F.
            """
            if self._adjV2F is None:
                self._compute_vertex_adj()
            return self._adjV2F[iV]

        # vertex_to_edges is inherited

        ##### Misc #####

        def other_face_side(self, C: int, F: int) -> int|None:
            """Given a cell C and a face F in C, returns the index of the cell C' that is on the other side of F.
            Returns None if C' does not exist (F is a boundary face) or if F is not a valid face of C.

            Args:
                C (int): cell id
                F (int): face id

            Returns:
                int|None: cell id
            """
            F2C = self.face_to_cells(F)
            if len(F2C) != 2 : return None
            if C==F2C[0]: return F2C[1]
            elif C==F2C[1]: return F2C[0]
            return None
        
        def common_face(self, C1: int, C2: int) -> int|None:
            """Given two cells C1 and C2, returns the index of their common face, or None if they do not share any face.

            Args:
                C1 (int): cell id
                C2 (int): cell id

            Returns:
                int|None: face id
            """
            common = np.intersect1d(self.cells[C1], self.cells[C2])
            if len(common) != self.cells._data.shape[1]: return None
            return self.face_id(*common)


        def in_cell_index(self, C: int, V: int) -> int:
            """Index of vertex V in cell C. None if V is not in cell C

            Parameters:
                C (int): cell id
                V (int): vertex id

            Returns:
                int: index of V in C (in terms of cell corners).
            """
            if self.mesh.is_tetrahedral():
                i_start, i_len = 4*C, 4
            else:
                i_start, i_len = 8*C, 8
            for i in range(i_len):
                if self.mesh.cell_corners[i_start+i] == V:
                    return i
            return None

        def in_cell_face_index(self,C: int, F: int) -> int:
            """Index of face F in cell C. None if F is not a valid face of C.

            Args:
                C (int): cell id
                F (int): face id

            Returns:
                int: index of F in C (in terms of cell faces)
            """
            if self.mesh.is_tetrahedral():
                i_start, i_len = 4*C, 4
            else:
                i_start, i_len = 6*C, 6
            for i in range(i_len):
                if self.mesh.cell_faces[i_start+i] == F:
                    return i
            return None


      
    class _BoundaryConnectivity(SurfaceMesh._Connectivity):

        def __init__(self, master):
            """
            Special version of the SurfaceMesh Connectivity object, where everything goes through the indirection boundary id -> complete mesh id.

            Attributes:
                m2b_vertex (dict): volume mesh -> boundary indirection for vertices
                m2b_edge (dict): volume mesh -> boundary indirection for edges
                m2b_face (dict): volume mesh -> boundary indirection for faces

                b2m_vertex (dict): boundary -> volume mesh indirection for vertices
                b2m_edge (dict): boundary -> volume mesh indirection for edges
                b2m_face (dict): boundary -> volume mesh indirection for faces

            """
            self.complete_mesh = master

            # indirection maps between boundary elements and their indices in the full mesh
            self.m2b_vertex : dict = None # mesh to boundary
            self.m2b_edge : dict = None
            self.m2b_face : dict = None

            self.b2m_vertex : dict = None # boundary to mesh
            self.b2m_edge : dict = None
            self.b2m_face : dict = None

            mesh = self._extract_surface_boundary() # builds indirection maps for vertices and faces
            super().__init__(mesh) # the boundary mesh can be accessed using the .mesh attribute
                    

        def _extract_surface_boundary(self):
            """
            Builds indirection maps. This is performed automatically upon first connectivity request.
            """
            from .raw import RawMeshData
            boundary = RawMeshData()
            # indirection maps
            self.m2b_vertex, self.b2m_vertex = dict(), dict()
            self.m2b_edge, self.b2m_edge = dict(), dict()
            self.m2b_face, self.b2m_face = dict(), dict()
            vertex_set = set()
            bnd_faces = []
            for i,iF in enumerate(self.complete_mesh.boundary_faces):
                self.m2b_face[iF] = i
                self.b2m_face[i] = iF
                bnd_faces.append(iF)
                for v in self.complete_mesh.faces[iF]:
                    vertex_set.add(v)

            # re order vertices
            for i,v in enumerate(vertex_set):
                boundary.vertices.append(self.complete_mesh.vertices[v])
                self.m2b_vertex[v] = i
                self.b2m_vertex[i] = v

            # generate set of faces and check for orientation
            for iF in bnd_faces:
                iC = self.complete_mesh.connectivity.face_to_cells(iF)[0]
                pA,pB,pC = (self.complete_mesh.vertices[_x] for _x in self.complete_mesh.faces[iF])
                bA,bB,bC = (self.m2b_vertex[v] for v in self.complete_mesh.faces[iF])
                D = [x for x in self.complete_mesh.cells[iC] if x not in self.complete_mesh.faces[iF]][0] # fourth point in cell but not on face
                pD = self.complete_mesh.vertices[D]
                if geom.det_3x3(pA-pD,pB-pD,pC-pD)>0:
                    boundary.faces.append((bA,bB,bC))
                else:
                    boundary.faces.append((bA,bC,bB))

            # build indirection map for edges
            for e in self.complete_mesh.boundary_edges:
                u,v = self.complete_mesh.edges[e]
                bu,bv = self.m2b_vertex[u], self.m2b_vertex[v]
                be = self.edge_id(bu,bv)
                self.m2b_edge[e] = be
                self.b2m_edge[be] = e
            return boundary.instantiate(2)
        
        ##### Vertex to elements #####

        def vertex_to_vertices(self, V : int):
            Vb = self.m2b_vertex.get(V,None)
            if Vb is None: return []
            return [self.b2m_vertex[_v] for _v in super().vertex_to_vertices(Vb)]
        
        def vertex_to_edges(self, V : int) :
            Vb = self.m2b_vertex.get(V,None)
            if Vb is None: return []
            edges = [self.edge_id(Vb,Wb) for Wb in super().vertex_to_vertices(Vb)]
            return [self.b2m_edge[e] for e in edges]

        def vertex_to_faces(self, V):
            Vb = self.m2b_vertex.get(V,None)
            if Vb is None: return
            return [self.b2m_face[f] for f in super().vertex_to_faces(Vb)]
            
        ##### Faces to elements #####

        def face_to_vertices(self, F):
            Fb = self.m2b_face.get(F,None)
            if Fb is None : return []
            return list(self.mesh.faces[Fb])

        def in_face_index(self, F, V):
            """Index of vertex V in face F. None if V is not in face F

            Parameters:
                F (int): face index
                V (int): vertex index

            Returns:
                int
            """
            Fb,Vb = self.m2b_face.get(F,None), self.m2b_vertex.get(V,None)
            if Fb is None or Vb is None : return None
            return super().in_face_index(Fb,Vb)
        
        def face_to_edges(self, F):
            Fb = self.m2b_face.get(F,None)
            if Fb is None: return []
            return  [self.b2m_edge[_e] for _e in super().face_to_edges(Fb)]

        def face_to_faces(self, F):
            Fb = self.m2b_face.get(F,None)
            if Fb is None: return []
            return [self.b2m_face[_f] for _f in super().face_to_faces(Fb)]