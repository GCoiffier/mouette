from ..mesh_data import RawMeshData
from ..mesh_attributes import Attribute
from .base import Mesh
from .surface import PolyLine, SurfaceMesh
from ...utils import keyify
from ...geometry.geometry import det_3x3
from ... import config

import numpy as np

class VolumeMesh(Mesh):

    def __init__(self, data : RawMeshData = None):
        Mesh.__init__(self, 3, data)
        self.connectivity = VolumeMesh._Connectivity(self)

        # Boundary data
        self._boundary_faces : list = None
        self._interior_faces : list = None

        self._is_vertex_on_border : Attribute = None 
        self._boundary_vertices : list = None
        self._interior_vertices : list = None

        self._is_edge_on_border : Attribute = None
        self._boundary_edges : list = None
        self._interior_edges : list = None

        self.boundary_connectivity = None

    def enable_boundary_connectivity(self):
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
        return self.boundary_connectivity.mesh

    def _compute_interior_boundary_faces(self):
        self._interior_faces = []
        self._boundary_faces = []
        for F in self.id_faces:
            if self.is_face_on_border(F):
                self._boundary_faces.append(F)
            else:
                self._interior_faces.append(F)

    def _compute_interior_boundary_vertices(self):
        self._is_vertex_on_border = self.vertices.create_attribute("border",bool)
        for iF in self.boundary_faces:
            for v in self.faces[iF]:
                self._is_vertex_on_border[v] = True
        self._interior_vertices = []
        self._boundary_vertices = []
        for v in self.id_vertices:
            if self._is_vertex_on_border[v]:
                self._boundary_vertices.append(v)
            else:
                self._interior_vertices.append(v)

    def _compute_interior_boundary_edges(self):
        self._is_edge_on_border = self.edges.create_attribute("border", bool)
        for iF in self.boundary_faces:
            n = len(self.faces[iF])
            for i in range(n):
                u,v = self.faces[iF][i], self.faces[iF][(i+1)%n]
                self._is_edge_on_border[self.connectivity.edge_id(u,v)] = True
        self._interior_edges = []
        self._boundary_edges = []
        for e in self.id_edges:
            if self._is_edge_on_border[e]:
                self._boundary_edges.append(e)
            else:
                self._interior_edges.append(e)

    def is_face_on_border(self, *args) -> bool:
        """Simple test to determine if a given face is on the boundary of the mesh.
        
        Parameters:
            Either an integer index representing a face, or n integer indices representing the vertices
            
        Returns:
            bool: Returns True is the given face exists and is on the boundary of the mesh
        """
        if len(args)==1:
            n = len(self.connectivity.face_to_cells(args[0]))
        else:
            n = len(self.connectivity.face_to_cells(self.connectivity.face_id(*args)))
        return n<2
    
    def is_vertex_on_border(self, V) -> bool:
        if self._is_vertex_on_border is None:
            self._compute_interior_boundary_vertices()
        return self._is_vertex_on_border[V]
    
    def is_edge_on_border(self, *args) -> bool:
        """Simple test to determine if a given edge is on the boundary of the mesh.

        Parameters:
            Either an integer index representing an edge, or two integer indices representing two (adjacent) vertices

        Returns:
            bool: Returns True if the given edge is on the boundary of the mesh.
        """
        if self._is_edge_on_border is None:
            self._compute_interior_boundary_edges()
        if len(args)==1:
            return self._is_edge_on_border[args[0]]
        return self._is_edge_on_border[self.connectivity.edge_id(args[0],args[1])]

    def is_cell_tet(self, ic) -> bool:
        return len(self.cells[ic])==4
    
    def is_cell_hex(self, ic) -> bool:
        # /!\ not sufficient for a cell to be an hex but good enough
        return len(self.cells[ic])==8

    def is_tetrahedral(self) -> bool:
        """
        Returns:
            bool: True if the mesh is tetrahedral (all cells are tetrahedra)
        """
        return np.all([self.is_cell_tet(ic) for ic in self.id_cells])

    @property
    def boundary_faces(self):
        if self._boundary_faces is None:
            self._compute_interior_boundary_faces() 
        return self._boundary_faces

    @property
    def interior_faces(self):
        if self._interior_faces is None :
            self._compute_interior_boundary_faces() 
        return self._interior_faces

    @property
    def interior_edges(self):
        if self._interior_edges is None :
            self._compute_interior_boundary_edges()
        return self._interior_edges

    @property
    def boundary_edges(self):
        if self._boundary_edges is None:
            self._compute_interior_boundary_edges()
        return self._boundary_edges

    @property
    def boundary_vertices(self):
        if self._boundary_vertices is None :
            self._compute_interior_boundary_vertices()
        return self._boundary_vertices

    @property
    def interior_vertices(self):
        if self._interior_vertices is None:
            self._compute_interior_boundary_vertices()
        return self._interior_vertices

    class _Connectivity(SurfaceMesh._Connectivity): 

        def __init__(self, master):
            super().__init__(master)
            self._adjC2C : Attribute = None
            self._adjV2C : dict = None

            self._adjC2F : dict = None
            self._adjF2C : dict = None
            
            self._adjC2E : dict = None
            self._adjE2C : dict = None
        
        def clear(self):
            super().clear()
            self._adjC2C : Attribute = None
            self._adjV2C : dict = None
            
            self._adjC2F : dict = None
            self._adjF2C : dict = None

            self._adjE2F : dict = None
            self._adjE2C : dict = None
            self._adjC2E : dict = None

        def _compute_connectivity(self):
            super()._compute_connectivity()
            
            self._adjV2C = dict([(i,set()) for i in self.mesh.id_vertices ])
            for iC,C in enumerate(self.mesh.cells):
                for V in C:
                    self._adjV2C[V].add(iC)

            for U in self.mesh.id_vertices:
                self._adjV2C[U] = list(self._adjV2C[U])

        def _compute_adjacent_cell(self):
            # GEOGRAM_API CellDescriptor tet_descriptor = {
            #     4,         // nb_vertices
            #     4,         // nb_faces
            #     {3,3,3,3}, // nb_vertices in face
            #     {          // faces
            #         {1,3,2},
            #         {0,2,3},
            #         {3,1,0},
            #         {0,1,2}
            #     },
            #     6,         // nb_edges
            #     {          // edges
            #         {1,2}, {2,3}, {3,1}, {0,1}, {0,2}, {0,3}
            #     },
            #     {          // edges adjacent faces
            #         {0,3}, {0,1}, {0,2}, {2,3}, {3,1}, {1,2}
            #     }
            # };
            if self.mesh.cell_faces.has_attribute("adjacent_cell"):
                self._adjC2C = self.mesh.cell_faces.get_attribute("adjacent_cell")
            else:
                self._adjC2C = self.mesh.cell_faces.create_attribute("adjacent_cell", int, 1, default_value= config.NOT_AN_ID)
                for iC, cell in enumerate(self.mesh.cells):
                    v0,v1,v2,v3 = cell
                    # face fi does not contain vertex vi
                    f0,f1,f2,f3 = self.face_id(v1,v3,v2), self.face_id(v0,v2,v3), self.face_id(v3,v1,v0), self.face_id(v0,v1,v2)
                    for iF, F in enumerate((f0,f1,f2,f3)):
                        for iC2 in self.face_to_cells(F):
                            if iC2 != iC: self._adjC2C[(iC,iF)] = iC2
                            

        def _compute_cell_adj(self):
            self._adjC2F = dict([(i,[]) for i in self.mesh.id_cells ])
            self._adjF2C = dict([(i,[]) for i in self.mesh.id_faces ])

            for iC,C in enumerate(self.mesh.cells):
                if len(C)==4: # tetrahedra : every subset of 3 elements is a face
                    for i in range(4):
                        # By convention, ith adjacent face does not contain vertex i of the tet
                        F = C[:i] + C[i+1:]
                        iF = self.face_id(*F)
                        self._adjC2F[iC].append(iF)
                        self._adjF2C[iF].append(iC)                        
                else: # not all subsets are faces -> more complicated
                    self._adjC2F[iC] = set()
                    for iF,F in enumerate(self.mesh.faces):
                            sF = set(F)
                            sC = set(C)
                            if sC.issuperset(sF):
                                self._adjC2F[iC].add(iF)
                                self._adjF2C[iF].append(iC)
            for iC in self.mesh.id_cells:
                self._adjC2F[iC] = list(self._adjC2F[iC])
            for iF in self.mesh.id_faces:
                self._adjF2C[iF] = list(self._adjF2C[iF])

        def _compute_edge_id(self):
            super()._compute_edge_id() # for edge_id computation
            self._adjE2F = dict([(e, []) for e in self.mesh.id_edges])
            self._adjE2C = dict([(e, set()) for e in self.mesh.id_edges])

            for f in self.mesh.id_faces:
                c = set(self.face_to_cells(f))
                for e in self.face_to_edges(f):
                    self._adjE2F[e].append(f)
                    self._adjE2C[e] |= c
            for e in self.mesh.id_edges:
                self._adjE2C[e] = list(self._adjE2C[e])
            
            if config.sort_neighborhoods:
                self._sort_edge_neighborhoods()

        def _sort_edge_neighborhoods(self):
            """
            Sorts cells adjacent to an edge so that they are in direct order
            """
            if not self.mesh.is_tetrahedral(): return # only makes sense for tet meshes
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

        ##### Faces - Cells #####

        def n_F2C(self, F):
            return len(self.face_to_cells(F))

        def face_to_cells(self, iF):
            if self._adjF2C is None:
                self._compute_cell_adj()
            return self._adjF2C[iF]

        def cell_to_face(self, iC):
            if self._adjC2F is None:
                self._compute_cell_adj()
            return self._adjC2F[iC]

        ##### Cell - Cell #####

        def cell_to_cell(self, iC):
            if self._adjC2C is None:
                self._compute_adjacent_cell()
            return [ self._adjC2C[(iC,i)] for i in range(len(self.mesh.cells[iC])) if self._adjC2C[(iC,i)] != config.NOT_AN_ID]

        def other_face_side(self, C, F):
            if len(self.face_to_cells(F)) != 2 : return None
            C1,C2 = self.face_to_cells(F)
            if C==C1: return C2
            if C==C2 : return C1
            return None
        
        def common_face(self, C1, C2):
            common_vert = set(self.mesh.cells[C1]).intersection(self.mesh.cells[C2])
            if len(common_vert) != 3: return None
            return self.face_id(*common_vert)

        ##### Vertex - Cell #####
        def vertex_to_cell(self, V):
            if self._adjV2C is None:
                self._compute_connectivity()
            return self._adjV2C[V]
        
        def cell_to_vertex(self, C):
            return self.mesh.cells[C]
        
        def in_cell_index(self, C, V):
            """Index of vertex V in cell C. None if V is not in cell C

            Parameters:
                C (int): cell index
                V (int): vertex index

            Returns:
                int
            """
            for (i,v) in enumerate(self.mesh.cells[C]):
                if v==V: return i
            return None

        def in_cell_face_index(self,C,F):
            face_set = set(self.mesh.faces[F])
            for i,_ in enumerate(self.mesh.cells[C]):
                cell_set_i = set(self.mesh.cells[C][:i] + self.mesh.cells[C][(i+1):])
                if face_set == cell_set_i:
                    return i
            return None

        ##### Edge - face #####

        # face_to_edge already defined in parent class SurfaceMesh.connectivity
        # edge_to_face is not since SurfaceMesh takes advantage of the half edge structure
        
        def edge_to_face(self, e):
            if self._adjE2F is None:
                self._compute_edge_id()
            return self._adjE2F[e]

        ##### Edge - Cells #####

        def cell_to_edge(self, c):
            if self._adjC2E is None:
                self._adjC2E = dict()
            if self._adjC2E.get(c, None) is None:
                # build connectivity for cell C
                self._adjC2E[c] = []
                verts = self.mesh.cells[c]
                for i in range(len(verts)):
                    for j in range(i):
                        A,B = verts[i], verts[j]
                        e = self.edge_id(A,B)
                        if e is not None:
                            self._adjC2E[c].append(e)
            return self._adjC2E[c]

        def edge_to_cell(self,e):
            if self._adjE2C is None:
                self._compute_edge_id()
            return self._adjE2C[e]

    class _BoundaryConnectivity(SurfaceMesh._Connectivity):

        def __init__(self, master):
            self.complete_mesh = master

            # indirection maps between boundary elements and their indices in the full mesh
            self.m2b_vertex : dict = None # mesh to boundary
            self.m2b_edge : dict = None
            self.m2b_face : dict = None

            self.b2m_vertex : dict = None # boundary to mesh
            self.b2m_edge : dict = None
            self.b2m_face : dict = None

            mesh = self._extract_surface_boundary() # builds indirection maps for vertices and faces
            super().__init__(mesh)

            # build indirection map for edges
            for e in self.complete_mesh.boundary_edges:
                u,v = self.complete_mesh.edges[e]
                bu,bv = self.m2b_vertex[u], self.m2b_vertex[v]
                be = self.edge_id(bu,bv)
                self.m2b_edge[e] = be
                self.b2m_edge[be] = e            

        def _extract_surface_boundary(self):
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
                if det_3x3(pA-pD,pB-pD,pC-pD)>0:
                    boundary.faces.append((bA,bB,bC))
                else:
                    boundary.faces.append((bA,bC,bB))
            return SurfaceMesh(boundary)
        
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

        def vertex_to_face_quad(self, V):
            raise NotImplementedError
            
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