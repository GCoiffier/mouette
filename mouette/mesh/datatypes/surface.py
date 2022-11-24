from ..mesh_data import RawMeshData
from ..mesh_attributes import Attribute
from .base import Mesh
from .linear import PolyLine
from ... import utils
from ... import config

class SurfaceMesh(Mesh):

    def __init__(self, data : RawMeshData = None):
        Mesh.__init__(self, 2, data)

        if self.is_triangular():
            self.half_edges = SurfaceMesh._HalfEdgeStructure(self)
        else:
            self.half_edges = None

        self.connectivity = SurfaceMesh._Connectivity(self)

        # Boundary data
        self._boundary_edges : list = None
        self._interior_edges : list = None

        self._is_vertex_on_border : Attribute = None 
        self._boundary_vertices : list = None
        self._interior_vertices : list = None

    def __str__(self):
        out = "mouette.mesh.SurfaceMesh object\n"
        out += "| {} vertices\n".format(len(self.vertices))
        out += "| {} edges\n".format(len(self.edges))
        out += "| {} faces\n".format(len(self.faces))
        return out

    @property
    def id_vertices(self):
        """
        Shortcut for range(len(self.vertices))
        """
        return range(len(self.vertices))

    @property
    def id_edges(self):
        """
        Shortcut for range(len(self.edges))
        """
        return range(len(self.edges))

    @property
    def id_faces(self):
        """
        Shortcut for range(len(self.faces))
        """
        return range(len(self.faces))

    @property
    def id_corners(self):
        """
        Shortcut for range(len(self.face_corners))
        """
        return range(len(self.face_corners))

    def ith_vertex_of_face(self, fid, vid):
        return self.faces[fid][vid]

    def pt_of_face(self,fid):
        return (self.vertices[_v] for _v in self.faces[fid])

    def is_triangular(self) -> bool:
        """
        Returns:
            bool: True if the mesh is triangular (all faces are triangles)
        """
        for f in self.faces:
            if len(f)!=3: return False
        return True

    def is_quad(self) -> bool:
        for f in self.faces:
            if len(f) != 4: return False
        return True

    def recompute_edges(self):
        """Recomputes the set of edges according to the set of faces """
        self.edges.clear()
        edge_set = set()
        for f in self.faces:
            nf = len(f)
            for i in range(nf):
                edge = utils.keyify(f[i], f[(i+1)%nf])
                if edge not in edge_set:
                    edge_set.add(edge)
                    self.edges.append(edge)

    def clear_boundary_data(self):
        self._boundary_edges : list = None
        self._interior_edges : list = None

        self._is_vertex_on_border : Attribute = None 
        self._boundary_vertices : list = None
        self._interior_vertices : list = None

    ##### Accessors for boundary and exterior ######

    def _compute_interior_boundary_edges(self):
        self._interior_edges = []
        self._boundary_edges = []
        for (u,v) in self.half_edges:
            id = self.connectivity.edge_id(u,v)
            if self.is_edge_on_border(u,v):
                self._boundary_edges.append(id)
            else:
                self._interior_edges.append(id)
    
    def _compute_interior_boundary_vertices(self):
        self._boundary_vertices = set()
        self._is_vertex_on_border = self.vertices.create_attribute("border", bool)
        for e in self.boundary_edges:
            a,b = self.edges[e]
            self._is_vertex_on_border[a] = True
            self._is_vertex_on_border[b] = True
            self._boundary_vertices.add(a)
            self._boundary_vertices.add(b)
        self._boundary_vertices = list(self.boundary_vertices)
        
        self._interior_vertices = []
        for x in self.id_vertices:
            if not self._is_vertex_on_border[x]:
                self._interior_vertices.append(x)

    def is_edge_on_border(self,u,v) -> bool:
        return self.half_edges.adj(u,v)[0] is None or self.half_edges.adj(v,u)[0] is None

    def is_vertex_on_border(self,u) -> bool:
        if self._is_vertex_on_border is None:
            self._compute_interior_boundary_vertices()
        return self._is_vertex_on_border[u]

    @property
    def interior_edges(self):
        if self._interior_edges is None:
            self._compute_interior_boundary_edges()
        return self._interior_edges

    @property
    def boundary_edges(self):
        if self._boundary_edges is None:
            self._compute_interior_boundary_edges()
        return self._boundary_edges

    @property
    def boundary_vertices(self):
        if self._boundary_vertices is None:
            self._compute_interior_boundary_vertices()
        return self._boundary_vertices
    
    @property
    def interior_vertices(self):
        if self._interior_vertices is None:
            self._compute_interior_boundary_vertices()
        return self._interior_vertices

    ###### Half Edges ######

    class _HalfEdgeStructure:
        def __init__(self, master):
            self.mesh : SurfaceMesh = master
            self._half_edges : dict = None

        def clear(self):
            self._half_edges : dict = None

        def _compute_half_edges(self):
            self._half_edges = dict()
            for i,F in enumerate(self.mesh.faces):
                n = len(F)
                for v in range(n):
                    self._half_edges[ (F[v], F[(v+1)%n] ) ] = (i,v,(v+1)%n)

        def __iter__(self):
            if self._half_edges is None:
                self._compute_half_edges()
            return self._half_edges.__iter__()

        def adj(self, u : int, v : int):
            """Pair (u,v) of vertex -> triangle to the left of edge (u,v) if edge (u,v) exists, None otherwise
            Also returns local indexes of u and v in the triangle (and None if (u,v) does not exists)
            
            Calling this function with edge (v,u) yield the triangle of the other side of the edge
            """
            if self._half_edges is None:
                self._compute_half_edges()
            if (u,v) in self._half_edges:
                return self._half_edges[(u,v)]
            return None,None,None

        def opposite(self, u : int, v : int, T : int):
            """Given a pair of vertices (u,v) and a face T, returns the face (and local indexes of u and v) on the other side of edge (u,v)

            if (u,v) are not two vertices of the face T, returns None
            """
            T1,u1,v1 = self.adj(u,v)
            T2,v2,u2 = self.adj(v,u)
            if T1==T:
                return T2,u2,v2
            if T2==T:
                return T1,u1,v1
            return None,None,None

        def edge_to_triangles(self, u, v):
            return self.adj(u,v)[0], self.adj(v,u)[0]

        def common_edge(self,iF1,iF2):
            F1 = self.mesh.faces[iF1]
            n = len(F1)
            for i in range(n):
                A,B = F1[i], F1[(i+1)%n]
                if self.opposite(A,B,iF1)[0]==iF2:
                    return A,B
            return None,None

        def next_around(self, v : int, u: int):
            """ 
            Turning clockwise in the neighbourhood of vertex v, finds the vertex after u.
            Returns None is such a vertex does not exist (reach boundary)
            """
            T, iV, iU = self.adj(v,u)
            if T is None: return None
            return self.mesh.ith_vertex_of_face(T, 3-iV-iU)

        def prev_around(self, v, u):
            """ 
            Turning counter clockwise in the neighbourhood of vertex v, finds the vertex after u.
            Returns None is such a vertex does not exist (reach boundary)
            """
            T, iU, iV = self.adj(u,v)
            if T is None: return None
            return self.mesh.ith_vertex_of_face(T, 3-iV-iU)

    ###### connectivity ######

    class _Connectivity(PolyLine._Connectivity):

        def __init__(self, master):
            super().__init__(master)

            self._adjV2F : dict = dict() # vertex -> triangle
            # V2F is not None because connectivity is built on the fly at each query
            
            self._adjF2F : dict = None # triangle -> triangle
            self._adjVF2Cn : dict = None # vertex,face -> corner
            self._adjCn2F : dict = None # corner -> face
            self._adjF2Cn : dict = None # face -> first corner

            self._face_id : dict = None
        
        def clear(self):
            super().clear()
            self._adjV2F : dict = dict() # vertex -> triangle            
            self._adjF2F : dict = None # triangle -> triangle
            self._adjVF2Cn : dict = None # vertex,face -> corner
            self._adjCn2F : dict = None # corner -> face
            self._adjF2Cn : dict = None # face -> first corner
            self._face_id : dict = None     

        def face_id(self, a, b, c):
            if self._face_id is None:
                self._compute_vertex_adj()
            key = utils.keyify(a,b,c)
            return self._face_id.get(key, None)

        def _compute_vertex_adj(self):
            super()._compute_vertex_adj()

            self._adjV2F = dict([(i,set()) for i in self.mesh.id_vertices])
            self._face_id = dict()

            for iF,F in enumerate(self.mesh.faces):
                key = utils.keyify(F)
                self._face_id[key] = iF
                for V in F:
                    self._adjV2F[V].add(iF)

            for U in self.mesh.id_vertices:
                self._adjV2F[U] = list(self._adjV2F[U])
            
            # Neighborhood should be sorted to have a clockwise order
            if config.sort_neighborhoods and isinstance(self.mesh, SurfaceMesh):
                self._sort_vertex_neighborhoods()

        def _sort_vertex_neighborhoods(self):
            for A in self.mesh.interior_vertices:
                if not self._adjV2V[A] : continue # no adjacent vertices to sort
                indexV, indexF = dict(), dict()
                B = self._adjV2V[A][0]
                T = self.mesh.half_edges.adj(A,B)[0]
                indexV[B] = 0
                indexF[T] = 0
                for i in range(len(self._adjV2V[A])):
                    B = self.mesh.half_edges.next_around(A,B) # B before T since root triangle is handled outside of the loop
                    T = self.mesh.half_edges.adj(A,B)[0]
                    indexV[B] = i+1
                    indexF[T] = i+1

                self._adjV2V[A].sort(key = lambda u : indexV[u])
                self._adjV2F[A].sort(key = lambda u : indexF[u])

            for A in self.mesh.boundary_vertices:
                if not self._adjV2V[A] : continue # not adjacent vertices to sort
                indexV,indexF = dict(), dict()
                B = self._adjV2V[A][0]
                ind = 0
                indexV[B] = ind
                while self.mesh.half_edges.adj(B,A)[0] is not None:
                    T = self.mesh.half_edges.adj(B,A)[0] # T before B because root triangle is not handled outside of the loop
                    B = self.mesh.half_edges.prev_around(A,B)
                    ind-=1
                    indexV[B] = ind
                    indexF[T] = ind
                # reset and go the other way around
                B = self._adjV2V[A][0]
                ind = 0
                while self.mesh.half_edges.adj(A,B)[0] is not None:
                    T = self.mesh.half_edges.adj(A,B)[0] # T before B because root triangle is not handled outside of the loop
                    B = self.mesh.half_edges.next_around(A,B)
                    ind +=1
                    indexV[B] = ind
                    indexF[T] = ind
                    
                self._adjV2V[A].sort(key = lambda u : indexV[u])
                self._adjV2F[A].sort(key = lambda u : indexF[u]) 
        
        def _compute_face_adj(self):
            self._adjF2F = dict([(t, []) for t in self.mesh.id_faces])
            for (A,B) in self.mesh.edges:
                F1, _, _ = self.mesh.half_edges.adj(A,B)
                F2, _, _ = self.mesh.half_edges.adj(B,A)
                if F1 is not None and F2 is not None:
                    self._adjF2F[F1].append(F2)
                    self._adjF2F[F2].append(F1)

        def _compute_corner_adj(self):
            self._adjVF2Cn = dict()
            self._adjCn2F = dict()
            self._adjF2Cn = dict()
            c = 0
            for iF,F in enumerate(self.mesh.faces):
                self._adjF2Cn[iF] = c
                for v in F:
                    assert self.mesh.face_corners[c] == v
                    self._adjVF2Cn[(v,iF)] = c
                    self._adjCn2F[c] = iF
                    c += 1

        ##### Vertex to Faces #####

        def n_VtoF(self, V):
            L = self._adjV2F.get(V, None)
            if L is None:
                self._adjV2F[V] = []
                for iT,T in enumerate(self.mesh.faces):
                    if V in T:
                        self._adjV2F[V].append(iT)
            return len(self._adjV2F[V])

        def vertex_to_face(self, V):
            L = self._adjV2F.get(V, None)
            if L is None :
                self._compute_vertex_adj()
            return self._adjV2F[V]
        
        ##### Corners #####

        def vertex_to_corner_in_face(self, V, F):
            if self._adjVF2Cn is None:
                self._compute_corner_adj()
            return self._adjVF2Cn.get((V,F), None)

        def vertex_to_corner(self, V):
            return [self.vertex_to_corner_in_face(V,_f) for _f in self.vertex_to_face(V)]

        def corner_to_face(self,C):
            if self._adjCn2F is None:
                self._compute_corner_adj()
            return self._adjCn2F.get(C,None)
        
        def face_to_first_corner(self,F):
            if self._adjF2Cn is None:
                self._compute_corner_adj()
            return self._adjF2Cn[F]

        def face_to_corners(self,F):
            if self._adjF2Cn is None:
                self._compute_corner_adj()
            return [self._adjF2Cn[F] + _i for _i in range(len(self.mesh.faces[F]))]

        ##### Faces to Vertex #####

        def face_to_vertex(self, F):
            return list(self.mesh.faces[F])

        def in_face_index(self, F, V):
            """Index of vertex V in face F. None if V is not in face F

            Parameters:
                F (int): face index
                V (int): vertex index

            Returns:
                int
            """
            for (i,v) in enumerate(self.mesh.faces[F]):
                if v==V: return i
            return None
        
        ##### Face to Edge ######

        def face_to_edge(self, F):
            lF = self.mesh.faces[F]
            n = len(lF)
            return [self.edge_id(lF[i],lF[(i+1)%n]) for i in range(n)]

        ###### Face to face ######

        def face_to_face(self, F):
            if self._adjF2F is None:
                self._compute_face_adj()
            return self._adjF2F[F]