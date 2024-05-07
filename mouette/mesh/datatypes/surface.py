from ..mesh_data import RawMeshData
from ..mesh_attributes import Attribute
from .base import Mesh
from .linear import PolyLine
from ... import utils
from ... import config

class SurfaceMesh(Mesh):
    """A data structure for representing polygonal surfaces.
    
    Attributes:
        vertices (DataContainer): the container for all vertices
        edges (DataContainer): the container for all edges
        faces (DataContainer): the container for all faces
        face_corners (DataContainer): the container for all corner of faces

        boundary_edges (list): list of all edge indices on the boundary
        interior_edges (list): list of all interior edge indices (all edges \\ boundary_edges)
        boundary_vertices (list): list of all vertex indices on the boundary
        interior_vertices (list): list of all interior verticex indices (all vertices \\ boundary_vertices)

        connectivity (_SurfaceConnectivity): the connectivity utility class
    """

    def __init__(self, data : RawMeshData = None):
        Mesh.__init__(self, 2, data)

        self.connectivity = SurfaceMesh._Connectivity(self)

        # Boundary data
        self._boundary_edges : list = None
        self._interior_edges : list = None

        self._is_vertex_on_border : Attribute = None
        self._boundary_vertices : list = None
        self._interior_vertices : list = None

        self._is_triangular : bool = None
        self._is_quad : bool = None

    def __str__(self):
        out = "mouette.mesh.SurfaceMesh object\n"
        out += "| {} vertices\n".format(len(self.vertices))
        out += "| {} edges\n".format(len(self.edges))
        out += "| {} faces\n".format(len(self.faces))
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
    def id_corners(self):
        """
        Shortcut for `range(len(self.face_corners))`
        """
        return range(len(self.face_corners))

    def ith_vertex_of_face(self, fid: int, i: int) -> int:
        """
        helper function to get the i-th vertex of a face, i.e. `self.faces[fid][i]`

        Args:
            fid (int): face id
            i (int): vertex id in face. Should be 0 <= vid < len(face)

        Returns:
            int: the id of the i-th vertex in face `fid` (`self.faces[fid][i]`)
        """
        return self.faces[fid][i]

    def pt_of_face(self, fid:int):
        """
        point coordinates of vertices of face `fid`

        Args:
            fid (int): face id

        Returns:
            Iterable: iterator of Vec objects representing point coordinates of vertices
        """
        return (self.vertices[_v] for _v in self.faces[fid])

    def _compute_mesh_type(self):
        self._is_triangular = True
        self._is_quad = True
        for f in self.faces:
            self._is_triangular = self._is_triangular and len(f)==3
            self._is_quad = self._is_quad and len(f)==4

    def is_triangular(self) -> bool:
        """
        Returns:
            bool: True if the mesh is triangular (all faces are triangles)
        """
        if self._is_triangular is None:
            self._compute_mesh_type()
        return self._is_triangular

    def is_quad(self) -> bool:
        """
        Returns:
            bool: True if the mesh is quadrangular (all faces are quad)
        """
        if self._is_quad is None:
            self._compute_mesh_type()
        return self._is_quad

    def clear_boundary_data(self):
        """
        Clear all boundary data.
        Next call to a boundary/interior container or method will recompute everything
        """
        self._boundary_edges : list = None
        self._interior_edges : list = None

        self._is_vertex_on_border : Attribute = None 
        self._boundary_vertices : set = None
        self._interior_vertices : set = None

    ##### Accessors for boundary and exterior ######

    def _compute_interior_boundary_edges(self):
        self._interior_edges = []
        self._boundary_edges = []
        for e,(u,v) in enumerate(self.edges):
            if self.is_edge_on_border(u,v):
                self._boundary_edges.append(e)
            else:
                self._interior_edges.append(e)
    
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

    def is_edge_on_border(self, u:int, v:int) -> bool:
        """
        whether edge (u,v) is a boundary edge or not

        Args:
            u (int): vertex id
            v (int): vertex id

        Returns:
            bool: whether edge (u,v) is a boundary edge or not. Returns False if (u,v) is not a valid edge.
        """
        if self.connectivity.edge_id(u,v) is None: return False
        return self.connectivity.direct_face(u,v) is None or self.connectivity.direct_face(v,u) is None

    def is_vertex_on_border(self, u:int) -> bool:
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

    ###### Connectivity ######

    class _Connectivity(PolyLine._Connectivity):
        """Connectivity for surface meshes. Is accessed via the `.connectivity` attribute of the `SurfaceMesh` class.

        Warning:
            If your mesh is not manifold, there is no guarantee that the connectivity arrays will be correct.
        """

        def __init__(self, master):
            super().__init__(master)

            self._half_edges : dict = None # (v1,v2) -> (corner, previous, next, opposite, face, i1, i2)
            self._Cn2he      : dict = None # corner -> (v1,v2)
            self._adjVF2Cn   : dict = None # vertex,face -> corner
            self._adjV2Cn    : dict = None # vertex -> corners
            self._adjF2Cn    : dict = None # face -> corners
            self._face_id    : dict = None # (list of vertices) -> face
        
        def clear(self):
            """
            Resets connectivity. 
            The next query in the code will regenerate internal arrays.
            """
            super().clear()
            self._half_edges : dict = None # (v1,v2) -> (corner, previous, next, opposite, face, i1, i2)
            self._Cn2he      : dict = None # corner -> (v1,v2)
            self._adjVF2Cn   : dict = None # vertex,face -> corner
            self._adjV2Cn    : dict = None # vertex -> corners
            self._adjF2Cn    : dict = None # face -> first corner
            self._face_id    : dict = None # (list of vertices) -> face

        def _compute_face_ids(self):
            self._face_id = dict()
            for iF, F in enumerate(self.mesh.faces):
                self._face_id[utils.keyify(F)] = iF

        def face_id(self, *args) -> int:
            """ The id of a face
            Args:
                int*: integers representing indices of vertices of the face (not necessarily in the correct order)

            Returns:
                int: A face index or None if the given tuple is invalid
            """
            if self._face_id is None:
                self._compute_face_ids()
            key = utils.keyify(*args)
            return self._face_id.get(key, None)

        def _compute_connectivity(self):
            super()._compute_connectivity()

            ### Compute corner connectivity dictionnaries
            self._adjV2Cn = dict([(i,set()) for i in self.mesh.id_vertices])
            self._adjVF2Cn = dict()
            self._adjF2Cn = dict()
            for iC in self.mesh.id_corners:
                v,f = self.mesh.face_corners.element(iC), self.mesh.face_corners.adj(iC)
                self._adjV2Cn[v].add(iC)
                self._adjVF2Cn[(v,f)] = iC
                if f not in self._adjF2Cn:
                    self._adjF2Cn[f] = iC
            for v in self.mesh.id_vertices:
                # recast sets as list for indexing and sorting
                self._adjV2Cn[v] = list(self._adjV2Cn[v])
                
            ### Compute half edges
            self._half_edges = dict()
            self._Cn2he = dict()
            for iF,F in enumerate(self.mesh.faces):
                n = len(F)
                for iV in range(n):
                    P, Pprev, Pnext = F[iV], F[(iV-1)%n], F[(iV+1)%n]
                    iC, iCprev, iCnext = (self._adjVF2Cn[(p,iF)] for p in (P,Pprev,Pnext)) 
                    self._half_edges[ (P, Pnext) ] = [iC, iCprev, iCnext, None, iF, iV, (iV+1)%n]
                    self._Cn2he[iC] = (P,Pnext)
            for (A,B) in self._half_edges.keys():
                iC1 = self._half_edges.get((A,B), [None])[0] # index of corner if it exists, else None 
                iC2 = self._half_edges.get((B,A), [None])[0]
                if iC1 is not None and iC2 is not None:
                    # iC1 and iC2 are opposite corners
                    self._half_edges[(A,B)][3] = iC2
                    self._half_edges[(B,A)][3] = iC1

            ### Sorting vertex neighborhoods
            if config.sort_neighborhoods and isinstance(self.mesh, SurfaceMesh):
                # Neighborhood should be sorted to have a clockwise order
                # Ignore sorting for volume meshes
                self._sort_vertex_neighborhoods()
            
        def _sort_vertex_neighborhoods(self):
            for A in self.mesh.id_vertices:
                corners_A = self._adjV2Cn[A]
                if len(corners_A)==0: continue
                sort_index = dict([(c,0) for c in corners_A])
                ind = 0
                is_boundary = False
                Cn = corners_A[0]
                for _ in range(len(sort_index)):
                    sort_index[Cn] = ind 
                    ind -= 1 
                    Cn = self.opposite_corner(self.previous_corner(Cn))
                    if Cn is None: 
                        is_boundary = True
                        break
                if is_boundary:
                    # also go counter clockwise 
                    ind = 0
                    Cn = corners_A[0]
                    for _ in range(len(sort_index)):
                        sort_index[Cn] = ind 
                        ind +=1
                        Cn = self.opposite_corner(Cn)
                        if Cn is None : break
                        Cn = self.next_corner(Cn)
                self._adjV2Cn[A].sort(key = lambda c : sort_index[c])
                sort_indexV = dict()
                for v in self._adjV2V[A]:
                    # all vertices have an associated corner except the last one (opposite half edge does not exist on boundary)
                    sort_indexV[v] = sort_index.get(self.half_edge_to_corner(A,v), -float("inf"))
                self._adjV2V[A].sort(key = lambda v : sort_indexV[v])

        ##### Vertices to elements

        def vertex_to_faces(self, V: int) -> list:
            """
            Neighborhood of vertex `V` in terms of faces.

            Args:
                V (int): vertex id

            Returns:
                list: list of faces `F` such that `V` is a vertex of `F`.
            """
            return [self.corner_to_face(iC) for iC in self.vertex_to_corners(V)]

        def vertex_to_corners(self, V: int) -> list:
            """
            List of face corners that correspond to vertex `V`

            Args:
                V (int): vertex id

            Returns:
                list: the list of corners `C` such that `mesh.corners[C]==V`
            """
            if self._adjV2Cn is None :
                self._compute_connectivity()
            return self._adjV2Cn.get(V, None)
        
        def vertex_to_corner_in_face(self, V: int, F: int) -> int:
            """
            The corner `C` corresponding to vertex `V` in face `F`.

            Args:
                V (int): vertex id
                F (int): face id

            Returns:
                int: corner id, or `None` if `V` is not a vertex of `F`.
            """
            if self._adjVF2Cn is None:
                self._compute_connectivity()
            return self._adjVF2Cn.get((V,F), None)
        
        #### Corner to element
        
        def previous_corner(self, C: int) -> int:
            """Previous corner of `C` around its associated face

            Args:
                C (int): corner index

            Returns:
                int: index of the previous corner
            """
            if self._half_edges is None:
                self._compute_connectivity()
            key = self._Cn2he.get(C,None)
            if key is None: return None
            return self._half_edges[key][1]

        def next_corner(self, C:int) -> int:
            """Next corner of `C` around its associated face

            Args:
                C (int): corner index

            Returns:
                int: index of the next corner
            """
            if self._half_edges is None:
                self._compute_connectivity()
            key = self._Cn2he.get(C,None)
            if key is None: return None
            return self._half_edges[key][2]

        def opposite_corner(self, C: int) -> int:
            """Opposite corner of `C` in terms of half edges. 
            If `C.vertex = A` and `C.next.vertex = B`, then returns the corner D such that `D.vertex = B` and `D.vertex.next = A`

            Args:
                C (int): corner index

            Returns:
                int: index of the opposite corner
            """
            if self._half_edges is None:
                self._compute_connectivity()
            key = self._Cn2he.get(C,None)
            if key is None: return None
            return self._half_edges[key][3]

        def corner_to_half_edge(self, C: int) -> int:
            if self._Cn2he is None:
                self._compute_connectivity()
            return self._Cn2he.get(C,None)

        def corner_to_face(self, C: int) -> int:
            """
            The face inside which corner `C` belongs.

            Args:
                C (int): corner id

            Returns:
                int: face id
            """
            return self.mesh.face_corners.adj(C)
        
        #### Edge to element
        
        def half_edge_to_corner(self, u: int, v: int) -> int:
            return self._half_edges.get((u,v), [None])[0]

        def direct_face(self, u: int, v: int, return_inds: bool = False):
            """Pair (u,v) of vertex -> triangle to the left of edge (u,v) if edge (u,v) exists, None otherwise
            Also returns local indexes of u and v in the triangle (and None if (u,v) does not exists)
            
            Calling this function with edge (v,u) yield the triangle of the other side of the edge

            Args:
                u (int): first vertex index
                v (int): second vertex index
                return_inds (bool, optional): Whether to return local indices of u and v in the face. Defaults to False.

            Returns:
                Index of a face or None. If return_inds is True, tuple made of index of face and local indices of u and v in face or (None,None,None). 
            """
            if self._half_edges is None:
                self._compute_connectivity()
            if (u,v) in self._half_edges:
                if return_inds: return self._half_edges[(u,v)][4:]
                return self._half_edges[(u,v)][4]
            else:
                if return_inds: return None,None,None
                return None

        def edge_to_faces(self, u: int, v: int):
            return self.direct_face(u,v), self.direct_face(v,u)

        def opposite_face(self, u : int, v : int, T : int, return_inds: bool = False):
            """Given a pair of vertices (u,v) and a face T, returns the face (and local indexes of u and v) on the other side of edge (u,v)

            if (u,v) are not two vertices of the face T, returns None
            """
            if return_inds:
                T1, u1, v1 = self.direct_face(u,v, True)
                T2, v2, u2 = self.direct_face(v,u, True)
                if T1==T: return T2,u2,v2
                if T2==T: return T1,u1,v1
                return None,None,None
            else:
                T1, T2 = self.direct_face(u,v), self.direct_face(v,u)
                if T==T1: return T2
                if T==T2: return T1
                return None

        def common_edge(self, iF1: int, iF2: int):
            """Returns the two vertices (u,v) of the edge that separates faces iF1 and iF2 if it exists, and (None,None) otherwise.

            Args:
                iF1 (int): first face index
                iF2 (int): second face index

            Returns:
                (int,int): (u,v) pair of vertex indices, or (None,None)
            """
            F1 = self.mesh.faces[iF1]
            n = len(F1)
            for i in range(n):
                A,B = F1[i], F1[(i+1)%n]
                if self.opposite_face(A,B,iF1)==iF2:
                    return utils.keyify(A,B)
            return None,None

        ##### Face to element

        def face_to_vertices(self, F:int) -> list:
            """
            Neighborhood of face `F` in terms of vertices.

            Note:
                Equivalent to `mesh.faces[F]`

            Args:
                F (int): face id

            Returns:
                list: list of vertices `V` such that `V` is a vertex of `F`.
            """
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
        
        def face_to_edges(self, F:int) -> list:
            """
            List of edges that bound face `F`.

            Args:
                F (int): face id

            Returns:
                list: list of edges `E` such that `E` is a boundary edge of face `F`
            """
            lF = self.mesh.faces[F]
            n = len(lF)
            return [self.edge_id(lF[i],lF[(i+1)%n]) for i in range(n)]

        def face_to_first_corner(self, F: int) -> int:
            """
            One corner `C` of the face `F` (the first in order of appearance in the `face_corners` container)

            Args:
                F (int): face id

            Returns:
                int: corner id
            """
            if self._adjF2Cn is None:
                self._compute_connectivity()
            return self._adjF2Cn[F]

        def face_to_corners(self, F: int) -> list:
            """
            list of corners of face `F`

            Args:
                F (int): face id

            Returns:
                list: list of corners of face `F`
            """
            if self._adjF2Cn is None:
                self._compute_connectivity()
            return [self._adjF2Cn[F] + _i for _i in range(len(self.mesh.faces[F]))]

        def face_to_faces(self, F:int) -> list:
            """
            List of faces that are adjacent to face `F

            Args:
                F (int): face id

            Returns:
                list: list of faces `G` that are adjacent to `F`
            """
            if self._adjF2Cn is None:
                self._compute_connectivity()
            opposites = [self.opposite_corner(C) for C in self.face_to_corners(F)]
            return [self.corner_to_face(Op) for Op in opposites if Op is not None]