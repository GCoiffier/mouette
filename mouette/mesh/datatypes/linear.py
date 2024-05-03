from ..mesh_data import RawMeshData
from .base import Mesh
from ... import utils

class PolyLine(Mesh): 
    """
    A data structure for representing polylines.

    Attributes:
        vertices (DataContainer): the container for all vertices
        edges (DataContainer): the container for all edges
        __str__: Representation of the object and its elements as a string.
    """

    def __init__(self, data : RawMeshData = None):
        Mesh.__init__(self, 1, data)
        self.connectivity = PolyLine._Connectivity(self) 

    def __str__(self):
        out = "mouette.mesh.Polyline object\n"
        out += "| {} vertices\n".format(len(self.vertices))
        out += "| {} edges\n".format(len(self.edges))
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

    class _Connectivity:
        """
        Connectivity handler class. Allows connectivity queries on the object, like neighbours of a vertex, etc.

        Connectivity is computed lazily only when needed in the code. For instance, the first call to `mesh.connectivity.vertex_to_vertices` will generate the vertex_to_vertices dictionnary, so that the next call will not perform any computation but a lookup in the array.
        """
        
        def __init__(self, master):
            self.mesh = master

            self._edge_id : dict = None # (v1, v2) -> edge | None
            self._adjV2V : dict = None # vertex -> vertex

        def clear(self):
            """
            Resets connectivity. 
            The next query in the code will regenerate internal arrays.
            """
            self._edge_id = None
            self._adjV2V = None

        def _compute_connectivity(self):
            self._adjV2V = dict([(i,set()) for i in self.mesh.id_vertices ])

            for (A,B) in self.mesh.edges:
                assert A != B # should not be possible
                self._adjV2V[A].add(B)
                self._adjV2V[B].add(A)
                    
            for U in self.mesh.id_vertices:
                self._adjV2V[U] = list(self._adjV2V[U])

        def _compute_edge_id(self):
            self._edge_id = dict()
            for iE,E in enumerate(self.mesh.edges):
                key = utils.keyify(E)
                self._edge_id[key] = iE

        def edge_id(self, V1:int, V2:int)->int:
            """
            id of an edge. If `self.edges[i]` contains edges `(A,B)`, then `edge_id(A,B)=edge_id(B,A)=i`
            If (A,B) is not a valid edge of the mesh, returns `None`

            Args:
                V1 (int): first vertex of the edge
                V2 (int): second vertex of the edge

            Returns:
                int: the id of edge (V1,V2), or `None` if the edge does not exist.
            """
            if self._edge_id is None:
                self._compute_edge_id()
            key = utils.keyify(V1,V2)
            return self._edge_id.get(key, None)

        def other_edge_end(self, E:int, V:int) -> int:
            """
            Vertex at the opposite end of edge `E` from vertex `V`.
            Returns `None` if `V` is not adjacent to edge `E`

            Args:
                E (int): edge id
                V (int): vertex id

            Returns:
                int: the vertex `W` such that `E` is the edge `(V,W)`. Returns `None` if `V` is not adjacent to edge `E`
            """
            A,B = self.mesh.edges[E]
            if V==A: return B
            if V==B: return A
            return None

        ##### Vertex to elements

        def vertex_to_vertices(self, V: int) -> list:
            """
            Neighborhood of vertex `V` in terms of vertices.

            Args:
                V (int): vertex id

            Returns:
                list: list of vertices `W` such that `(V,W)` is a valid edge in the polyline.
            """
            if self._adjV2V is None : 
                self._compute_connectivity()
            return self._adjV2V[V]

        def vertex_to_edges(self, V: int) -> list:
            """
            Neighborhood of vertex `V` in terms of edges.

            Args:
                V (int): vertex id

            Returns:
                list: list of edges E such that V belongs to E.
            """
            return [ self.edge_id(V, u) for u in self.vertex_to_vertices(V)]


        ##### Edge to elements

        def edge_to_vertices(self, E: int) -> list:
            """Returns the two vertex indices that are adjacent to edge `E

            Args:
                E (int): edge index

            Returns:
                list: two vertex indices
            """
            return self.mesh.edges[E]