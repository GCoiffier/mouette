from ..data_container import DataContainer
from .base import Mesh
from ... import utils
import numpy as np
from numba import njit

@njit(cache=True)
def _polyline_csr_kernel(E, nV):
    """Builds the vertex -> vertex adjacency of a polyline in CSR format.

    Args:
        E (np.ndarray): edges, int64 array of shape (nE,2)
        nV (int): number of vertices

    Returns:
        v2v, v2v_off, v2e: neighbors of vertex i are v2v[v2v_off[i]:v2v_off[i+1]], and v2e holds the matching edge ids.
    """
    nE = E.shape[0]
    ### Count the number of neighbors of each vertex
    counter = np.zeros(nV, dtype=np.int64)
    for iE in range(nE):
        counter[E[iE,0]] += 1
        counter[E[iE,1]] += 1
    ### Offsets are the cumulative sum of degrees
    v2v_off = np.zeros(nV+1, dtype=np.int64)
    v2v_off[1:] = np.cumsum(counter)
    ### Fill CSR arrays (total size is the sum of degrees = 2*nE)
    v2v = np.zeros(2*nE, dtype=np.int64)
    v2e = np.zeros(2*nE, dtype=np.int64)
    counter[:] = 0
    for iE in range(nE):
        A,B = E[iE,0], E[iE,1]
        iA = v2v_off[A] + counter[A]
        v2v[iA] = B
        v2e[iA] = iE
        counter[A] += 1
        iB = v2v_off[B] + counter[B]
        v2v[iB] = A
        v2e[iB] = iE
        counter[B] += 1
    return v2v, v2v_off, v2e

class PolyLine(Mesh): 
    """
    A data structure for representing polylines.

    Attributes:
        vertices (DataContainer): the container for all vertices
        edges (DataContainer): the container for all edges
        __str__: Representation of the object and its elements as a string.
    """

    def __init__(self, vertices : DataContainer, edges : DataContainer):
        vertices = vertices or DataContainer(id="vertices")
        edges = edges or DataContainer(id="edges")
        Mesh.__init__(self, vertices=vertices, edges=edges)
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

            self._edge_id : dict = None # (v1, v2) -> edge
            self._adjV2V : list = None # vertex -> vertex

        def clear(self):
            """
            Resets connectivity. 
            The next query in the code will regenerate internal arrays.
            """
            self._edge_id = None
            self._adjV2V = None

        def _compute_connectivity(self):
            self._adjV2V = [[] for _ in self.mesh.id_vertices]

            for (A,B) in self.mesh.edges:
                assert A != B # should not be possible
                self._adjV2V[A].append(int(B))
                self._adjV2V[B].append(int(A))

        def _compute_edge_id(self):
            self._edge_id = dict()
            for iE,(A,B) in enumerate(self.mesh.edges):
                key = max(A,B) + len(self.mesh.vertices)*min(A,B)
                self._edge_id[key] = iE

        def edge_id(self, V1:int, V2:int)->int|None:
            """
            Identifier of an edge. If `self.edges[i]` contains edges `(A,B)`, then `edge_id(A,B)=edge_id(B,A)=i`
            
            If (A,B) is not a valid edge of the mesh, returns `None`

            Args:
                V1 (int): first vertex of the edge
                V2 (int): second vertex of the edge

            Returns:
                int: the id of edge (V1,V2), or `None` if the edge does not exist.
            """
            if self._edge_id is None:
                self._compute_edge_id()
            key = max(V1,V2) + len(self.mesh.vertices)*min(V1,V2)
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
                list: list of vertices `W` such that `(V,W)` is a valid edge in the mesh.
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

            Note:
                Equivalent to `mesh.edges[E]`

            Returns:
                list: two vertex indices
            """
            return self.mesh.edges[E]
