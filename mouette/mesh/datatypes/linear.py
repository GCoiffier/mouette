from ..mesh_data import RawMeshData
from .base import Mesh
from .pointcloud import PointCloud
from ... import utils

class PolyLine(Mesh): 
    
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
        Shortcut for range(len(self.vertices))
        """
        return range(len(self.vertices))

    @property
    def id_edges(self):
        """
        Shortcut for range(len(self.edges))
        """
        return range(len(self.edges))

    class _Connectivity:
        
        def __init__(self, master):
            self.mesh = master

            self._edge_id : dict = None # (v1, v2) -> edge | None
            self._adjV2V : dict = None # vertex -> vertex

        def clear(self):
            self._edge_id = None
            self._adjV2V = None

        def _compute_vertex_adj(self):
            self._adjV2V = dict([(i,set()) for i in self.mesh.id_vertices ])

            for (A,B) in self.mesh.edges:
                assert A != B # should not be possible
                self._adjV2V[A].add(B)
                self._adjV2V[B].add(A)
                    
            for U in self.mesh.id_vertices:
                self._adjV2V[U] = list(self._adjV2V[U])

        def _compute_edge_adj(self):
            self._edge_id = dict()
            for iE,E in enumerate(self.mesh.edges):
                key = utils.keyify(E)
                self._edge_id[key] = iE

        def edge_id(self, V1, V2):
            if self._edge_id is None:
                self._compute_edge_adj()
            key = utils.keyify(V1,V2)
            return self._edge_id.get(key, None)

        def other_edge_end(self, e, v) -> int:
            A,B = self.mesh.edges[e]
            if v==A: return B
            if v==B: return A
            return None

        ##### Vertex to Vertex #####

        def vertex_to_vertex(self, V : int):
            if self._adjV2V is None : 
                self._compute_vertex_adj()
            return self._adjV2V[V]

        def n_VtoV(self, V : int):
            if self._adjV2V is None:
                self._compute_vertex_adj()
            return len(self._adjV2V[V])

        ##### Vertex to Edges #####

        def n_VtoE(self, V : int):
            return self.n_VtoV(V)

        def vertex_to_edge(self, V : int) :
            return [ self.edge_id(V, u) for u in self.vertex_to_vertex(V)]