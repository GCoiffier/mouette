from ..mesh.datatypes import *
from ..mesh import RawMeshData
from .connection import SurfaceConnectionVertices
from .worker import Worker
from .. import geometry as geom
from ..geometry import Vec
from ..utils import PriorityQueue

from collections.abc import Iterable
from math import pi

class DiscreteExponentialMap(Worker):
    """
    Implementation of Discrete Exponential map

    References:
        - [1] Interactive decal compositing with discrete exponential maps, Schmidt et al., 2006

    Usage:
    ```
    conn = SurfaceConnectionVertices(mesh)
    expm = DiscreteExponentialMap(mesh, conn, rad)
    expm.run({0, 2, 42}) # computes map for vertices 0, 2 and 42

    expm.run() # computes map for all vertices

    u,v = expm.map(0, 3) # coordinates of vertex 3 in exp map of vertex 0
    u,v = expm.map(3, 0) # coordinates of vertex 0 in exp map of vertex 3. Exp map of 3 is computed on the go if necessary
    ```
    """
    def __init__(self, 
        mesh : SurfaceMesh, 
        conn : SurfaceConnectionVertices,
        radius : int = 2,
        verbose : bool = False
    ):
        """
        Args:
            mesh (SurfaceMesh): input surface mesh
            conn (SurfaceConnectionVertices): connection over vertices for the mesh.
            radius (int, optional): Radius in number of edges of the exponential map of each vertex. Defaults to 2.
            verbose (bool, optional): verbose options. Defaults to False.
        """
        super().__init__("ExpMap", verbose)
        self.mesh = mesh
        self.conn = conn
        self.radius = radius
        self._map : dict = dict()

    def run(self, vertices : Iterable = None):
        if vertices is None : 
            vertices = self.mesh.id_vertices
        for u in vertices:
            self._compute_map(u)
    
    def _compute_map(self, u : int):
        """
        Computes map originating from vertex u for all vertices v at most at distance 'self.d' from u in terms of number of edges
        """
        self._map[u] = dict()
        
        ### Perform graph traversal from origin u and stop as soon as distance > self.d
        visited = set()
        dist = dict()
        rot = dict()

        q = PriorityQueue()
        q.push(u,0) # weights are distance to the origin u
        dist[u] = 0
        rot[u] = 0.
        self._map[u][u] = Vec(0.,0.)

        while not q.empty():
            front = q.get()
            v,rad = front.x, front.priority
            if rad>=self.radius : continue # ignore since too far from origin
            if v in visited : continue # vertex was already set
            visited.add(v)
            for nv in self.mesh.connectivity.vertex_to_vertices(v):
                dv = dist[v] + geom.distance(self.mesh.vertices[v], self.mesh.vertices[nv])
                if dist.get(nv, float("inf")) > dv:
                    dist[nv] = dv
                    E = self.mesh.vertices[nv] - self.mesh.vertices[v]
                    E = self.conn.project(E, v)
                    rot[nv] = rot[v] + self.conn.transport(v,nv) - self.conn.transport(nv,v) + pi
                    self._map[u][nv] = self._map[u][v] + geom.rotate_2d(E, rot[v])
                if nv not in visited:
                    q.push(nv, rad+1)

    def map(self, v_source : int, v_target : int):
        """
        Returns the uv-coordinates of v_target in the tangent plane of v_source.
        If the map of v_source was not computed, it is computed on the go.

        Args:
            v_source (int): target vertex index, center of the exponential map
            v_target (int): query vertex index

        Returns:
            Vec: uv-coordinates of v_target as projected in the tangent plane of v_source. Returns None if v_target is not in the map of v_source (vertex is invalid or too far)
        """
        if v_source not in self._map:
            self._compute_map(v_source)
        return self._map[v_source].get(v_target, None)

    def export_map_as_mesh(self, v_source : int) -> SurfaceMesh:
        """
        Exports the map of vertex v_source as a flat surface mesh for visualization purposes 

        Args:
            v_source (int): source vertex index

        Returns:
            SurfaceMesh: exponential map of v_source as a surface mesh
        """
        m = RawMeshData()
        index_map = dict()

        if v_source not in self._map:
            self._compute_map(v_source)
        
        ### Add root
        m.vertices.append([0.,0.,0.])
        index_map[v_source] = 0.

        ### Compute point positions
        for iv,(v,pV) in enumerate(self._map[v_source].items()):
            index_map[v] = iv+1
            m.vertices.append(Vec(pV.x, pV.y, 0.))
        
        ### Add faces
        for F in self.mesh.faces:
            if all([x in self._map[v_source] for x in F]):
                m.faces.append([ index_map[x] for x in F ])

        return SurfaceMesh(m)


