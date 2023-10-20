from ..mesh.datatypes import *
from ..mesh import RawMeshData
from .connection import SurfaceConnectionVertices
from .worker import Worker
from .. import geometry as geom
from ..geometry import Vec
from ..utils import PriorityQueue

from collections.abc import Iterable
from math import pi,cos,sin

class DiscreteExponentialMap(Worker):
    """
    Implementation of Discrete Exponential map

    References:
        - [1] Interactive decal compositing with discrete exponential maps, Schmidt et al., 2006
    """
    def __init__(self, 
        mesh : SurfaceMesh, 
        conn : SurfaceConnectionVertices,
        dist : int = 2,
        verbose : bool = False
    ):
        super().__init__("ExpMap", verbose)
        self.mesh = mesh
        self.conn = conn
        self.d = dist
        self._map : dict = dict()

    def run(self, vertices : Iterable = None):
        if vertices is None : 
            vertices = self.mesh.id_vertices
        for u in vertices:
            self._compute_map(u)
    
    def _compute_map(self, u):
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
            v,prio = front.x, front.priority
            if prio>=self.d : continue # ignore since too far from origin
            if v in visited : continue # vertex was already set
            visited.add(v)
            for nv in self.mesh.connectivity.vertex_to_vertex(v):
                dv = dist[v] + geom.distance(self.mesh.vertices[v], self.mesh.vertices[nv])
                if dist.get(nv, float("inf")) > dv:
                    dist[nv] = dv
                    E = self.mesh.vertices[nv] - self.mesh.vertices[v]
                    E = self.conn.project(E, v)
                    rot[nv] = rot[v] + self.conn.transport(v,nv) - self.conn.transport(nv,v) + pi
                    self._map[u][nv] = self._map[u][v] + geom.rotate_2d(E, rot[v])
                if nv not in visited:
                    q.push(nv, prio+1)

    def map(self, u, v):
        if u not in self._map:
            self._compute_map(u)
        return self._map[u].get(v, None)

    def export_map_as_mesh(self, u):
        m = RawMeshData()
        index_map = dict()

        if u not in self._map:
            self._compute_map(u)
        
        ### Add root
        m.vertices.append([0.,0.,0.])
        index_map[u] = 0.

        ### Compute point positions
        for iv,(v,pV) in enumerate(self._map[u].items()):
            index_map[v] = iv+1
            m.vertices.append(Vec(pV.x, pV.y, 0.))
        
        ### Add faces
        for F in self.mesh.faces:
            if all([x in self._map[u] for x in F]):
                m.faces.append([ index_map[x] for x in F ])

        return SurfaceMesh(m)


