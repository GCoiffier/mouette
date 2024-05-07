from random import randint
from collections import deque

from ...mesh.datatypes import *
from ...mesh import merge
from ...attributes.misc_faces import face_barycenter
from ...utils import keyify
from .base import SpanningTree, SpanningForest

class FaceSpanningTree(SpanningTree):
    """
    A spanning tree defined over the connectivity of a mesh, built over dual edges.

    Warning:
        This tree considers a unique starting point and stops when all reachable faces are visited. 
        If the mesh is disconnected, the tree will be incomplete. In this case, use `FaceSpanningForest` class instead.
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, starting_face=None, forbidden_edges:set = None):
        super().__init__(mesh)
        if starting_face is not None:
            self.root = starting_face
        else:
            self.root = randint(0,len(self.mesh.faces)-1)
        
        if forbidden_edges is None:
            self.forbidden_edges : set = set()
        else:
            self.forbidden_edges : set = forbidden_edges
        self.parent = [None]*len(self.mesh.faces)
        self.children = [[] for v in self.mesh.id_faces]
        self.edges = []

    def compute(self):
        dist_to_root = [float("inf") for v in self.mesh.id_faces]
        seen = [False for v in self.mesh.id_faces]
        queue = deque()

        def put_neighbours_in_queue(f):
            for e in self.mesh.connectivity.face_to_edges(f):
                if e not in self.forbidden_edges:
                    a,b = self.mesh.edges[e]
                    nf = self.mesh.connectivity.opposite_face(a,b,f)
                    if (nf is not None) and (not seen[nf]):
                        queue.append((f,nf))
        
        put_neighbours_in_queue(self.root)
        dist_to_root[self.root] = 0
        seen[self.root] = True

        while len(queue):
            f,nf = queue.popleft()
            if seen[nf] : continue
            seen[nf] = True
            if dist_to_root[f] + 1 < dist_to_root[nf]:
                self.parent[nf] = f
                dist_to_root[nf] = dist_to_root[f] +1
            put_neighbours_in_queue(nf)

        # build children data from parent data
        for f in self.mesh.id_faces:
            p = self.parent[f]
            if p is not None:
                self.children[p].append(f)
                self.edges.append( keyify(p,f))
        super().compute() # sets the 'computed' flag

    def build_tree_as_polyline(self):
        """Builds the tree as a new polyline object. Useful for debug and visualization purposes

        Returns:
            PolyLine: the tree
        """
        output = PolyLine()
        if self.mesh.faces.has_attribute("barycenter"):
            bary = self.mesh.faces.get_attribute("barycenter")
        else:
            bary = face_barycenter(self.mesh)
        for iF in self.mesh.id_faces:
            output.vertices.append(bary[iF])
            if self.parent[iF] is not None:
                output.edges.append([iF, self.parent[iF]])
        return output

class FaceSpanningForest(SpanningForest):
    """A spanning forest that runs on dual edges.
    Unlike a spanning tree, will create new roots and expand trees until all faces of the mesh have been visited
    """
    
    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, forbidden_edges:set = None):
        super().__init__(mesh)
        self.forbidden_edges : set = forbidden_edges

    def compute(self) -> None :
        visited = [False]*len(self.mesh.faces)
        for f in self.mesh.id_faces:
            if not visited[f]:
                # create a tree starting from v
                self.roots.append(f)
                tree_v = FaceSpanningTree(self.mesh, f, self.forbidden_edges)()
                self.trees.append(tree_v)
                for (node, _) in tree_v.traverse():
                    visited[node] = True
        super().compute() # sets the 'computed' flag