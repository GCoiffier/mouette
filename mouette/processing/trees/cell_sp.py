from random import randint
from collections import deque

from ...mesh.datatypes import *
from ...attributes.misc_cells import cell_barycenter
from ...utils import keyify
from .base import SpanningTree, SpanningForest

class CellSpanningTree(SpanningTree):
    """
    A spanning tree defined over the connectivity of a volume mesh.

    Warning:
        This tree considers a unique starting point and stops when all reachable vertices are visited. 
        If the mesh is disconnected, the tree will be incomplete. In this case, use `CellSpanningForest` class instead.
    """

    @allowed_mesh_types(VolumeMesh)
    def __init__(self, mesh : VolumeMesh, starting_cell=None, forbidden_faces:set = None):
        super().__init__(mesh)
        if starting_cell is not None:
            self.root = starting_cell
        else:
            self.root = randint(0,len(self.mesh.cells)-1)
        
        if forbidden_faces is None:
            self.forbidden_faces = set()
        else:
            self.forbidden_faces : set = forbidden_faces
        self.parent = [None]*len(self.mesh.cells)
        self.children = [[] for v in self.mesh.id_cells]
        self.edges = []

    def compute(self):
        dist_to_root = [float("inf") for v in self.mesh.id_cells]
        seen = [False for _ in self.mesh.id_cells]
        queue = deque()

        def put_neighbours_in_queue(c):
            for F in self.mesh.connectivity.cell_to_face(c):
                if F in self.forbidden_faces: continue
                c2 = self.mesh.connectivity.other_face_side(c,F)
                if c2 is not None and not seen[c2]:
                    queue.append((c,c2))
        
        put_neighbours_in_queue(self.root)
        dist_to_root[self.root] = 0
        seen[self.root] = True

        while len(queue)>0:
            c,nc = queue.popleft()
            if seen[nc] : continue
            seen[nc] = True
            if dist_to_root[c] + 1 < dist_to_root[nc]:
                self.parent[nc] = c
                dist_to_root[nc] = dist_to_root[c] +1
            put_neighbours_in_queue(nc)

        # build children data from parent data
        for c in self.mesh.id_cells:
            p = self.parent[c]
            if p is not None:
                self.children[p].append(c)
                self.edges.append( keyify(p,c))
        super().compute() # sets the 'computed' flag

    def build_tree_as_polyline(self) -> PolyLine:
        """Builds the tree as a new polyline object. Useful for debug and visualization purposes

        Returns:
            PolyLine: the tree
        """
        output = PolyLine()
        if self.mesh.faces.has_attribute("barycenter"):
            bary = self.mesh.cells.attribute("barycenter")
        else:
            bary = cell_barycenter(self.mesh, persistent=False)
        for iC in self.mesh.id_cells:
            output.vertices.append(bary[iC])
            if self.parent[iC] is not None:
                output.edges.append([iC, self.parent[iC]])
        return output

class CellSpanningForest(SpanningForest):
    """A spanning forest that runs between cells. 
    Unlike a spanning tree, will create new roots and expand trees until all cells of the mesh have been visited
    """

    @allowed_mesh_types(VolumeMesh)
    def __init__(self, mesh : VolumeMesh):
        super().__init__(mesh)

    def compute(self) -> None : 
        visited = [False]*len(self.mesh.cells)
        for c in self.mesh.id_cells:
            if not visited[c]:
                # create a tree starting from c
                self.roots.append(c)
                tree_v = CellSpanningTree(self.mesh, c)()
                self.trees.append(tree_v)
                for (node, _) in tree_v.traverse():
                    visited[node] = True
        super().compute() # sets the 'computed' flag