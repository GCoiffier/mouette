from random import randint
from collections import deque
from math import isinf

from ...mesh.datatypes import *
from ...mesh.mesh_attributes import Attribute
from ...utils import keyify, UnionFind
from ...attributes.misc_edges import edge_length as attr_edge_length
from .base import SpanningForest, SpanningTree

class EdgeSpanningTree(SpanningTree):
    """A spanning tree defined over the connectivity of a mesh.
    Edges of a polyline, surface mesh or volume mesh form an undirected graph, from which we can extract a spanning tree
    
    Warning:
        This tree considers a unique starting point and stops when all reachable vertices are visited. 
        If the mesh is disconnected, the tree will be incomplete. In this case, use `EdgeSpanningForest` class instead.

    Attributes:
        parent (list): the list of indices of the parent node in the tree. `None` for the root.
        children (list): list of indices of children nodes.
        edges (list): list of edges that form the tree.
    """

    @forbidden_mesh_types(PointCloud)
    def __init__(self, mesh : Mesh, starting_vertex : int = None, avoid_boundary : bool = False, avoid_edges : set = None):
        """
        Parameters:
            mesh (Mesh): the input mesh (Polyline, Surface or Volume)
            starting_vertex (int, optional): Index of the root of the tree. If None is provided, root is chosen at random. Defaults to None.
            avoid_boundary (bool, optional): If True, boundary edges of a SurfaceMesh will not be traversed by the tree. Defaults to False.
            avoid_edges (set, optional): Set of edges that should not be travelled by the tree. 
                /!\\ If this set disconnects the mesh, the tree will be incomplete. Defaults to None.
        """
        super().__init__(mesh)
        self._avoidbound : bool = avoid_boundary
        self._avoidedges : set = avoid_edges
        if starting_vertex is not None:
            self.root = starting_vertex
        else:
            self.root = randint(0,len(self.mesh.vertices)-1)
        
        self.parent = [None]*len(self.mesh.vertices)
        self.children = [[] for _ in self.mesh.id_vertices]
        self.edges = []
    

    def _avoid_edge(self, a: int, b:int) -> bool:
        """If avoid_boundary was set to True, checks if the edge is on the boundary and should not be considered in the search.
        Always returns False otherwise.
        Parameters:
            a (int): index of first vertex of edge
            b (int): index of second vertex of edge

        Returns:
            bool : whether to avoid the edge or not
        """
        if self._avoidedges is not None:
            if self.mesh.connectivity.edge_id(a,b) in self._avoidedges:
                return True
        if not self._avoidbound or isinstance(self.mesh, PolyLine):
            return False
        return self.mesh.is_edge_on_border(a,b)

    def compute(self):
        dist_to_root = [float("inf") for v in self.mesh.id_vertices]
        seen = [False for v in self.mesh.id_vertices]
        queue = deque()

        def put_neighbours_in_queue(v):
            for nv in self.mesh.connectivity.vertex_to_vertices(v):
                if (not seen[nv]) and (not self._avoid_edge(v,nv)):
                    queue.append((v,nv))
        
        put_neighbours_in_queue(self.root)
        dist_to_root[self.root] = 0
        seen[self.root] = True

        while len(queue)>0:
            v,nv = queue.popleft()
            if seen[nv] : continue
            seen[nv] = True
            if dist_to_root[v] + 1 < dist_to_root[nv]:
                self.parent[nv] = v
                dist_to_root[nv] = dist_to_root[v] +1
            put_neighbours_in_queue(nv)

        # build children data from parent data
        for v in self.mesh.id_vertices:
            if isinf(dist_to_root[v]): continue # vertex was not reached by the tree
            p = self.parent[v]
            if p is not None:
                self.children[p].append(v)
                self.edges.append( keyify(p,v))
        super().compute() # sets the 'computed' flag

    def build_tree_as_polyline(self) -> PolyLine:
        """Builds the tree as a new polyline object. Useful for debug and visualization purposes

        Returns:
            PolyLine: the tree
        """
        output = PolyLine()
        for v in self.mesh.vertices:
            output.vertices.append(v)
        for v,father in self.traverse():
            if father is None: continue
            output.edges.append(keyify(v, father))
        return output

class EdgeMinimalSpanningTree(EdgeSpanningTree):
    """Same as EdgeSpanningTree but uses Kruskal's algorithm instead of a Breadth First Search.
    This implies that the resulting spanning tree is minimal in term of edge lengths.

    Inherits from `EdgeSpanningTree`.

    Warning:
        This tree considers a unique starting point and stops when all reachable vertices are visited. 
        If the mesh is disconnected, the tree will be incomplete. In this case, use `EdgeSpanningForest` class instead.
    """

    @forbidden_mesh_types(PointCloud)
    def __init__(self, mesh : Mesh, starting_vertex=None, avoid_boundary : bool = False, weights="length"):
        """
        Parameters:
            mesh (Mesh): the input mesh (Polyline, Surface or Volume)
            starting_vertex (int, optional): Index of the root of the tree. If None is provided, root is chosen at random. Defaults to None.
            avoid_boundary (bool, optional): If True, boundary edges will not be traversed by the tree. Defaults to False.
            weights (str | dict) : provided weights of each edge. Options are:
                - "one" : uniform weight = 1 for every edge
                - "length" : use the length of the edge
                - any dict : custom weights
            Defaults to "length".
        """
        super().__init__(mesh, starting_vertex, avoid_boundary=avoid_boundary)
        if not ((isinstance(weights, str) and weights in ["one", "length"]) or isinstance(weights, dict) or isinstance(weights, Attribute)):
            raise Exception("Acceptable weights are 'one', 'length' or a custom dict or Attribute on edges")
        self.weights = weights

    def compute(self):
        if self.weights=="one":
            edge_length = lambda _ : 1.
        elif self.weights== "length":
            _edge_length = attr_edge_length(self.mesh, persistent=False)
            edge_length = lambda e : _edge_length[e]
        else:
            edge_length = lambda e : self.weights[e]

        if not self._avoidbound or isinstance(self.mesh, PolyLine):
            edges = [e for e in self.mesh.id_edges]
        else:
            edges = [e for e,(A,B) in enumerate(self.mesh.edges) if not self.mesh.is_edge_on_border(A,B)]

        edges.sort(key = lambda e : edge_length(e))
        neighbours = [set() for _ in self.mesh.id_vertices]
        uf = UnionFind(self.mesh.id_vertices)
        for e in edges:
            a,b = self.mesh.edges[e]
            if not uf.connected(a,b):
                uf.union(a,b)
                self.edges.append(keyify(a,b))
                neighbours[a].add(b)
                neighbours[b].add(a)

        # build parents and children from neighbours -> BFS from root
        queue = deque()
        self.parent[self.root] = None
        self.children[self.root] = list(neighbours[self.root])
        for v in neighbours[self.root]:
            queue.append((v, self.root))
        while len(queue)>0:
            v,prev= queue.popleft()
            self.parent[v] = prev
            self.children[v] = [x for x in neighbours[v] if x != prev]
            for child in self.children[v]:
                queue.append((child,v))
        super().compute() # sets the 'computed' flag
        
class EdgeSpanningForest(SpanningForest):
    """A spanning forest that runs on edges. 
    Unlike a spanning tree, will create new roots and expand trees until all vertices of the mesh have been visited
    """

    @forbidden_mesh_types(PointCloud)
    def __init__(self, mesh : Mesh):
        super().__init__(mesh)

    def compute(self) -> None :
        visited = [False]*len(self.mesh.vertices)
        for v in self.mesh.id_vertices:
            if not visited[v]:
                # create a tree starting from v
                self.roots.append(v)
                tree_v = EdgeSpanningTree(self.mesh, v)()
                self.trees.append(tree_v)
                for (node, _) in tree_v.traverse():
                    visited[node] = True
        super().compute() # sets the 'computed' flag