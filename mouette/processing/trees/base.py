from abc import ABC, abstractmethod
from collections import deque
from ...mesh.datatypes import *
from ...mesh import merge

class SpanningTree(ABC):
    """A spanning tree defined over the connectivity of a mesh"""

    def __init__(self, mesh : Mesh):
        self.mesh : Mesh = mesh

        self.root = None

        self.parent : list = None # v -> parent of v
        self.children : list = None # v -> list of indices of children of v (inverse of self.parent)
        self.edges : list = None # list of (u,v) with u<v and u parent of v (or v parent of u)

        self._computed : bool = False

    def __call__(self):
        self.compute()
        return self

    @abstractmethod
    def compute(self) -> None :
        """Runs the traversal and builds the tree"""
        self._computed = True

    def traverse(self, order="BFS"):
        """
        Iterator on the nodes of a tree. 
        Returns tuple of form (node, parent)
        
        Parameters:
            order (str, optional):  BFS or DFS order. Defaults to "BFS".
        """
        if not self._computed:
            raise Exception("Tree was not computed. Call the .compute() method first")

        if order not in ["BFS", "DFS"]:
            raise Exception("Error when traversing a spanning tree: given 'order' argument should be 'DFS' or 'BFS'. Got {}".format(order))
            
        isBFS = (order=="BFS")
        queue = deque()
        def pop():
            if isBFS: return queue.popleft()
            return queue.pop()
        queue.append((self.root, None))
        while len(queue)>0:
            node,parent = pop()
            yield node,parent
            for child in self.children[node]:
                queue.append((child,node))

    @abstractmethod
    def build_tree_as_polyline(self) -> PolyLine : 
        pass

class SpanningForest(ABC):
    """
    Unlike SpanningTree, SpanningForest allows to define several roots and is guaranteed to visit every element of a mesh.
    """

    @forbidden_mesh_types(PointCloud)
    def __init__(self, mesh : Mesh):
        self.mesh  : Mesh = mesh
        self.trees : list = [] # a list of SpanningTree objects
        self.roots : list = []

    def __call__(self):
        self.compute()
        return self
        
    @property
    def n_trees(self) -> int:
        """Number of trees in the forest
        
        Returns:
            int:
        """
        return len(self.trees)

    def __getitem__(self, key):
        return self.trees[key]

    @property
    def edges(self) -> list:
        """List of edges in the trees

        Returns:
            list: list of edges
        """
        _edges = []
        for tree in self.trees:
            _edges += tree.edges
        return _edges

    @abstractmethod
    def compute(self) -> None : 
        pass

    def traverse(self, order="BFS"):
        """
        Iterator on the nodes of a tree 
        
        Parameters:
            order (str, optional):  BFS or DFS order. Defaults to "BFS".
        """
        for tree in self.trees:
            for el in tree.traverse(order=order):
                yield el

    def build_tree_as_polyline(self) -> PolyLine :
        """Builds the tree as a new polyline object. Useful for debug and visualization purposes

        Returns:
            PolyLine: the tree
        """
        trees = [t.build_tree_as_polyline() for t in self.trees]
        return merge(trees)