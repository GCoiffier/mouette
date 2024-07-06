import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import deque
from ..utils import check_argument, PriorityQueue
from ..geometry import AABB, distance, Vec

class KDTree:

    @dataclass
    class Node:
        id : int # id of the node
        split_axis : int # which axis does the node split
        parent : int  # id of parent node (-1 if root node)
        split_value : float = None
        left : int = None # id of the left child
        right : int = None # id of the right child
        bb : AABB = None # bounding box

    @dataclass
    class Leaf:
        id : int # id of the leaf
        split_axis : int = None # which axis does the leaf split (useful at build time, not so much after)
        parent : int = None # id of the parent node
        points : np.ndarray = None # indices of contained points
        bb : AABB = None # bounding box of contained points

        @property
        def size(self):
            return self.points.size

    class BuildStrategy(Enum):
        BALANCED = 0 # Consider the median value as the pivot at each depth. Slower but garanteed to provide a balanced tree
        FAST = 1 # Take the median of a random sampling of 100 elements. Faster than fully balanced, almost balanced in practice
        RANDOM = 2 # Consider a random pivot at each depth

        @classmethod
        def from_string(cls, txt : str):
            if txt.lower() == "balanced": return cls.BALANCED
            if txt.lower() == "fast": return cls.FAST
            if txt.lower() == "random": return cls.RANDOM

    def __init__(self, points : np.ndarray, max_leaf_size : int=10, strategy: str="fast"):
        """Implementation of a kd-tree for space partitionning and fast nearest point queries.

        Args:
            points (np.ndarray): array of points (shape (N,d)) to consider. Can be of any dimension d, though the kd-tree algorithm have poor performance when the dimension grows.
            max_leaf_size (int, optional): Maximum number of points inside a leaf. Defaults to 10.
            strategy (str, optional): Node splitting strategy. Can be either 'fast', 'balanced' or 'random'. Defaults to "fast".

        Raises:
            Exception: fails if the points are not of the correct (N,d) shape.
        """
        
        points = np.array(points)
        if len(points.shape)!=2:
            raise Exception("Expected an array of points of shape (N,dim)")
        self.points = points
        self.n_pts, self.dim = points.shape
        
        check_argument("strategy", strategy.lower(), str, ["balanced", "fast", "random"])
        self.build_strategy = KDTree.BuildStrategy.from_string(strategy)

        self.nodes = []
        self._nid = 0
        root = self._new_leaf(0, None, np.arange(self.n_pts)) # initialize a leaf node that contains all points
        root.bb = AABB.infinite(self.dim)
        queue = deque()
        queue.append(root)
        while len(queue)>0:
            leaf = queue.popleft()
            if leaf.size <= max_leaf_size: # the leaf has the correct size -> add it to the tree
                self.nodes.append(leaf)
            else: # the leaf needs to be split
                # split the points according to the current axis
                split_value, pts_less, pts_more = self._split_points(leaf.points, leaf.split_axis)
                
                # we create a new node to replace it and append two leaves that will be its children
                node = KDTree.Node(leaf.id, leaf.split_axis, parent=leaf.parent, bb=leaf.bb, split_value=split_value)

                # create and link the leaves
                leaf_less = self._new_leaf((leaf.split_axis + 1)%self.dim, leaf.id, pts_less)
                leaf_more = self._new_leaf((leaf.split_axis + 1)%self.dim, leaf.id, pts_more)
                node.left, node.right = leaf_less.id, leaf_more.id

                # split the bounding box
                bbmax_less = np.copy(leaf.bb.maxi)
                bbmax_less[leaf.split_axis] = split_value
                bbmin_more = np.copy(leaf.bb.mini)
                bbmin_more[leaf.split_axis] = split_value
                leaf_less.bb = AABB(leaf.bb.mini, bbmax_less)
                leaf_more.bb = AABB(bbmin_more, leaf.bb.maxi)

                # append and continue
                self.nodes.append(node)
                queue.append(leaf_less)
                queue.append(leaf_more)

    def _new_leaf(self, axis, parent, points):
        leaf = KDTree.Leaf(self._nid, axis, parent, points)
        self._nid +=1
        return leaf

    def _split_points(self, pt_idx, axis):
        pts_ax = self.points[pt_idx,axis] # 1D array of the considered coordinate to split 
        pivot = self._find_pivot(pts_ax)
        pivot_filter = pts_ax <= pivot
        idx_less = np.extract(pivot_filter, pt_idx)
        idx_more = np.extract(~pivot_filter, pt_idx)
        return pivot, idx_less, idx_more
    
    def _find_pivot(self, pts_ax):
        if self.build_strategy == KDTree.BuildStrategy.FAST:
            samples = np.random.choice(pts_ax, min(50, pts_ax.size), replace=False)
            return np.median(samples)
        elif self.build_strategy == KDTree.BuildStrategy.BALANCED:
            return np.median(pts_ax)
        elif self.build_strategy == KDTree.BuildStrategy.RANDOM:
            return np.random.choice(pts_ax, 1)[0]
        else:
            raise Exception("Should not happen.")

    def is_leaf(self, node_id:int) -> bool:
        return isinstance(self.nodes[node_id], KDTree.Leaf)

    def query(self, pt:Vec, k:int=1) -> list:
        """Query the tree to find the k nearest neighbors of point 'pt'

        Args:
            pt (Vec): the position to query
            k (int, optional): number of nearest points to query. Defaults to 1.

        Returns:
            list: the indices of the k nearest points of 'pt' in increasing distance order
        """
        found = PriorityQueue()
        n_found = 0
        queue = deque()
        queue.append(0) # start from the root
        while len(queue)>0:
            node_id = queue.pop() # stack order for depth first search
            if self.is_leaf(node_id):
                leaf = self.nodes[node_id]
                for idx in leaf.points:
                    found.push(idx, -distance(self.points[idx], pt))
                    n_found += 1
                    while n_found>k:
                        # keep only k closest
                        found.pop() 
                        n_found -= 1
            else:
                node = self.nodes[node_id]
                furthest_so_far = -found.front.priority if not found.empty() else float("inf")
                dist_left = self.nodes[node.left].bb.distance(pt)
                dist_right = self.nodes[node.right].bb.distance(pt)                
                for dist,child in sorted([(dist_left, node.left), (dist_right,node.right)]):
                    if furthest_so_far > dist:
                        queue.append(child)
        return [found.pop().x for _ in range(n_found)][::-1]
    
    def query_radius(self, pt:Vec, r:float) -> list:
        """Query the tree to find all points that are at distance <= r from the position 'pt'.

        Args:
            pt (Vec): the position to query
            r (float): radius of the query ball

        Returns:
            list: all indices of points that are at distance at most r from the query point.
        """
        queue = deque()
        found_pt = []
        queue.append(0)
        while len(queue)>0:
            node_id = queue.popleft()
            if self.nodes[node_id].bb.distance(pt)>r : 
                continue # no point inside the node that have distance < r
            if self.is_leaf(node_id): # read leaf
                leaf = self.nodes[node_id]
                found_pt += [idx for idx in leaf.points if distance(self.points[idx],pt)<=r]
            else: # read node
                node = self.nodes[node_id]
                queue.append(node.left)
                queue.append(node.right)
        return found_pt  
