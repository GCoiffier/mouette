import numpy as np
import numba as nb
from numba import njit, prange
from collections import deque
from .. import geometry as geom
from enum import Enum

@njit(inline="always", fast_math=True)
def distance2_to_aabb(q, node_min, node_max):
    """Distance from point q to the axis-aligned bounding box defined by node_min and node_max positions."""
    d = np.zeros_like(q)
    for i in range(d.shape[0]):
        d[i] = max(node_min[i] - q[i], 0., q[i] - node_max[i])
    return np.dot(d,d)

@njit(inline="always", fast_math=True)
def distance2_to_triangle(p, a, b, c):
    """Squared distance from point p to triangle (a,b,c) in 3D, and the closest point on the triangle.
    Voronoi region test from Ericson, Real-Time Collision Detection, section 5.1.5."""
    ab, ac, ap = b - a, c - a, p - a
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    if nx * nx + ny * ny + nz * nz <= 1e-12 * np.dot(ab, ab) * np.dot(ac, ac):
        # degenerate (zero area) triangle: region tests are unreliable, take the closest of the three edges
        q = geom.closest_point_on_segment(p, a, b)
        q2 = geom.closest_point_on_segment(p, b, c)
        if geom.distance2(p,q2) < geom.distance2(p, q): q = q2
        q2 = geom.closest_point_on_segment(p, a, c)
        if geom.distance2(p, q2) < geom.distance2(p, q): q = q2
        return geom.distance2(p,q), q
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0. and d2 <= 0.: # vertex region a
        q = a
    else:
        bp, cp = p - b, p - c
        d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
        d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        if d3 >= 0. and d4 <= d3: # vertex region b
            q = b
        elif d6 >= 0. and d5 <= d6: # vertex region c
            q = c
        # denominators below are squared edge lengths and squared double area, all nonzero since the triangle is not degenerate
        elif vc <= 0. and d1 >= 0. and d3 <= 0.: # edge region ab
            q = a + (d1 / (d1 - d3)) * ab
        elif vb <= 0. and d2 >= 0. and d6 <= 0.: # edge region ac
            q = a + (d2 / (d2 - d6)) * ac
        elif va <= 0. and d4 >= d3 and d5 >= d6: # edge region bc
            q = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
        else: # inside the face
            denom = va + vb + vc
            q = a + (vb / denom) * ab + (vc / denom) * ac
    return geom.distance2(p, q), q


@njit(parallel=True, cache=True)
def _query_bvh_parallel(Q, V, S, perm, node_min, node_max, left, right, start, count, primitive_mode, n_chunks):
    """Query function for the BVH that runs in parallel threads using numba. The query points are split in n_chunks contiguous chunks distributed over the threads by prange.

    This function has to be defined outside of the class since it can only consider numba-compatible arguments.
    """
    M = Q.shape[0]
    dist = np.full(M, fill_value=np.inf)
    closest = np.empty_like(Q)
    for c in prange(n_chunks):
        stack = np.empty(256, dtype=np.int64)  # one traversal stack per chunk, reused by both traversals of all its points
        for query_id in range(c * M // n_chunks, (c + 1) * M // n_chunks):
            pt = Q[query_id]
            stack[0] = 0  # start from the root
            stack_counter = 1
            while stack_counter > 0:
                stack_counter -= 1
                node_id = stack[stack_counter]
                if distance2_to_aabb(pt, node_min[node_id], node_max[node_id]) >= dist[query_id]: continue  # best distance is already better than anything inside this node
                if left[node_id] == -1:  # leaf node: test every element inside
                    for i in range(start[node_id], start[node_id] + count[node_id]):
                        ip = perm[i]
                        if primitive_mode == 0:
                            dist_i = geom.distance2(pt, V[ip])
                            close_i = V[ip]
                        elif primitive_mode == 1:
                            close_i = geom.closest_point_on_segment(pt, V[S[ip, 0]], V[S[ip, 1]])
                            dist_i = geom.distance2(pt, close_i)
                        else:
                            dist_i, close_i = distance2_to_triangle(pt, V[S[ip, 0]], V[S[ip, 1]], V[S[ip, 2]])

                        if dist_i < dist[query_id]:
                            dist[query_id] = dist_i
                            closest[query_id] = close_i
                else:
                    l, r = left[node_id], right[node_id]
                    dl = distance2_to_aabb(pt, node_min[l], node_max[l])
                    dr = distance2_to_aabb(pt, node_min[r], node_max[r])
                    # push the farther child first so that the nearer one is explored first
                    if dl > dr:
                        stack[stack_counter], stack[stack_counter+1] = l, r
                    else:
                        stack[stack_counter], stack[stack_counter+1] = r, l
                    stack_counter+=2
    return np.sqrt(dist), closest



class BoundingVolumeHierarchy:

    class MODE:
        VERTICES = 0
        EDGES = 1
        TRIANGLES = 2

    def __init__(self, 
        vertices: np.ndarray,
        simplices: np.ndarray = None, 
        **kwargs
    ):
        """
        The BVH is dimension-agnostic and works with 3D as well as 2D data.

        Args:
            vertices (np.ndarray): _description_
            simplices (np.ndarray, optional): _description_
           
        Keyword Args:
            leaf_size (int, optional): _description_. Defaults to 10.
            n_workers (int, optional): Number of parallel workers to consider for queries. If unspecified, will be set to the maximum number of numba threads.
        
        Attributes:
            perm (np.ndarray): ...
            bb_pmin (np.ndarray): ...
            bb_pmax (np.ndarray): ...
            left (np.ndarray): ...
            right (np.ndarray): ...
            node_start (np.ndarray): ...
            node_count (np.ndarray): ...
        """
        self.leaf_size = kwargs.get("leaf_size", 10)
        self.n_workers = kwargs.get("n_workers", nb.get_num_threads())
        self.V = np.ascontiguousarray(vertices)
        if simplices is None:
            self.F = np.empty((0,2), dtype=np.int64) # needs to be a 2D int array for numba to type S[ip,0] in the query function
            self.mode = BoundingVolumeHierarchy.MODE.VERTICES
        else:
            self.F = np.ascontiguousarray(simplices)
            if self.F.shape[1] != self.V.shape[1]:
                raise Exception(f"Invalid dimensionality in BVH. Simplices were specified but are of dimension {self.F.shape[1]} while points are of dimension {self.V.shape[1]}. BVH is only defined for triangles in 3D or edges in 2D.")
            if self.F.shape[1] == 2:
                self.mode = BoundingVolumeHierarchy.MODE.EDGES
            elif self.F.shape[1] == 3:
                self.mode = BoundingVolumeHierarchy.MODE.TRIANGLES
            else:
                raise Exception("Received a wrong argument as `simplices`")
        self._build()

    @property
    def n_nodes(self):
        return self.node_count.shape[0]

    def _build(self):
        dim = self.V.shape[1]
        # Build the list of primitive AABBs and centroids
        if self.mode == BoundingVolumeHierarchy.MODE.VERTICES:
            prim_min = prim_max = centroids = self.V
        else:
            PV = self.V[self.F] # shape (nP, n_vertices_per_primitive, dim)
            prim_min, prim_max, centroids = PV.min(axis=1), PV.max(axis=1), PV.mean(axis=1)
        nP = centroids.shape[0]
        self.perm = np.arange(nP) # permutation to apply to primitives.

        ## Initialize containers
        estimated_n_leaves = nP//max(1, int(0.5*self.leaf_size)) # take a margin
        max_nodes = 2*estimated_n_leaves + 1 # if at most K leaves => at most 2*K +1 nodes in total in binary tree
        self.bb_pmin = np.zeros((max_nodes, dim))
        self.bb_pmax = np.zeros((max_nodes, dim))
        self.left = np.full(max_nodes, -1, dtype=np.int64)
        self.right = np.full(max_nodes, -1, dtype=np.int64)
        self.node_start = np.zeros(max_nodes, dtype=np.int64)
        self.node_count = np.zeros(max_nodes, dtype=np.int64)
        n_nodes = 1

        stack = deque() 
        stack.append((0, 0, nP))

        while len(stack)>0 : # while stack is not empty
            # get first stack element
            node_id, i_start, i_count = stack.pop()
            
            # handle the node: store its span and compute its bounding box
            self.node_start[node_id] = i_start
            self.node_count[node_id] = i_count

            elements_of_node = self.perm[i_start: i_start+i_count] # apply permutation to get the correct primitives
            self.bb_pmin[node_id] = prim_min[elements_of_node].min(axis=0)
            self.bb_pmax[node_id] = prim_max[elements_of_node].max(axis=0)

            if i_count <= self.leaf_size:
                # The node is sufficiently small : end its treatment here
                continue

            # Otherwise, the node needs to be split into two children along its largest axis.
            axis = np.argmax(self.bb_pmax[node_id] - self.bb_pmin[node_id])
            elements_of_node = elements_of_node.copy()
            order = np.argsort(centroids[elements_of_node, axis])

            self.perm[i_start:i_start + i_count] = elements_of_node[order]
            half = i_count // 2
            self.left[node_id] = n_nodes
            self.right[node_id] = n_nodes + 1
            # insert left and right child in the stack
            stack.append((n_nodes, i_start, half))
            stack.append((n_nodes + 1, i_start + half, i_count - half))
            n_nodes += 2

        # Strip containers to the exact number of nodes
        self.bb_pmin = self.bb_pmin[:n_nodes]
        self.bb_pmax = self.bb_pmax[:n_nodes]
        self.left = self.left[:n_nodes]
        self.right = self.right[:n_nodes]
        self.node_count = self.node_count[:n_nodes]
        self.node_start = self.node_start[:n_nodes]


    def query(self, x : np.ndarray):
        if len(x.shape)==1: x = x[np.newaxis, :]
        n_chunks = max(1, min(x.shape[0], 64 * self.n_workers))
        return _query_bvh_parallel(x, self.V, self.F, self.perm, self.bb_pmin, self.bb_pmax, self.left, self.right, self.node_start, self.node_count, self.mode, n_chunks)
    