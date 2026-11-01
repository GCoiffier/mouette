import numpy as np
import numba as nb
import warnings
from numba import njit, prange
from collections import deque
from .. import geometry as geom

@njit(inline="always", fastmath=True)
def distance2_to_aabb(q, node_min, node_max):
    """Distance from point q to the axis-aligned bounding box defined by node_min and node_max positions."""
    d = np.zeros_like(q)
    for i in range(d.shape[0]):
        d[i] = max(node_min[i] - q[i], 0., q[i] - node_max[i])
    return np.dot(d,d)

@njit(inline="always", cache=True)
def _query_bvh_distance(pt, V, S, perm, node_min, node_max, left, right, start, count, primitive_mode, stack):
    """Finds the closest primitive to point pt.
    
    Returns:
        float: the distance from pt to the closest primitive
        Vec: the closest point
        int: index of the closest primitive
        int: feature of the closest primitive on which the closest point lies (see geom.closest_point_on_segment and geom.closest_point_on_triangle). Always 0 for points.
    """
    lowest_dist = np.inf
    closest_pt = np.empty_like(pt)
    closest_prim, closest_feature = -1, 0
    stack[0] = 0  # start from the root
    stack_counter = 1
    while stack_counter > 0:
        stack_counter -= 1
        node_id = stack[stack_counter]
        if distance2_to_aabb(pt, node_min[node_id], node_max[node_id]) >= lowest_dist: continue  # best distance is already better than anything inside this node
        if left[node_id] == -1:  # leaf node: test every element inside
            for i in range(start[node_id], start[node_id] + count[node_id]):
                ip = perm[i]
                if primitive_mode == 0:
                    close_i, feature_i = V[ip].copy(), 0
                elif primitive_mode == 1:
                    close_i, feature_i = geom.closest_point_on_segment(pt, V[S[ip, 0]], V[S[ip, 1]])
                else:
                    close_i, feature_i = geom.closest_point_on_triangle(pt, V[S[ip, 0]], V[S[ip, 1]], V[S[ip, 2]])
                dist_i = geom.distance2(pt, close_i)
                if dist_i < lowest_dist:
                    lowest_dist = dist_i
                    closest_pt = close_i
                    closest_prim, closest_feature = ip, feature_i
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
    return np.sqrt(lowest_dist), closest_pt, closest_prim, closest_feature


@njit(inline="always")
def _pseudonormal(ip, feature, S, FN, VN, EN, primitive_mode):
    """Pseudonormal of the feature of primitive ip on which a closest point lies (see geom.closest_point_on_segment and geom.closest_point_on_triangle)."""
    if feature == 0: return FN[ip] # interior of the face (triangles) or of the edge (segments)
    if primitive_mode == 2 and feature >= 4: return EN[ip, feature - 4] # interior of an edge of a triangle
    return VN[S[ip, feature - 1]] # a vertex


@njit(parallel=True, cache=True)
def _query_bvh_parallel(Q, V, S, perm, node_min, node_max, left, right, start, count, primitive_mode, oriented, FN, VN, EN, n_chunks):
    """Query function for the BVH that runs in parallel threads using numba. The query points are split in n_chunks contiguous chunks distributed over the threads by prange.

    This function has to be defined outside of the class since it can only consider numba-compatible arguments.
    """
    M = Q.shape[0]
    dist = np.full(M, fill_value=np.inf)
    closest = np.empty_like(Q)
    for c in prange(n_chunks):
        stack = np.empty(256, dtype=np.int64)  # one traversal stack per chunk, reused for all its points
        for query_id in range(c * M // n_chunks, (c + 1) * M // n_chunks):
            pt = Q[query_id]
            d, q, ip, feature = _query_bvh_distance(pt, V, S, perm, node_min, node_max, left, right, start, count, primitive_mode, stack)
            if oriented and np.dot(pt - q, _pseudonormal(ip, feature, S, FN, VN, EN, primitive_mode)) < 0.:
                d = -d # pt is on the opposite side of the pseudonormal
            dist[query_id] = d
            closest[query_id] = q
    return dist, closest


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
        Bounding Volume Hierarchy for closest point and distance queries to a set of points, of edges in 2D or of triangles in 3D.
        The BVH is dimension-agnostic and works with 3D as well as 2D data.

        Args:
            vertices (np.ndarray): array of shape (n_vertices, dim) of vertex positions.
            simplices (np.ndarray, optional): array of shape (n_simplices, dim) of vertex indices: edges in 2D or triangles in 3D. If not provided, the BVH is built over the vertices only. Defaults to None.
           
        Keyword Args:
            leaf_size (int, optional): Maximal number of primitives stored in a leaf of the hierarchy. Defaults to 10.
            n_workers (int, optional): Number of parallel workers to consider for queries. If unspecified, will be set to the maximum number of numba threads.
        
        Attributes:
            perm (np.ndarray): permutation of the primitives such that the primitives of each node are contiguous.
            bb_pmin (np.ndarray): minimal corner of the bounding box of each node.
            bb_pmax (np.ndarray): maximal corner of the bounding box of each node.
            left (np.ndarray): index of the left child of each node (-1 for leaves).
            right (np.ndarray): index of the right child of each node (-1 for leaves).
            node_start (np.ndarray): index in `perm` of the first primitive of each node.
            node_count (np.ndarray): number of primitives in each node.

        Raises:
            Exception: if the simplices do not have as many vertices as the dimension of the points.
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
        dim = self.V.shape[1]

        self._normals_computed = False
        # empty arrays so that the numba query function keeps a unique signature
        self.FN = np.empty((0, dim))
        self.VN = np.empty((0, dim))
        self.EN = np.empty((0, 3, dim))

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

    def is_node_leaf(self, node_id) -> bool:
        return self.left[node_id] == -1

    def _compute_pseudonormals(self):
        """Computes the pseudonormals of the simplices (Bærentzen and Aanæs, Signed distance computation using the angle weighted pseudonormal, 2005):
        - FN: unit normal of each simplex (normal of the triangle in 3D, normal of the edge in 2D).
        - VN: pseudonormal of each vertex, as the sum of the normals of the adjacent simplices weighted by their angle at the vertex (in 3D) or unweighted (in 2D).
        - EN: pseudonormal of the three edges of each triangle (AB, BC, CA), as the sum of the normals of its adjacent triangles (3D only).
        Pseudonormals other than FN are not normalized since only their direction matters.
        """
        V, F = self.V, self.F
        dim = V.shape[1]
        self.VN = np.zeros((V.shape[0], dim))
        if self.mode == BoundingVolumeHierarchy.MODE.EDGES:
            E = V[F[:, 1]] - V[F[:, 0]]
            self.FN = np.stack((E[:, 1], -E[:, 0]), axis=1) # right-hand side: outward for a counter-clockwise boundary
            self.FN /= np.maximum(np.linalg.norm(self.FN, axis=1), 1e-30)[:, None]
            np.add.at(self.VN, F[:, 0], self.FN)
            np.add.at(self.VN, F[:, 1], self.FN)
            self.EN = np.empty((0, 3, dim))
            return

        # Triangles
        P = V[F] # (nF, 3, 3)
        self.FN = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        self.FN /= np.maximum(np.linalg.norm(self.FN, axis=1), 1e-30)[:, None]
        # vertex pseudonormals: weighted by the angles at the corners
        for k in range(3):
            u = P[:, (k+1)%3] - P[:, k]
            v = P[:, (k+2)%3] - P[:, k]
            cos = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-30)
            angles = np.arccos(np.clip(cos, -1., 1.))
            np.add.at(self.VN, F[:, k], angles[:, None] * self.FN)
        # edge pseudonormals: sum of the normals of the (one or two) adjacent triangles
        corner_edges = np.stack((F, np.roll(F, -1, axis=1)), axis=2).reshape(-1, 2) # edges AB, BC, CA of each triangle, shape (3*nF, 2)
        _, edge_id = np.unique(np.sort(corner_edges, axis=1), axis=0, return_inverse=True)
        edge_id = edge_id.reshape(-1)
        edge_normals = np.zeros((edge_id.max() + 1, 3))
        np.add.at(edge_normals, edge_id, np.repeat(self.FN, 3, axis=0))
        self.EN = np.ascontiguousarray(edge_normals[edge_id].reshape(-1, 3, 3))
        self._normals_computed = True

    def query_distance(self, x : np.ndarray, oriented: bool = False):
        """Computes the distance from query points to the closest primitive of the BVH, and the corresponding closest points.

        Args:
            x (np.ndarray): query points, either a single point of shape (dim,) or an array of shape (n_queries, dim).
            oriented (bool, optional): Whether to orient the distance using the angle-weighted pseudonormal of the closest primitive feature (face, edge or vertex). 
                The distance is then negative for points lying on the opposite side of the normals, which is the inside of the shape for a closed mesh whose faces are consistently oriented outward 
                (in 2D, for closed polylines whose edges run counter-clockwise around the interior). The orientation of the input is trusted: the sign is meaningless for inconsistently oriented meshes.
                The pseudonormals are computed on the fly during the first query.
                Defaults to False.

        Returns:
            np.ndarray: array of shape (n_queries,) of distances.
            np.ndarray: array of shape (n_queries, dim) of closest points.
        """
        if len(x.shape)==1: x = x[np.newaxis, :]
        oriented = oriented and self.mode != BoundingVolumeHierarchy.MODE.VERTICES
        if oriented and not self._normals_computed:
            self._compute_pseudonormals()
        n_chunks = max(1, min(x.shape[0], 64 * self.n_workers))
        return _query_bvh_parallel(x, self.V, self.F, self.perm, self.bb_pmin, self.bb_pmax, self.left, self.right, self.node_start, self.node_count, self.mode, oriented, self.FN, self.VN, self.EN, n_chunks)
