import numpy as np
from scipy.spatial import KDTree
from enum import Enum

from ..mesh.datatypes import *
from ..mesh.mesh_data import RawMeshData
from ..mesh.mesh_attributes import ArrayAttribute
from .worker import Worker
from .. import geometry as geom
from ..geometry import Vec
from .. import utils

class PointCloudNormalEstimator(Worker):
    """An estimator of normal directions for unstructured point clouds, based on k-nearest neighbors and singular value decomposition of their correlation matrix.


    References:
        [1] Surface reconstruction from unorganized points, Hoppe et al., 1992

    Example:
        [https://github.com/GCoiffier/mouette/blob/main/examples/point_cloud_normal_estimation.py](https://github.com/GCoiffier/mouette/blob/main/examples/point_cloud_normal_estimation.py)
    """

    class OrientStrategy(Enum):
        NONE = 0
        SPANNING_TREE = 1

        @classmethod
        def from_string(cls, txt : str):
            txt = txt.lower()
            if "mst" in txt : return cls.SPANNING_TREE
            return cls.NONE
        

    def __init__(self, 
        n_neighbors : int = 5,
        save_on_pc : bool = True,
        compute_curvature : bool = False,
        orientation_mode : str = "mst",
        verbose = True, 
        **kwargs):
        """
        Args:
            n_neighbors (int, optional): number of neighbor points to consider for local plane fitting. Defaults to 5.
            save_on_pc (bool, optional): whether to store the resulting normals onto the point cloud as a vertex attribute, or as independent arrays. Defaults to True.
            compute_curvature (bool, optional): whether to also compute an estimate of curvature on the point cloud. This curvature estimate is computed as the ratio of the smallest eigenvalue and the sum of eigenvalues of the correlation matrix at each point (between 0 and 1). Defaults to False.
            orientation_mode (str, optional): Heuristic strategy for consistent orientation of normals. Choices are ["None", "mst"] Defaults to "mst".
            verbose (bool, optional): verbose mode. Defaults to True.


        Keyword Args:
            normal_attribute_name (str): The name of the attribute in which normals are stored. Ignored if save_on_pc is set to False. Defaults to "normals".
            curvature_attribute_name (str): The name of the attribute in which the curvature estimation is stored. Ignored if save_on_pc is set to False. Defaults to "curvature".
        """
        super().__init__("PCNormalEstimation", verbose)
        self._n_neighbors : int = n_neighbors
        self._save_on_pc : bool = save_on_pc
        self._orientation_mode = PointCloudNormalEstimator.OrientStrategy.from_string(orientation_mode)
        
        self._normal_attr_name = kwargs.get("normal_attribute_name", "normals")
        self._curvature_attr_name = kwargs.get("curvature_attribute_name", "curvature")
        self._compute_curvature : bool = compute_curvature
        
        self.pc : PointCloud = None
        self.normals : ArrayAttribute = None
        self.curvature : ArrayAttribute = None

    def clear(self):
        self.pc = None
        self.normals = None
        self.curvature = None

    def normals_as_vector_field(self, scale:float = 1.):
        if self.normals is None : return None 
        pl = RawMeshData()
        for i in self.pc.id_vertices:
            P1 = self.pc.vertices[i]
            P2 = self.pc.vertices[i] + scale * self.normals[i]
            pl.vertices += [P1, P2]
            pl.edges.append((2*i,2*i+1))
        return PolyLine(pl)

    @allowed_mesh_types(PointCloud)
    def run(self, point_cloud : PointCloud, **kwargs):

        ### Prepare data and containers
        self.log("Prepare data and containers")
        self.pc = point_cloud
        points = np.asarray(point_cloud.vertices)
        
        kdtree = kwargs.get("kdtree", None)
        if kdtree is not None:
            utils.check_argument("kdtree", kdtree, KDTree)
        else:
            self.log("Compute kd-tree")
            kdtree = KDTree(points)
        _,KNN = kdtree.query(points, self._n_neighbors+1)

        if self._save_on_pc:
            self.normals = point_cloud.vertices.create_attribute(self._normal_attr_name, float, 3, dense=True)
            self.curvature = point_cloud.vertices.create_attribute(self._curvature_attr_name, float, 1, dense=True)
        else:
            self.normals = np.zeros((len(point_cloud.vertices), 3))
            self.curvature = np.zeros(len(point_cloud.vertices))

        ### Run normal estimation
        self.log("Estimate normals with SVD")
        self._estimate_normals_svd(KNN)

        ### Consistently orient the normals
        if self._orientation_mode == PointCloudNormalEstimator.OrientStrategy.SPANNING_TREE:
            self.log("Consistently orient normals (MST strategy)")
            self._correct_normal_orientation_mst(kdtree, KNN)

    def _estimate_normals_svd(self, KNN):
        points = np.asarray(self.pc.vertices)
        for i in self.pc.id_vertices:
            pts_i = points[KNN[i,:],:]
            center = np.mean(pts_i, axis=0)
            mat_i = pts_i-center
            svd = np.linalg.svd(mat_i.T, full_matrices=False)
            self.normals[i] = svd[0][:,-1]
            if self._compute_curvature :
                l0,l1,l2 = svd[1]
                assert l0 >= l1 >= l2
                self.curvature[i] = l2/(l0+l1+l2)

    def _correct_normal_orientation_mst(self, kdtree, KNN):
        ### Build Minimal Spanning tree
        edges = set()
        for i in self.pc.id_vertices:
            for j in KNN[i,:]:
                if j==i: continue
                edges.add(utils.keyify(i,j))
        pr_queue = utils.PriorityQueue()
        for (i,j) in edges:
            pr_queue.push((i,j), 1 - abs(np.dot(self.normals[i], self.normals[j])))

        UF = utils.UnionFind(self.pc.id_vertices)
        adjacency = [[] for _ in self.pc.id_vertices]
        edges_mst = []
        while not pr_queue.empty():
            elem = pr_queue.pop()
            i,j = elem.x
            if UF.find(i) != UF.find(j):
                UF.union(i,j)
                adjacency[i].append(j)
                adjacency[j].append(i)
                edges_mst.append([i,j])

        ### Set starting orientation
        aabb = geom.AABB.of_mesh(self.pc)
        exterior_pt = aabb.center + Vec(0, 2*aabb.span[1], 0)
        _,top_pt = kdtree.query(exterior_pt)
        if np.dot(self.normals[top_pt], exterior_pt - self.pc.vertices[top_pt])<0.:
            self.normals[top_pt] *= -1 # if this normal is not outward, we are doomed
        
        ### Traverse spanning tree
        to_visit = []
        visited = np.zeros(len(self.pc.vertices), dtype=bool)
        for adj_pt in adjacency[top_pt]:
            to_visit.append((top_pt, adj_pt))
        visited[top_pt] = True
        
        while len(to_visit)>0:
            parent, node = to_visit.pop()
            if visited[node] : continue
            visited[node] = True
            Nv = self.normals[node]
            Np = self.normals[parent] # suppose that this one is correctly oriented
            if np.dot(-Nv,Np)>np.dot(Nv,Np):
                self.normals[node] *= -1
            for child in adjacency[node]:
                if not visited[child]:
                    to_visit.append((node,child))