import numpy as np
import scipy.sparse as sp
from tqdm import trange

from .worker import Worker
from ..mesh.datatypes import *
from ..mesh.mesh_attributes import ArrayAttribute
from ..operators import adjacency_matrix
from .. import geometry as geom
from ..geometry import Vec
from .. import attributes

class ShapeOperator(Worker):

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, verbose: bool = True):
        """Computes the shape operator at each vertex of a triangle mesh. The method is based on least-square fitting of a second order polynomial to a neighborhood of points near each vertex. The size of this neighbordhood can be specified in the `.run()` method to vary the amount of smoothing.

        Args:
            mesh (SurfaceMesh): The input surface mesh. Should be triangulated.
            verbose (bool, optional): Verbose mode. Defaults to True.

        Attributes:
            shape_op (np.ndarray): a Vx2x2 array containing the 2x2 shape operator matrix for each vertex
        
        References:
            [1] Estimating Differential Quantities Using Polynomial Fitting of Osculating Jets, F.Cazals & M.Pouget (2003)
            [2] https://github.com/alecjacobson/geometry-processing-curvature
        """
        super().__init__("ShapeOperator", verbose)

        self.mesh : SurfaceMesh = mesh
        assert self.mesh.is_triangular()
        self.shape_op : np.ndarray = None
        self._gaussian_curv : ArrayAttribute = None
        self._mean_curv : ArrayAttribute = None
        self._patches : list = None
        self._varea : ArrayAttribute = None
        self.vertex_normals : ArrayAttribute  = attributes.vertex_normals(self.mesh, persistent=False)

    def run(self, patch_size : int = 3):
        """Runs the computation of the shape operator for each vertex.

        Args:
            patch_size (int, optional): max number of edges that separate the central vertex to a considered neighbor. Higher values mean that the polynomial approximation is fitted onto a larger neighborhood of the point, providing more stability but smoothing out the result. Defaults to 3.
        """
                
        # build adjacency matrix to recover patches
        self._patches = [[] for _ in range(len(self.mesh.vertices))]
        adj = adjacency_matrix(self.mesh)
        adj.setdiag(1) # adjacency + Id
        if patch_size>1:
            adj = pow(adj, patch_size-1)
        adj = adj.tocoo()
        for i in range(adj.row.shape[0]):
            self._patches[adj.row[i]].append(adj.col[i])

        # initialize resulting array and iterate over vertices
        self.shape_op = np.zeros((len(self.mesh.vertices), 2, 2))
        itr = trange(len(self.mesh.vertices)) if self.verbose else self.mesh.id_vertices
        for v in itr:
            self.shape_op[v] = self._handle_vertex(v)

    def __getitem__(self, key):
        if self.shape_op is None: return None
        return self.shape_op[key, :, :]

    def _handle_vertex(self, id_v):
        ### Find local best tangent plane
        P = self.mesh.vertices[id_v]
        points = np.asarray([self.mesh.vertices[_v] for _v in self._patches[id_v]])
        center = np.mean(points, axis=0)
        mat = points-center
        svd = np.linalg.svd(mat.T, full_matrices=False)
        
        ### Build local coordinate system
        N = svd[0][:, -1]
        if np.dot(N, self.vertex_normals[id_v])<0: N *= -1
        X = P-center
        X = Vec.normalized(X - np.dot(X, N)*N)
        Y = geom.cross(X,N)
        
        # coordinates of all points in this basis:
        U = np.sum(X*(points - P), axis=1)
        V = np.sum(Y*(points - P), axis=1)
        W = np.sum(N*(points - P), axis=1)

        ### Fit best second order approximation
        # min the sum of (P(u,v) - w)^2
        mat = np.vstack((U, V, U**2, U*V, V**2)).T
        a1,a2,a3,a4,a5 = np.linalg.lstsq(mat, W)[0]
        
        ### Compute shape operator from fitted jet
        E = 1 + a1*a1
        F = a1*a2
        G = 1 + a2*a2
        sqr = np.sqrt(a1*a1+a2*a2+1)
        e = 2*a3/sqr
        f = a4/sqr
        g = 2*a5/sqr
        S1 = np.array([[e, f],[f,g]])
        S2 = np.linalg.inv(np.array([[E,F],[F,G]]))
        return -S1@S2


    @property
    def vertex_area(self):
        if self._varea is None:
            f_area = attributes.face_area(self.mesh, persistent=False)
            self._varea = ArrayAttribute(float, len(self.mesh.vertices))
            attributes.interpolate_faces_to_vertices(self.mesh, f_area, self._varea)
        return self._varea
    
    @property
    def gaussian_curvature(self) -> ArrayAttribute:
        """Gaussian curvature estimator at each vertex. Defined as the determinant of the shape operator. The values are weighted by the local areas around each vertex.

        Returns:
            ArrayAttribute: Gaussian curvature as one float per vertex
        """
        if self.shape_op is None: return None
        if self._gaussian_curv is None:
            self._gaussian_curv = self.mesh.vertices.create_attribute("gaussian_curvature", float, dense=True)
            for v in self.mesh.id_vertices:
                self._gaussian_curv[v] = np.linalg.det(self.shape_op[v,:,:])*self.vertex_area[v]/3
        return self._gaussian_curv

    @property
    def mean_curvature(self) -> ArrayAttribute:
        """Mean curvature estimator at each vertex. Defined as half the trace of the shape operator.

        Returns:
            ArrayAttribute: Mean curvature as one float per vertex.
        """
        if self.shape_op is None: return None
        if self._mean_curv is None:
            self._mean_curv = self.mesh.vertices.create_attribute("mean_curvature", float, dense=True)
            for v in self.mesh.id_vertices:
                self._mean_curv[v] = np.linalg.trace(self.shape_op[v,:,:])/2
        return self._mean_curv