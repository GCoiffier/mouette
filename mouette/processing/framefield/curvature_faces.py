from .faces2d import _BaseFrameField2DFaces
from ...mesh.datatypes import *
from ...attributes import curvature_matrices
from ... import operators
import numpy as np
from scipy.sparse import linalg

class PrincipalDirectionsFaces(_BaseFrameField2DFaces):
    """
    Principal curvature direction estimation using a frame field on faces.

    Implementation based on 'Restricted Delaunay Triangulations and Normal Cycle',  David Cohen-Steiner and Jean-Marie Morvan, 2003
    """

    def __init__(self, 
        supporting_mesh : SurfaceMesh,
        features : bool = True,
        verbose :bool = False,
        **kwargs):
        """
        Principal curvature direction estimation using a frame field on faces.

        Args:
            supporting_mesh (SurfaceMesh): the surface mesh on which to perform the estimation
            verbose (bool, optional): verbose mode. Defaults to False.
        
        Additionnal arguments:
            TODO
        

        Note:
            Order of the frame field is fixed at 4 since principal curvature directions form an orthonormal basis.
        """
        super().__init__(supporting_mesh, 4, features, verbose, **kwargs)
        self.curv_mat_faces : np.ndarray = None # shape (|T|,3,3)

    def _initialize_curvature(self):
        curv_mat_edges = curvature_matrices(self.mesh)
        adj_e = dict([(e,set()) for e in self.mesh.id_edges])
        for T in self.mesh.id_faces:
            for T2 in self.mesh.connectivity.face_to_faces(T):
                for e in self.mesh.connectivity.face_to_edges(T2):
                    adj_e[e].add(T)

        self.curv_mat_faces = np.zeros((len(self.mesh.faces),3,3))
        for e in self.mesh.id_edges:
            for T in adj_e[e]:
                self.curv_mat_faces[T] += curv_mat_edges[e]

    def initialize(self):
        self.log("Initialize attributes")
        self._initialize_attributes() # /!\ may change mesh combinatorics near boundary
        self.log("Compute curvature matrices")
        self._initialize_curvature()
        self.log("Initialize variables")
        self._initialize_variables()
        self.initialized = True

    def optimize(self):
        for T in self.mesh.id_faces:
            if abs(self.var[T])!=0: continue # already initialized => T is on the boundary
            u,s,_ = np.linalg.svd(self.curv_mat_faces[T,:,:], hermitian=True)
            X,Y = self.conn.base(T)
            # three vectors -> normal and two principal components
            # eigenvalue of normal is 0
            # PC are orthogonal (eigenvects of symmetric matrix) -> we rely on the eigenvect of greatest eigenvalue and take orthog direction
            if s[0]<1e-10 : # zero matrix -> no curvature information -> we take the first basis vector and hope to smooth correctly
                self.var[T] = 0 + 0j
            else:
                # representation vector
                v = u[:,0]
                c = complex(X.dot(v), Y.dot(v))
                self.var[T] = (c/abs(c))**4

        if self.n_smooth > 0:
            # diffuse the curvature results to get a smoother results (especially where curvature was not defined)
            lap = operators.laplacian_triangles(self.mesh, order=4, connection=self.conn).tocsc()
            A = operators.area_weight_matrix_faces(self.mesh)
            self.log(f"Solve linear system {self.n_smooth} times with diffusion")
            alpha = self.smooth_attach_weight or self._compute_attach_weight(A, 0.1)
            A = A.astype(complex)
            self.log("Attach weight: {}".format(alpha))
            # TODO : take boundary into account
            # if len(self.feat.feature_vertices)>0: # we have a boundary
            #     raise NotImplementedError
            mat = lap - alpha * A
            solve = linalg.factorized(mat)
            for _ in range(self.n_smooth):
                valI2 = alpha * A.dot(self.var)
                self.var = solve(-valI2)
                self.normalize()
        self.smoothed = True