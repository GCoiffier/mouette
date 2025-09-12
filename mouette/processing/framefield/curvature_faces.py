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
        features : bool = False,
        verbose : bool = False,
        confidence_threshold : float = 0.5,
        smooth_threshold : float = 0.7,
        **kwargs):
        super().__init__(supporting_mesh, 4, features, verbose, **kwargs)
        self.curv_mat_faces : np.ndarray = None # shape (|T|,3,3)
        self.confidence_threshold = confidence_threshold
        self.smooth_threshold = smooth_threshold

    def _initialize_curvature(self):
        curv_mat_edges = curvature_matrices(self.mesh)
        adj_e = dict([(e,set()) for e in self.mesh.id_edges])
        for T in self.mesh.id_faces:
            for v in self.mesh.faces[T] :
                for T2 in self.mesh.connectivity.vertex_to_faces(v):
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
        self.log("Computing curvature via SVD")
        confidence = self.mesh.faces.create_attribute("confidence", float, dense=True)
        eigs = self.mesh.faces.create_attribute("eigs", float, 3, dense=True)
        for T in self.mesh.id_faces:
            if abs(self.var[T])!=0: continue # already initialized => T is on the boundary
            U,S,V = np.linalg.svd(self.curv_mat_faces[T,:,:], hermitian=True)
            # three vectors -> normal and two principal components
            # eigenvalue of normal is 0
            # PC are orthogonal (eigenvects of symmetric matrix) -> we rely on the eigenvect of greatest eigenvalue and take orthog direction
            eigs[T] = S
            if S[0]<0.01 : # zero matrix -> no curvature information -> we take the first basis vector and hope to smooth correctly
                self.var[T] = 0 + 0j
                confidence[T] = 0.
                continue

            confidence[T] = (S[0] - S[1])/S[0]
            if confidence[T]<self.confidence_threshold : # zero matrix -> no curvature information
                self.var[T] = 0 + 0j
            else:
                # representation vector
                eig = V[0,:]
                c = self.conn.project(eig, T)
                c = complex(c[0], c[1])
                self.var[T] = (c/abs(c))**4

        lap, A = None, None
        if self.confidence_threshold>0:
            self.log("Completing curvature on flat regions")
            lap = operators.laplacian_triangles(self.mesh, order=4, connection=self.conn).tocsc()
            A = operators.area_weight_matrix_faces(self.mesh)
            fixedInds,freeInds = [],[]
            for T in self.mesh.id_faces:
                near_feature = any([e in self.feat.feature_edges for e in self.mesh.connectivity.face_to_edges(T)])
                if confidence[T] < self.confidence_threshold and not near_feature:
                    freeInds.append(T)
                else:
                    fixedInds.append(T)
            if freeInds:
                lapI = lap[freeInds,:][:,freeInds]
                lapB = lap[freeInds,:][:,fixedInds]
                # for lapI and lapB, only lines of freeInds are relevant : lines of fixedInds link fixed variables -> useless constraints
                valB = lapB.dot(self.var[fixedInds]) # right hand side
                res = linalg.spsolve(lapI, - valB)
                self.var[freeInds] = res
        
        if self.n_smooth > 0:
            self.log(f"Smoothing frame field {self.n_smooth} times with diffusion.")
            # diffuse the curvature results to get a smoother results (especially where curvature was not defined)
            if lap is None:
                lap = operators.laplacian_triangles(self.mesh, order=4, connection=self.conn).tocsc()
                A = operators.area_weight_matrix_faces(self.mesh)
            self.log(f"Solve linear system {self.n_smooth} times with diffusion")
            alpha = self.smooth_attach_weight or self._compute_attach_weight(A)
            A = A.astype(complex)
            self.log("Attach weight: {}".format(alpha))
            
            fixedInds = set()
            for ie in self.feat.feature_edges:
                u,v = self.mesh.edges[ie]
                T1,T2 = self.mesh.connectivity.edge_to_faces(u,v)
                if T1 is not None: fixedInds.add(T1)
                if T2 is not None: fixedInds.add(T2)
            for T in self.mesh.id_faces:
                if confidence[T]>self.smooth_threshold:
                    fixedInds.add(T)
            freeInds = set(range(len(self.mesh.id_faces))) - fixedInds
            fixedInds, freeInds = list(fixedInds), list(freeInds)
            
            if len(fixedInds)>0:
                lapI = lap[freeInds,:][:,freeInds]
                lapB = lap[freeInds,:][:,fixedInds]
                AI = A[freeInds, :][:,freeInds]
                # for lapI and lapB, only lines of freeInds are relevant : lines of fixedInds link fixed variables -> useless constraints
                valB = lapB.dot(self.var[fixedInds]) # right hand side
                mat = lapI -  alpha * AI
                solve = linalg.factorized(mat)
                for _ in range(self.n_smooth):
                    valI2 = alpha * AI.dot(self.var[freeInds])
                    res = solve(- valB - valI2)
                    self.var[freeInds] = res
                    self.normalize()
            else:
                mat = lap - alpha * A
                solve = linalg.factorized(mat)
                for _ in range(self.n_smooth):
                    valI2 = alpha * A.dot(self.var)
                    self.var = solve(-valI2)
                    self.normalize()
        self.smoothed = True