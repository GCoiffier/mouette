from .vertex2d import _BaseFrameField2DVertices
from ...mesh.mesh_attributes import Attribute
from ...mesh.datatypes import *
from ...import attributes
from ... import geometry as geom
from ...utils.maths import *
from ... import operators

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
from numba import jit, prange


@jit(parallel=True, nopython=True)
def aggregate_mats(curvV, curvE, iV,iE, n):
    for i in prange(n):
        curvV[iV[i],:,:] += curvE[iE[i],:,:]
    return curvV

class PrincipalDirectionsVertices(_BaseFrameField2DVertices):
    """
    Principal curvature direction estimation using a frame field on vertices.
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
        mesh : SurfaceMesh, 
        feature_edges : bool = False,
        patch_size : int = 2,
        confidence_threshold : float = 0.5,
        smooth_threshold : float = 0.7,
        verbose : bool = True,
        **kwargs):
        """

        Args:
            mesh (SurfaceMesh): _description_
            feature_edges (bool, optional): _description_. Defaults to False.
            patch_size (int, optional): _description_. Defaults to 2.
            confidence_threshold (float, optional): _description_. Defaults to 0.5.
            smooth_threshold (float, optional): _description_. Defaults to 0.7.
            verbose (bool, optional): _description_. Defaults to True.
        
        Keyword Args:
            n_smooth (int, optional): Number of smoothing steps to perform. Defaults to 3.
            
            smooth_attach_weight (float, optional): Custom attach weight to previous solution during smoothing steps. 
                If not provided, will be estimated automatically during optimization. Defaults to None.

            use_cotan (bool, optional): whether to use cotan for a better approximation of the Laplace-Beltrami operator. 
                If False, will use a simple adjacency laplacian operator (See the _operators_ module). Defaults to True.

            smooth_normals : Whether to initialize the frame field as a mean of adjacent feature edges (True), or following one of the edges (False). has no effect for frame field on faces. Defaults to True.
            
            custom_connection (SurfaceConnection, optional): custom connection object to be used for parallel transport. If not provided, a connection will be automatically computed (see SurfaceConnection class). Defaults to None.
            
            custom_feature (FeatureEdgeDetector, optional): custom feature edges to be used in frame field optimization. If not provided, feature edges will be automatically detected. If the 'features' flag is set to False, features of this object are ignored. Defaults to None.
        """
        super().__init__(mesh, 
            4, # order is always 4 
            feature_edges,
            verbose,
            **kwargs
        )
        self.patch_size : int = max(1, int(patch_size)) # 

        self.confidence_threshold : float = confidence_threshold
        self.smooth_threshold : float = smooth_threshold

        self.curv_mat_vert : np.ndarray = None # shape (|V|,3,3)
        self.M : sp.lil_matrix = None

    def run(self):
        self.log("Compute curvature matrices")
        self.initialize()
        self.log("Optimize")
        self.optimize()
        self.log("Done.")
        return self

    def _build_patch_connectivity_matrix(self):
        self.log("Compute vertex neighborhoods")
        V2E = operators.vertex_to_edge_operator(self.mesh)
        self.M = operators.adjacency_matrix(self.mesh).tolil()
        self.M.setdiag(1) # adjacency + Id
        self.M = pow(self.M, self.patch_size-1) @ V2E
        self.M.minimum(1)

    def _initialize_curv_matrices(self):
        self.log("Aggregate curvature matrices on patches")
        curv_mat_edges = attributes.curvature_matrices(self.mesh)
        self.curv_mat_vert = np.zeros((len(self.mesh.vertices),3,3), dtype=np.float64)
        row,col = self.M.nonzero()
        self.curv_mat_vert = aggregate_mats(self.curv_mat_vert, curv_mat_edges, row, col, self.M.count_nonzero())

    def initialize(self):
        super()._initialize_attributes()
        self._build_patch_connectivity_matrix()
        self._initialize_curv_matrices()
        self._initialize_variables()
        self.initialized = True

    def optimize(self):
        confidence = self.mesh.vertices.create_attribute("confidence", float, dense=True)
        # to_complete = self.mesh.vertices.create_attribute("to_complete", bool, dense=True)
        # to_smooth = self.mesh.vertices.create_attribute("to_smooth", bool, dense=True)

        self.log("Computing curvature via SVD")
        for v in self.mesh.id_vertices:
            if v in self.feat.feature_vertices: continue # do not override frames on boundary
                        
            U,S,V = np.linalg.svd(self.curv_mat_vert[v,:,:], hermitian=True)
            # three vectors -> normal and two principal components
            # eigenvalue of normal is 0
            # PC are orthogonal (eigenvects of symmetric matrix) -> we rely on the eigenvect of greatest eigenvalue and take orthog direction
            if S[0]<1e-8: # the patch is perfectly flat
                self.var[v] = 0 + 0j
                confidence[v] = 0.
                continue

            confidence[v] = (S[0] - S[1])/S[0]
            # to_complete[v] = confidence[v]<self.confidence_threshold # DEBUG
            if confidence[v]<self.confidence_threshold : # zero matrix -> no curvature information
                self.var[v] = 0 + 0j
            else:
                # representation vector
                eig = V[0,:]
                c = self.conn.project(eig, v)
                c = complex(c[0], c[1])
                self.var[v] = (c/abs(c))**4

        lap, A = None, None
        if self.confidence_threshold>0:
            self.log("Completing curvature on flat regions")
            lap = operators.laplacian(self.mesh, cotan=self.use_cotan, connection=self.conn, order=4)
            A = operators.area_weight_matrix(self.mesh)
            fixedInds,freeInds = [],[]
            for v in self.mesh.id_vertices:
                if confidence[v] < self.confidence_threshold:
                    freeInds.append(v)
                else:
                    fixedInds.append(v)
            if freeInds:
                lapI = lap[freeInds,:][:,freeInds]
                lapB = lap[freeInds,:][:,fixedInds]
                # for lapI and lapB, only lines of freeInds are relevant : lines of fixedInds link fixed variables -> useless constraints
                valB = lapB.dot(self.var[fixedInds]) # right hand side
                res = linalg.spsolve(lapI, - valB)
                self.var[freeInds] = res

        self.normalize()
        if self.n_smooth > 0:
            self.log(f"Smoothing frame field {self.n_smooth} times with diffusion.")
            # diffuse the curvature results to get a smoother results
            if lap is None:
                lap = operators.laplacian(self.mesh, cotan=self.use_cotan, connection=self.conn, order=4)
                A = operators.area_weight_matrix(self.mesh)
            alpha = self.smooth_attach_weight or 1.
            self.log("Attach Weight", alpha)

            fixedInds = set()
            if len(self.feat.feature_vertices)>0:
                # Build fixed/free vertex partition
                for v in self.mesh.id_vertices:
                    if v in self.feat.feature_vertices:
                        fixedInds.add(v)
            for v in self.mesh.id_vertices:
                if confidence[v]>self.smooth_threshold:
                    fixedInds.add(v)
            freeInds = set(range(len(self.mesh.id_vertices))) - fixedInds
            fixedInds, freeInds = list(fixedInds), list(freeInds)
            # for v in freeInds: to_smooth[v] = True # DEBUG
            
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