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
def aggregate_mats(curvV, curvE, areas, iV,iE, n):
    nvert = curvV.shape[0]
    total_area = np.zeros(nvert, dtype=np.float64)
    for i in prange(n):
        v,e = iV[i], iE[i]
        curvV[v,:,:] += areas[e]*curvE[e,:,:]
        total_area[v] += areas[e]
    for v in prange(nvert):
        curvV[v,:,:] /= total_area[v]
    return curvV

# version without numba
# def aggregate_mats(curvV, curvE, areas, iV,iE, n):
#     nvert = curvV.shape[0]
#     total_area = np.zeros(nvert, dtype=np.float64)
#     for i in range(n):
#         v,e = iV[i], iE[i]
#         curvV[v,:,:] += areas[e]*curvE[e,:,:]
#         total_area[v] += areas[e]
#     for v in range(nvert):
#         curvV[v,:,:] /= total_area[v]
#     return curvV

class PrincipalDirectionsVertices(_BaseFrameField2DVertices):
    """
    Principal curvature direction estimation using a frame field on vertices.

    Implementation based on 'Restricted Delaunay Triangulations and Normal Cycle',  David Cohen-Steiner and Jean-Marie Morvan, 2003
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
        mesh : SurfaceMesh, 
        feature_edges : bool = False,
        verbose : bool = True,
        **kwargs):
        """
        Parameters:
            mesh (SurfaceMesh): the surface mesh on which to perform the estimation
            feature_edges (bool, optional): Flag to take into account feature edges. Defaults to False.
            patch_size (int, optional): Neighborhood size in curvature estimation. Greater means more averaging but also more computation. Defaults to 3.
            curv_threshold (float, optional): Minimal mean curvature value for the directions to be computed. Defaults to 0.01.
            verbose (bool, optional): verbose mode. Defaults to True.
        
        Additionnal Parameters:
            TODO
        
        Note:
            Order of the frame field is fixed at 4 since principal curvature directions form an orthonormal basis.
        """
        
        super().__init__(mesh, 
            4, # order is always 4 
            feature_edges,
            verbose,
            **kwargs
        )
        self.complete_ff : bool = kwargs.get("complete_ff", True)
        self.patch_size : int = kwargs.get("patch_size", 3)
        self.curv_threshold : float = kwargs.get("curv_threshold", 0.01)
        self.face_areas : Attribute = None
        self.curv_mat_vert : np.ndarray = None # shape (|V|,3,3)
        self.M : sp.lil_matrix = None

    def run(self):
        self.log("Compute curvature matrices")
        self.initialize()
        self.log("Optimize")
        self.optimize()
        self.log("Done.")
        return self

    def _initialize_attributes(self):
        super()._initialize_attributes()
        self.face_areas = attributes.face_area(self.mesh)

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
        areas = np.zeros(len(self.mesh.edges), dtype=np.float64)
        for e,(a,b) in enumerate(self.mesh.edges):
            areas[e] = sum([self.face_areas[_T] for _T in self.mesh.connectivity.edge_to_faces(a,b) if _T is not None])/2
        self.curv_mat_vert = aggregate_mats(self.curv_mat_vert, curv_mat_edges, areas, row, col, self.M.count_nonzero())

    def initialize(self):
        self._initialize_attributes()
        self._build_patch_connectivity_matrix()
        self._initialize_curv_matrices()
        self._initialize_variables()
        self.initialized = True

    def optimize(self):
        curvW = self.mesh.vertices.create_attribute("curvCoeff", float, dense=True)

        self.log("Computing curvature via SVD")
        for v in self.mesh.id_vertices:
            if v in self.feat.feature_vertices: continue # do not override frames on boundary
            
            U,S,V = np.linalg.svd(self.curv_mat_vert[v,:,:], hermitian=True)
            # three vectors -> normal and two principal components
            # eigenvalue of normal is 0
            # PC are orthogonal (eigenvects of symmetric matrix) -> we rely on the eigenvect of greatest eigenvalue and take orthog direction

            X,Y = self.conn.base(v)
            curv_coeff_v = geom.norm(S)
            if self.n_smooth > 0: curvW[v] = curv_coeff_v

            if curv_coeff_v<self.curv_threshold : # zero matrix -> no curvature information
                self.var[v] = 0 + 0j
            else:
                # representation vector
                eig = V[0,:]
                c = complex(X.dot(eig), Y.dot(eig))
                self.var[v] = (c/abs(c))**4
        
        lap = None
        if self.complete_ff and self.curv_threshold>0:
            lap = operators.laplacian(self.mesh, cotan=self.use_cotan, connection=self.conn, order=4)
            self.log("Completing curvature on flat regions")
            freeAttr = self.mesh.vertices.create_attribute("free", bool)
            A = operators.area_weight_matrix(self.mesh)
            fixedInds,freeInds = [],[]
            for v in self.mesh.id_vertices:
                if abs(self.var[v])>1e-8:
                    fixedInds.append(v)
                else:
                    freeAttr[v] = True
                    freeInds.append(v)
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
                lap = operators.laplacian(self.mesh, cotan=self.use_cotan, connection=self.conn, order=4)
                A = operators.area_weight_matrix(self.mesh)
            alpha = self.smooth_attach_weight or 1.
            self.log("Attach Weight", alpha)

            if len(self.feat.feature_vertices)>0:
                # Build fixed/free vertex partition
                fixedInds,freeInds = [],[]
                for v in self.mesh.id_vertices:
                    if v in self.feat.feature_vertices:
                        fixedInds.append(v)
                    else:
                        freeInds.append(v)
                lapI = lap[freeInds,:][:,freeInds]
                lapB = lap[freeInds,:][:,fixedInds]
                AI = A[freeInds, :][:,freeInds]
                # for lapI and lapB, only lines of freeInds are relevant : lines of fixedInds link fixed variables -> useless constraints
                valB = lapB.dot(self.var[fixedInds]) # right hand side
                mat = lapI -  alpha * AI
                for _ in range(self.n_smooth):
                    valI2 = alpha * AI.dot(self.var[freeInds])
                    res = linalg.spsolve(mat, - valB - valI2)
                    self.var[freeInds] = res
                    self.normalize()
            else:
                mat = lap - alpha * A
                for _ in range(self.n_smooth):
                    valI2 = alpha * A.dot(self.var)
                    self.var = linalg.spsolve(mat, - valI2)
                    self.normalize()
        else: 
            self.normalize()
        self.smoothed = True