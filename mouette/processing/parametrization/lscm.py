from ...mesh.datatypes import *
from .base import BaseParametrization
from ...geometry import Vec
from ... import geometry as geom
from ...attributes.glob import euler_characteristic

import numpy as np

from osqp import OSQP
import scipy.sparse as sp
from scipy.sparse import linalg

class LSCM(BaseParametrization):
    """
    Least-Square Conformal Map algorithm for computing a parametrization of a mesh.
    /!\\ The mesh should have the topology of a disk.
    Computed UVs are stored in the self.uvs container

    References:
        - [1] _Least Squares Conformal Maps for Automatic Texture Atlas Generation_, Levy et al. (2002)
        - [2] _Spectral Conformal Parameterization_, Mullen et al. (2008)
        - [3] _Intrinsic Parameterizations of Surface Meshes_, Desbrun et al. (2002)
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, verbose : bool=True, **kwargs):
        """
        Args:
            mesh (SurfaceMesh): the supporting mesh. Should be a surface with disk topology.
            verbose (bool, optional): verbose mode. Defaults to True.
            
        Keyword Args:
            eigen (bool, optional): whether to solve a linear system with two fixed points or use an eigen solver. Defaults to True
            save_on_corners (bool, optional): whether to store the results on face corners or vertices. Defaults to True
            solver_verbose (bool, optional): verbose level. Defaults to False.
        """
        super().__init__("LSCM", mesh, verbose=verbose, **kwargs)
        self.residual : float = None # Final value of LSCM energy (residual of least square)
        self._eigen = kwargs.get("eigen", True)
        self._solver_verbose = kwargs.get("solver_verbose", False)

    def run(self):
        """
        Calls the solver on the LSCM system.

        Raises:
            Exception: fails if the mesh is not a topological disk
        """
        if euler_characteristic(self.mesh)!=1:
            raise Exception("Mesh is not a topological disk. Cannot run LSCM.")
        if self._eigen:
            U = self._solve_eigen()
        else:
            U = self._solve(verbose=self._solver_verbose)
        U = U.reshape((U.size//2, 2))
        U = self._scale(U)        
        # Retrieve uvs and write them in attribute
        if self.save_on_corners:
            self.uvs = self.mesh.face_corners.create_attribute("uv_coords", float, 2, dense=True)
            for c,v in enumerate(self.mesh.face_corners):
                self.uvs[c] = Vec(U[v])
        else:
            self.uvs = self.mesh.vertices.create_attribute("uv_coords", float, 2, dense=True)
            for v in self.mesh.id_vertices:
                self.uvs[v] = Vec(U[v])

    def _build_system(self):
        n = len(self.mesh.vertices)
        m = len(self.mesh.faces)
        cols, rows, vals = (np.zeros(12*m) for _ in range(3))
        for t,(i,j,k) in enumerate(self.mesh.faces):
            Pi, Pj, Pk = (self.mesh.vertices[_v] for _v in (i,j,k))
            X,Y,_ = geom.face_basis(Pi,Pj,Pk)
            Qi, Qj, Qk = ( Vec(X.dot(_P), Y.dot(_P)) for _P in (Pi,Pj,Pk))
            
            for _i, (r,c,v) in enumerate([
                # first row
                (2*t, 2*i, Qk[0] - Qj[0]),
                (2*t, 2*j, Qi[0] - Qk[0]),
                (2*t, 2*k, Qj[0] - Qi[0]),
                (2*t, 2*i+1, Qk[1] - Qj[1]),
                (2*t, 2*j+1, Qi[1] - Qk[1]),
                (2*t, 2*k+1, Qj[1] - Qi[1]),
                # second row
                (2*t+1, 2*i, Qk[1] - Qj[1]),
                (2*t+1, 2*j, Qi[1] - Qk[1]),
                (2*t+1, 2*k, Qj[1] - Qi[1]),
                (2*t+1, 2*i+1, Qj[0] - Qk[0]),
                (2*t+1, 2*j+1, Qk[0] - Qi[0]),
                (2*t+1, 2*k+1, Qi[0] - Qj[0])
            ]):
                rows[12*t+_i], cols[12*t+_i], vals[12*t+_i] = r, c, v 
        A = sp.csc_matrix((vals, (rows, cols)),  shape=(2*m,2*n))
        self.log("Size of system :", A.shape)
        self.log(f"({A.nnz} non zero values)")
        return A

    def _solve(self, verbose:bool):
        self.log("Building system...")
        n = len(self.mesh.vertices)
        A = self._build_system()

        # index locked 1 has value (0,0) and index locked2 has value (1,1)
        e = self.mesh.boundary_edges[0]
        locked1, locked2 = self.mesh.edges[e]
        locked1, locked2 = min(locked1, locked2), max(locked1, locked2)

        instance = OSQP()
        Q = A.transpose() @ A
        Cst = sp.lil_matrix((4,2*n))
        Cst[0, 2*locked1] = 1
        Cst[1, 2*locked1+1] = 1
        Cst[2, 2*locked2] = 1
        Cst[3, 2*locked2+1] = 1
        Cst = Cst.tocsc()
        Cst_rhs = np.array([0.,0.,1.,0.])
        
        self.log("Solving system...")
        instance.setup(Q, A=Cst, l=Cst_rhs, u=Cst_rhs, eps_abs=0., verbose=verbose)
        res = instance.solve()
        self.log(f"System solved in {res.info.run_time} s")
        self.residual = res.info.obj_val
        self.log(f"Residual: {self.residual}")
        return res.x

    def _solve_eigen(self):
        self.log("Building system...")
        A = self._build_system()
        self.log("Solving for eigenvector...")
        # u,s,v = linalg.svds(A, 2, which="SM", solver="lobpcg", maxiter=1000)
        # U = v[0]
        eigs, U = linalg.eigsh(A.transpose() @ A, 2, which="SM", tol=1e-2)
        self.log("Smallest eigenvalues:", eigs)
        U = U[:,1]
        return U

    def _scale(self, U):
        # scale UVs in bounding box [0;1]^2
        xmin,xmax,ymin,ymax = float("inf"), -float("inf"), float("inf"), -float("inf")
        for i in range(U.shape[0]):
            u,v = U[i][0], U[i][1]
            xmin = min(xmin, u)
            xmax = max(xmax, u)
            ymin = min(ymin, v)
            ymax = max(ymax, v)
        scale_x = xmax-xmin
        scale_y = ymax-ymin
        scale = min(scale_x, scale_y)
        # apply scale
        return U/scale