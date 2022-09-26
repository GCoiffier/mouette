from ...mesh.datatypes import *
from ...mesh.mesh import copy
from ...mesh.mesh_attributes import Attribute, ArrayAttribute
from ...geometry import Vec
from ..worker import Worker
from ... import geometry as geom
from ...attributes.glob import euler_characteristic

import numpy as np
from math import atan2

from osqp import OSQP
import scipy.sparse as sp
from scipy.sparse import linalg

class LSCM(Worker):
    """
    Least-Square Conformal Map algorithm for computing a parametrization of a mesh.
    /!\\ The mesh should have the topology of a disk.

    Usage :
    ```
    lscm = LSCM(mesh)
    lscm.run([options])
    ```

    or

    ```
    lscm = LSCM(mesh)([options]) # directly calls run
    ```

    This fills the self.uvs container
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, verbose:bool=True):
        super().__init__("LSCM", verbose=verbose)
        self.mesh : SurfaceMesh = mesh
        self._flat_mesh : SurfaceMesh = None
        self.uvs : Attribute = None # attribute on vertices 

    def run(self, eigen=True, save_on_mesh : bool = True, solver_verbose=False):
        if euler_characteristic(self.mesh)!=1:
            raise Exception("Mesh is not a topological disk. Cannot run LSCM.")
        if eigen:
            U = self._solve_eigen()
        else:
            U = self._solve(verbose=solver_verbose)
        U = U.reshape((U.size//2, 2))
        U = self._scale(U)
        if save_on_mesh:
            self.uvs = self.mesh.face_corners.create_attribute("uv_coords", float, 2, dense=True)
        else:
            self.uvs = ArrayAttribute(float, len(self.mesh.vertices), 2)
        for T in self.mesh.id_faces:
            for i,v in enumerate(self.mesh.faces[T]):
                self.uvs[3*T+i] = Vec(U[v])

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
        instance.setup(Q, A=Cst, l=Cst_rhs, u=Cst_rhs, verbose=verbose)
        res = instance.solve()
        self.log("Done")
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

    @property
    def flat_mesh(self):
        if self.uvs is None:
            return None
        if self._flat_mesh is None:
            # build the flat mesh : vertex coordinates are uv of original mesh
            self._flat_mesh = copy(self.mesh)
            for T in self.mesh.id_faces:
                for i,v in enumerate(self.mesh.faces[T]):
                    self._flat_mesh.vertices[v] = Vec(self.uvs[3*T+i][0], self.uvs[3*T+i][1], 0.)
        return self._flat_mesh