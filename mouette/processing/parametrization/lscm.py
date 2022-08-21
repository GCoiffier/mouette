
from ...mesh.datatypes import *
from ...mesh.mesh import copy
from ...mesh.mesh_attributes import Attribute, ArrayAttribute
from ...geometry import Vec
from ..worker import Worker
from ... import geometry as geom
from ...attributes.glob import euler_characteristic

import numpy as np
from math import atan2
from scipy.sparse.linalg import lsqr, eigsh
from scipy.sparse import coo_matrix

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

    def run(self, eigen=True, save_on_mesh = True, axis_align = True, verbose=True):
        self.verbose = verbose
        if euler_characteristic(self.mesh)!=1:
            raise Exception("Mesh is not a topological disk. Cannot run LSCM.")
        if eigen:
            U = self._solve_eigen()
        else:
            U = self._solve()
        U = U.reshape((U.size//2, 2))
        if axis_align:
            U = self._align_with_axes(U)
        U = self._scale(U)
        if save_on_mesh:
            self.uvs = self.mesh.vertices.create_attribute("uv_coords", float, 2, dense=True)
        else:
            self.uvs = ArrayAttribute(float, len(self.mesh.vertices), 2)
        for v in self.mesh.id_vertices:
            self.uvs[v] = Vec(U[v])

    def _build_system(self):
        n = len(self.mesh.vertices)
        m = len(self.mesh.faces)
        cols, rows, vals = (np.zeros(12*m) for _ in range(3))
        for t,T in enumerate(self.mesh.faces):
            i,j,k = T
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
        A = coo_matrix((vals, (rows, cols)),  shape=(2*m,2*n)).tolil()
        self.log("Size of system :", A.shape)
        return A

    def _solve(self):
        self.log("Building system...")
        n = len(self.mesh.vertices)
        A = self._build_system()

        # index locked 1 has value (0,0) and index locked2 has value (1,1)
        locked1 = np.random.randint(0,n)
        locked2 = np.random.randint(0,n)
        while locked2 == locked1:
            locked2 = np.random.randint(0,n)
        locked1, locked2 = min(locked1, locked2), max(locked1, locked2)

        B = np.squeeze(np.array(A.getcol(2*locked2).todense() + A.getcol(2*locked2+1).todense()))
        inds = [x for x in range(A.shape[1]) if x not in {2*locked1, 2*locked1+1, 2*locked2, 2*locked2+1}]
        A = A[:, inds] # remove corresponding columns 
        self.log("Solving system...")
        res = lsqr(A, -B)
        U_int = res[0]
        self.log("Done. Residual norm ||AX-B||=",res[3])
        U = np.zeros((2*n,))
        U[:2*locked1] = U_int[:2*locked1]
        U[2*locked1+2:2*locked2] = U_int[2*locked1:2*locked2-2]
        U[2*locked2] = 1
        U[2*locked2 + 1] = 1
        U[2*locked2+2:] = U_int[2*locked2-2:]
        return U

    def _solve_eigen(self):
        self.log("Building system...")
        A = self._build_system()
        B = A.transpose() @ A
        self.log("Solving for eigenvector...")
        egv, U = eigsh(B, 3, which="SM", tol=1e-4)
        self.log("Done. Smallest eigenvalues:", egv)
        U = U[:,0]
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

    def _align_with_axes(self,U):
        # rotate I.UVs so that feature edges are axis aligned
        orig = U[0]
        e = self.mesh.boundary_edges[0] # should exist since mesh is disk topology
        A,B = self.mesh.edges[e]
        vec = U[A] - U[B]
        angle = -atan2(vec[1], vec[0])
        # apply transformation
        for v in self.mesh.id_vertices:
            uvi = Vec(U[v]-orig)
            U[v] = geom.rotate_2d(uvi, angle) + orig
        return U

    @property
    def flat_mesh(self):
        if self.uvs is None:
            return None
        if self._flat_mesh is None:
            # build the flat mesh : vertex coordinates are uv of original mesh
            self._flat_mesh = copy(self.mesh)
            for i in self.mesh.id_vertices:
                self._flat_mesh.vertices[i] = Vec(self.uvs[i][0], self.uvs[i][1], 0.)
        return self._flat_mesh