from scipy.optimize import fmin_l_bfgs_b
import scipy.sparse as sp
import numpy as np
from numba import njit, prange

from .base import BaseParametrization
from ...mesh import copy
from ...mesh.datatypes import *
from ...mesh.mesh_attributes import ArrayAttribute
from ... import geometry as geom
from ...utils.argument_check import check_argument
from ...attributes.glob import euler_characteristic
from ...attributes.misc_corners import cotangent
from ...operators import area_weight_matrix

@njit(cache=True, fastmath=True)
def regul(x, eps=1e-18):
    return (x*x + eps)**(0.25)

@njit(cache=True, fastmath=True)
def regul_prime(x, eps=1e-18):
    y = x*x + eps
    return 0.5 * x / (y**0.75)

@njit(cache=True, parallel=True)
def cotan_embedding_energy_and_gradient(X, tris, locked):
    E = 0
    grad = np.zeros_like(X)
    nt = tris.shape[0]
    for t in prange(nt):
        A,B,C = tris[t,:]
        iuA,ivA,iuB,ivB,iuC,ivC = 2*A, 2*A+1, 2*B, 2*B+1, 2*C, 2*C+1
        uA,vA,uB,vB,uC,vC = X[iuA], X[ivA], X[iuB], X[ivB], X[iuC], X[ivC]

        At = uA * (vB - vC) + uB * (vC - vA) + uC * (vA - vB)
        grad[iuA] += 2*(vB - vC)
        grad[ivA] += 2*(uC - uB)
        grad[iuB] += 2*(vC - vA)
        grad[ivB] += 2*(uA - uC)
        grad[iuC] += 2*(vA - vB)
        grad[ivC] += 2*(uB - uA)
        
        l_bc = (uB-uC)**2 + (vB-vC)**2
        l_ac = (uA-uC)**2 + (vA-vC)**2  
        l_ab = (uA-uB)**2 + (vA-vB)**2

        L_ab = - l_ab + l_bc + l_ac 
        L_bc =   l_ab - l_bc + l_ac
        L_ac =   l_ab + l_bc - l_ac
        U = 2*l_bc*l_ac + 2*l_ac*l_ab + 2*l_bc*l_ab - l_bc*l_bc - l_ac*l_ac - l_ab*l_ab

        Dt = regul(U)
        Dtp = regul_prime(U)
        grad[iuA] += 4*Dtp * ( (uA - uB)*L_ab + (uA - uC)*L_ac )
        grad[ivA] += 4*Dtp * ( (vA - vB)*L_ab + (vA - vC)*L_ac )
        grad[iuB] += 4*Dtp * ( (uB - uA)*L_ab + (uB - uC)*L_bc )
        grad[ivB] += 4*Dtp * ( (vB - vA)*L_ab + (vB - vC)*L_bc )
        grad[iuC] += 4*Dtp * ( (uC - uA)*L_ac + (uC - uB)*L_bc )
        grad[ivC] += 4*Dtp * ( (vC - vA)*L_ac + (vC - vB)*L_bc )
        E += (Dt + 2*At)
    grad[locked] = 0.
    return E, grad

class CotanEmbedding(BaseParametrization):
    """
    Given a parametrization of a disk where boundary is fixed, we can optimize the difference between unsigned and signed areas of triangles to compute a parametrization that is foldover-free.

    Warning: 
        The input mesh should have the topology of a disk.

        UV coordinates are computed per vertex and not per corner. See `mouette.attributes.scatter_vertices_to_corners` for conversion.

    References:
        [1] _Embedding a triangular graph within a given boundary_, Xu et al. (2011)

    Example:
        See [https://github.com/GCoiffier/mouette/blob/main/examples/cotan_embedding.py](https://github.com/GCoiffier/mouette/blob/main/examples/cotan_embedding.py)
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, uv_init: ArrayAttribute, mode: str = "bfgs", verbose:bool=True, **kwargs):
        """
        Args:
            mesh (SurfaceMesh): the supporting mesh.
            uv_init (ArrayAttribute): array of initial uv-coordinates per vertices. np.array of shape (V,2) or mouette.ArrayAttribute object.
            verbose (bool, optional): verbose mode. Defaults to True.
            mode (str): optimization mode. Either "bfgs" or "alternate". Defaults to "bfgs".
        
        Keyword Args:
            tutte_if_convex (bool): if the boundary of the shape is convex and mode=="alternate", whether to simply run Tutte's embedding. Defaults to True
            solver_verbose (bool): solver verbose mode. Defaults to False.
        """
        kwargs["uv_attr"] = uv_init
        super().__init__("CotanEmbedding", mesh, verbose=verbose, **kwargs)
        self._save_on_corners = False
        self._mode = mode
        check_argument("mode", self._mode, str, ["bfgs", "alternate"])
        self._tutte_if_convex : bool = kwargs.get("tutte_if_convex", True)
        self._solver_verbose : bool = kwargs.get("solver_verbose", False)

    def run(self):
        """
        Runs the optimization

        Raises:
            Exception: Fails if the mesh is not a topological disk
            Exception: Fails if the mesh is not triangular
        """
        if euler_characteristic(self.mesh)!=1:
            raise Exception("Mesh is not a topological disk. Cannot run parametrization.")
        if not self.mesh.is_triangular():
            raise Exception("Mesh is not triangular.")
        
        # Call corresponding optimizer
        if self._mode=="bfgs":
            # Initialize variable vector from uvs
            var = np.concatenate([self.uvs[v] for v in self.mesh.id_vertices])
            locked = np.array([self.mesh.is_vertex_on_border(v) for v in self.mesh.id_vertices])
            locked = np.repeat(locked, 2)
            triangles = np.array(self.mesh.faces)
            self.log("Optimize cotan energy")
            var,energy,infos = fmin_l_bfgs_b(
                cotan_embedding_energy_and_gradient, # function and gradient to optimize
                var, # Initial variables
                args = (triangles, locked), # arguments to be given to the function to optimize
                maxiter = 1e4, # maximum number of iterations
                m = 11, # number of inner iterations for hessian approx.
                maxls = 30, # maximum number of linesearch
                factr = 10, # stops iterations if improvement is < factr*eps_machine
                disp=self._solver_verbose, # verbose level
            )
            self.log(f"Stopped after {infos['nit']} iterations")
            self.log(f"Final energy : {energy}")
            for v in self.mesh.id_vertices:
                self.uvs[v] = var[2*v:2*v+2]    

        elif self._mode=="alternate":
            var = self.uvs.as_array(len(self.mesh.vertices))
            var = self._optimize_alternate(var, self._tutte_if_convex, self._solver_verbose)
            for v in self.mesh.id_vertices:
                self.uvs[v] = var[v,:]

    def _optimize_alternate(self, var, tutte, verbose):

        def check_dets(var):
            for (A,B,C) in self.mesh.faces:
                uA,vA = var[A,:]
                uB,vB = var[B,:]
                uC,vC = var[C,:]
                if (uA * (vC - vB) + uB * (vA - vC) + uC * (vB - vA)) < 1e-8 : return False
            return True

        n_iter = 0
        n_iter_max = 100
        n_coeff_lap = 12*len(self.mesh.faces)
        rows = np.zeros(n_coeff_lap, dtype=np.int32)
        cols = np.zeros(n_coeff_lap, dtype=np.int32)
        coeffs = np.zeros(n_coeff_lap, dtype=np.float64)
        Iinds = np.array(self.mesh.interior_vertices, dtype=np.int32)
        Binds = np.array(self.mesh.boundary_vertices, dtype=np.int32)
        AreaW = area_weight_matrix(self.mesh, inverse=True).tocsc()
        if tutte:
            cot = cotangent(self.mesh, persistent=False)

        while n_iter<n_iter_max and not check_dets(var):
            n_iter += 1
            self.log(f"Iter {n_iter}/{n_iter_max}")
            # build cotan laplacian  
            _c = 0
            eps = 1e-10 if n_iter<10 else 0
            for iF,(A,B,C) in enumerate(self.mesh.faces):
                if tutte and n_iter==1:
                    cot_a,cot_b,cot_c = (cot[3*iF+_i]/2 for _i in range(3))
                else:
                    uA,vA = var[A,:]
                    uB,vB = var[B,:]
                    uC,vC = var[C,:] 
                    l_bc = (uB-uC)**2 + (vB-vC)**2
                    l_ac = (uA-uC)**2 + (vA-vC)**2  
                    l_ab = (uA-uB)**2 + (vA-vB)**2
                    Eli = 2*l_bc*l_ac + 2*l_ac*l_ab + 2*l_bc*l_ab - l_bc*l_bc - l_ac*l_ac - l_ab*l_ab
                    if eps==0:
                        El = np.sqrt(Eli)
                    else:
                        El = regul(Eli, eps)
                    cot_a = (l_ac+l_ab-l_bc)/El
                    cot_b = (l_bc+l_ab-l_ac)/El
                    cot_c = (l_bc+l_ac-l_ab)/El
                for (i, j, v) in [(A, B, cot_c/2), (B, C, cot_a/2), (C, A, cot_b/2)]:
                    rows[_c], cols[_c], coeffs[_c], _c = i, i, v, _c+1
                    rows[_c], cols[_c], coeffs[_c], _c = j, j, v, _c+1
                    rows[_c], cols[_c], coeffs[_c], _c = i, j, -v, _c+1
                    rows[_c], cols[_c], coeffs[_c], _c = j, i, -v, _c+1 
            L = AreaW * sp.csc_matrix((coeffs,(rows,cols)), dtype= np.float64)
            LI = L[Iinds,:][:,Iinds]
            LB = L[Iinds,:][:,Binds] 
            # solve linear system
            var[Iinds,0] = sp.linalg.spsolve(LI, -LB.dot(var[Binds,0])) # first system for u
            var[Iinds,1] = sp.linalg.spsolve(LI, -LB.dot(var[Binds,1])) # second system for v
        return var
    
    @property
    def flat_mesh(self) -> SurfaceMesh:
        """
        A flat representation of the mesh where uv-coordinates are copied to xy.

        Returns:
            SurfaceMesh: the flat mesh
        """
        if self.uvs is None: return None
        flat = copy(self.mesh)
        # build the flat mesh : vertex coordinates are uv of original mesh
        for T in self.mesh.id_faces:
            for i,v in enumerate(self.mesh.faces[T]):
                flat.vertices[v] = geom.Vec(self.uvs[v][0], self.uvs[v][1], 0.)
        return flat