from numba import njit, prange, set_num_threads
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .base import BaseParametrization
from ...mesh.datatypes import *
from ...mesh.mesh import copy
from ...mesh.mesh_attributes import ArrayAttribute
from ... import geometry as geom
from ...attributes.glob import euler_characteristic
from ...attributes.misc_faces import face_area

@njit(cache=True, fastmath=True)
def chi(D, eps):
    # regularization function
    return (D + np.sqrt(eps*eps + D*D))/2

@njit(cache=True, fastmath=True)
def chi_deriv(D,eps):
    return 0.5 + D/(2 * np.sqrt(eps*eps + D*D))

@njit(cache=True, fastmath=True)
def jacobian(pts, tri, ref_jac):
    iA,iB,iC = tri[0], tri[1], tri[2]
    A,B,C = pts[2*iA:2*iA+2], pts[2*iB:2*iB+2], pts[2*iC:2*iC+2]
    return np.vstack((B-A, C-A)).T @ ref_jac

@njit(cache=True, parallel=True)
def min_det(pts, tris, ref_jacs):
    n_tri = tris.shape[0]
    dets = np.zeros(n_tri, dtype=np.float64)
    for t in prange(n_tri):
        dets[t] = np.linalg.det(jacobian(pts, tris[t], ref_jacs[t]))
    return np.min(dets)

@njit(cache=True, parallel=True)
def winslow_energy_and_gradient(X, locked, tri, ref_jacs, areas, wf, wg, eps):
    E = 0
    grad = np.zeros_like(X)
    n = tri.shape[0]
    Z = np.array([[-1.,-1.], [1.,0.], [0.,1.]])
    for t in prange(n):
        J = jacobian(X, tri[t], ref_jacs[t])
        K = np.array([[J[1,1],-J[1,0]],[-J[0,1],J[0,0]]])
        detJ = np.linalg.det(J)
        chiJ = chi(detJ, eps)
        fJ = np.trace( np.transpose(J) @ J) / chiJ
        gJ = (detJ*detJ + 1) / chiJ
        # hJ = chiJ - np.log(chiJ)
        
        E += areas[t] * (wf * fJ  + wg * gJ)
        # E += areas[t] * ( wf * fJ + wg * hJ )

        # compute gradient according to J
        chi_derivJ = chi_deriv(detJ, eps)
        ZJ = Z @ ref_jacs[t]
        
        df_dj = (2*J -  K * fJ * chi_derivJ) / chiJ
        dg_dj = (2*detJ - gJ * chi_derivJ) / chiJ * K
        # dh_dj = (chi_derivJ * (1 - 1/chiJ)) * K
        dphi_dj = areas[t] * (wf * df_dj + wg * dg_dj)
        # dphi_dj = areas[t] * (wf * df_dj + wg * dh_dj)
        dphi_du = ZJ @ dphi_dj.T # chain rule for the actual variables
        for i,v in enumerate(tri[t]):
            if locked[v] : continue
            grad[2*v  ] += dphi_du[i,0]
            grad[2*v+1] += dphi_du[i,1]
    return E,grad

def untangle(
    points: np.ndarray,
    locked: np.ndarray,
    triangles: np.ndarray,
    ref_jacs: np.ndarray,
    verbose: bool = False,
    **kwargs
) -> np.ndarray :
    """
    Minimizes the regularized Winslow functional to untangle a 2D triangulation.

    Args:
        points (np.ndarray[float]): Initial position of points in 2D. Should be of shape (2*V,)
        locked (np.ndarray[bool]): Which points have a fixed position. Should be of shape (V,)
        triangles (np.ndarray[int]): Indices of triangles. Should be of shape (T,3)
        ref_jacs (np.ndarray[float]): Perfect element to consider for Jacobian computation for each triangle. Should be of shape (T,2,2).
        verbose (bool,optional): Verbose mode. Defaults to False.
    
    Keyword Args:
        areas (np.ndarray[float]): Areas of triangles in original mesh. Used as a weighting term in the energy's summation. Should be of shape (T,). Defaults to np.ones(T).
        weight_angles (float, optional): weight coefficient for the angle conservation term (f). Defaults to 1.
        weight_areas (float, optional): weight coefficient for the area conservation term (g). Defaults to 1.
        iter_max (int, optional): Maximum number of iterations in the L-BFGS solver. Defaults to 10000.
        n_eps_update (int, optional): number of updates of the regularization's epsilon. Defaults to 10.
        stop_if_positive (bool, optional): enable early stopping as soon as all dets are positive. Defaults to False.

    Returns:
        np.ndarray[float]: final positions of points in 2D, in shape (2V,)

    References:
        [1] _Foldover-free maps in 50 lines of code_, Garanzha et al., ACM ToG 2021
    """
    n_eps_update = kwargs.get("n_eps_update", 10)
    wf = kwargs.get("weight_angles", 1.)
    wg = kwargs.get("weight_areas", 1.)
    bfgs_iter_max = kwargs.get("iter_max", 10_000)
    stop_if_pos = kwargs.get("stop_if_positive",False)
    areas = kwargs.get("areas", None)
    set_num_threads(2)
        
    if areas is None:
        areas = np.ones(triangles.shape[0])

    if verbose:
        print("WEIGHT ANGLES:", wf)
        print("WEIGHT AREAS:", wg)

    for _ in range(n_eps_update):
        mindet = min_det(points, triangles, ref_jacs)
        if mindet>0 and stop_if_pos: break
        eps = np.sqrt(1e-12 + .04*min(mindet, 0)**2) # the regularization parameter e
        if verbose:
            print("min det=", mindet)
            print("epsilon=", eps)
        points = fmin_l_bfgs_b(
            winslow_energy_and_gradient, points, 
            args=(locked, triangles, ref_jacs, areas, wf, wg, eps),
            disp=10 if verbose else 0, maxiter=bfgs_iter_max
        )[0]
    
    if verbose: print("min det=", min_det(points, triangles, ref_jacs))

    return points


class WinslowInjectiveEmbedding(BaseParametrization):
    """
    Foldover-free map to the plane: computes an injective embedding in the plane starting from the provided $uv$-coordinates by minimizing the regularized Winslow functionnal
    of Garanzha et al. [1]. This class is essentially a wrapper around the `untangle` function.
    
    Warning: 
        The input mesh should have the topology of a disk.

        UV coordinates are computed per vertex and not per corner. See `mouette.attributes.scatter_vertices_to_corners` for conversion.

    References:
        [1] _Foldover-free maps in 50 lines of code_, Garanzha et al., ACM ToG 2021

    Example:
        See [https://github.com/GCoiffier/mouette/blob/main/examples/winslow_untangle.py](https://github.com/GCoiffier/mouette/blob/main/examples/winslow_untangle.py)
    """
            
    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, uv_init: ArrayAttribute, verbose : bool=True, **kwargs):
        """
        Args:
            mesh (SurfaceMesh): the supporting mesh. Should be a surface with disk topology.
            uv_init (ArrayAttribute): array of initial uv-coordinates per vertices. np.array of shape (V,2) or mouette.ArrayAttribute object.
            verbose (bool, optional): verbose mode. Defaults to True.
        
        Keyword Args:
            stop_if_positive (bool, optional): whether to stop the optimization as soon as all determinants are positive. Defaults to False.
            solver_verbose (bool, optional): Verbose tag for the L-BFGS solver. Defaults to False.
        """
        super().__init__("Winslow", mesh, verbose=verbose, uv_attr=uv_init, **kwargs)
        self._save_on_corners = False
        self._solver_verbose = kwargs.get("solver_verbose", False)
        self.lmbd = kwargs.get("lmbd", 1.)
        self.stop_if_pos = kwargs.get("stop_if_positive", False)
        
    def run(self):
        """
        Calls the solver.

        Raises:
            Exception: fails if the mesh is not a triangulation of a topological disk
        """
        if euler_characteristic(self.mesh)!=1:
            raise Exception("Mesh is not a topological disk")
        if not self.mesh.is_triangular():
            raise Exception("Mesh is not triangular.")

        ref_jacs = np.array([[[1.,0.], [0.,1.]] for _ in self.mesh.id_faces]) # reference element
        source_area = sum([geom.triangle_area_2D(self.uvs[A], self.uvs[B], self.uvs[C]) for (A,B,C) in self.mesh.faces])        
        scale = np.sqrt(len(self.mesh.faces)/source_area)

        points = np.concatenate([self.uvs[v] for v in self.mesh.id_vertices]) 
        points *= scale
        locked = np.array([self.mesh.is_vertex_on_border(v) for v in self.mesh.id_vertices])
        triangles = np.array(self.mesh.faces)
        
        areas = face_area(self.mesh, persistent=False, dense=True).as_array()

        final_points = untangle(points, locked, triangles, ref_jacs,
            areas=areas, weight_angles=1., weight_areas=self.lmbd, 
            verbose=self.verbose, stop_if_positive = self.stop_if_pos)
        final_points /= scale

        # Retrieve uvs and write them in attribute
        self.uvs = self.mesh.vertices.create_attribute("uv_coords", float, 2, dense=True)
        for v in self.mesh.id_vertices:
            self.uvs[v] = final_points[2*v:2*v+2]

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