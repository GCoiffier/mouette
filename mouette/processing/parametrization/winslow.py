from numba import jit, prange
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .base import BaseParametrization
from ...mesh.datatypes import *
from ...mesh.mesh import copy
from ...mesh.mesh_attributes import ArrayAttribute
from ... import geometry as geom
from ...attributes.glob import euler_characteristic
from ...attributes.misc_faces import face_area

@jit(cache=True)
def chi(D, eps):
    # regularization function
    return (D + np.sqrt(eps*eps + D*D))/2

@jit(cache=True)
def chi_deriv(D,eps):
    return 0.5 + D/(2 * np.sqrt(eps*eps + D*D))

@jit(cache=True)
def jacobian(t, pts, tri, ref_jacs):
    iA,iB,iC = tri[t,0], tri[t,1], tri[t,2]
    A,B,C = pts[2*iA:2*iA+2], pts[2*iB:2*iB+2], pts[2*iC:2*iC+2]
    return np.vstack((C-A, B-A)) @ ref_jacs[t]

@jit(cache=True, parallel=True)
def energy_and_gradient(X, locked, tri, area, ref_jacs, wf, wg, eps):
    E = 0
    grad = np.zeros_like(X)
    n = tri.shape[0]
    Z = np.array([[-1,-1],[0,1],[1,0]], dtype=np.float64)
    for t in prange(n):
        J = jacobian(t, X, tri, ref_jacs)
        detJ = np.linalg.det(J)
        chiJ = chi(detJ, eps)
        fJ = np.trace( np.transpose(J) * J) / chiJ
        gJ = (detJ*detJ + 1) / chiJ
        # hJ = chiJ - np.log(chiJ)
        
        E += area[t] * ( wf * fJ + wg * gJ)
        # E += area[t] * ( wf * fJ + wg * hJ )

        # compute gradient according to J
        chi_derivJ = chi_deriv(detJ, eps)
        ZJ = Z @ ref_jacs[t]
        for dim in range(2): # iterate over the columns of J
            a_i = J[:,dim] 
            if dim==0:
                b_i = np.array([ J[1,1], -J[0,1]])
            else:
                b_i = np.array([-J[1,0],  J[0,0]])
            dfda = (2 * a_i - fJ * chi_derivJ * b_i) / chiJ
            dgda = (2 * detJ - gJ * chi_derivJ) / chiJ  * b_i
            # dhda = chi_derivJ * (1 - 1/chiJ) * b_i

            dEda =  area[t] * (wf * dfda + wg * dgda)
            # dEda =  area[t] * ( wf * dfda + wg * dhda )

            # Apply chain rule to get gradient according to variables
            for k,iV in enumerate(tri[t]):
                grad[2*iV+dim] += np.dot(dEda, ZJ[k,:])
    grad[locked] = 0.
    return E,grad

def untangle(
    points: np.ndarray,
    locked: np.ndarray,
    triangles: np.ndarray,
    areas : np.ndarray,
    ref_jacs: np.ndarray,
    verbose: bool = False,
    **kwargs
) -> np.ndarray :
    """
    Untangle a triangular mesh in 2D

    Args:
        points (np.ndarray): Initial position of points in 2D
        locked (np.ndarray): Which points have a fixed position
        triangles (np.ndarray): Indices of triangles
        areas (np.ndarray): Areas of triangles in original mesh. Used as a weighting term in the energy's summation.
        ref_jacs (np.ndarray): Perfect element to consider for Jacobian computation for each triangle
        verbose (bool,optional): Verbose mode. Defaults to False.
    
    Keyword Args:
        weight_angles (float, optional):        
        weight_areas (float, optional):        
        n_eps_update (int, optional):
        stop_if_positive (bool, optional):

    Returns:
        np.ndarray: final positions of points in 2D
    """
    n_eps_update = kwargs.get("n_eps_update", 10)
    wf = kwargs.get("weight_angles", 1.)
    wg = kwargs.get("weight_areas", 1.)
    bfgs_iter_max = kwargs.get("iter_max", 10_000)
    stop_if_pos = kwargs.get("stop_if_positive",False)

    if verbose:
        print("WEIGHT ANGLES:", wf)
        print("WEIGHT AREAS:", wg)

    for _ in range(n_eps_update):
        mindet = min([ np.linalg.det(jacobian(t,points,triangles,ref_jacs)) for t in range(triangles.shape[0])])
        if mindet>0 and stop_if_pos: break
        eps = np.sqrt(1e-12 + .04*min(mindet, 0)**2) # the regularization parameter e
        if verbose:
            print("min det=", mindet)
            print("epsilon=", eps)
        callback = lambda X : energy_and_gradient(X, np.repeat(locked,2), triangles, areas, ref_jacs, wf, wg, eps)
        points = fmin_l_bfgs_b(callback, points, disp=10 if verbose else 0, maxiter=bfgs_iter_max)[0]
    
    if verbose:
        mindet = min([ np.linalg.det(jacobian(t,points,triangles,ref_jacs)) for t in range(triangles.shape[0])])
        print("min det=", mindet)

    return points


class WinslowInjectiveEmbedding(BaseParametrization):
    """
    Foldover-free maps to the plane.
    /!\\ The input mesh should have the topology of a disk.
    Computed UVs are stored in the self.uvs container.
    UVs are per vertex.

    References:
        - [1] _Foldover-free maps in 50 lines of code_, Garanzha et al., ACM ToG 2021
    """
            
    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, uv_init: ArrayAttribute, verbose : bool=True, **kwargs):
        """
        Initializes the Winslow embedding.

        Args:
            mesh (SurfaceMesh): the supporting mesh. Should be a surface with disk topology.
            uv_init (ArrayAttribute): array of initial uv-coordinates per vertices. np.array of shape (V,2) or mouette.ArrayAttribute object.
            verbose (bool, optional): verbose mode. Defaults to True.
        
        Keyword Args:
            stop_if_positive (bool, optional): whether to stop the optimization as soon as all determinants are positive. Defaults to False.
            solver_verbose (bool, optional): verbose level. Defaults to False.
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
            Exception: fails if the mesh is not a topological disk
        """
        if euler_characteristic(self.mesh)!=1:
            raise Exception("Mesh is not a topological disk")

        ref_jacs = np.array([[[1.,0.], [0.,1.]] for _ in self.mesh.id_faces]) # reference element
        source_area = sum([geom.triangle_area_2D(self.uvs[A], self.uvs[B], self.uvs[C]) for (A,B,C) in self.mesh.faces])        
        scale = np.sqrt(len(self.mesh.faces)/source_area)

        points = np.concatenate([self.uvs[v] for v in self.mesh.id_vertices]) 
        points *= scale
        locked = np.array([self.mesh.is_vertex_on_border(v) for v in self.mesh.id_vertices])
        triangles = np.array(self.mesh.faces)
        
        areas = face_area(self.mesh, persistent=False, dense=True).as_array()
        # areas = np.ones(len(self.mesh.faces))

        final_points = untangle(points, locked, triangles, areas, ref_jacs, 
            weight_angles=1., weight_areas=self.lmbd, verbose=self.verbose,
            stop_if_positive = self.stop_if_pos)
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