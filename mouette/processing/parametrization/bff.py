from .base import BaseParametrization
from ...mesh.datatypes import *
from ...mesh.mesh_attributes import *
from ... import attributes
from ... import geometry as geom
from ...operators import laplacian
from ..border import extract_border_cycle
from ...attributes.glob import euler_characteristic

from ..misc import reorder_vertices
from ..border import extract_border_cycle

import numpy as np
import cmath
import scipy.sparse as sp

class BoundaryFirstFlattening(BaseParametrization):
    """
    Boundary First Flattening: A conformal flattening algorithm with control over the boundary 
    conditions, either in terms of scale factor or curvature

    References:
        - [1] _Boundary First Flattening_, Rohan Sawhney and Keenan Crane, ACM ToG, 2017

    Warning:
        This algorithm reorders the vertices so that boundary vertices are labeled [0, N-1] in order. Therefore, the uvs coordinates are not computed on the original mesh but on a reordered copy. Access the final result with `self.mesh`
    """

    def __init__(self, 
        mesh: SurfaceMesh,
        bnd_scale_fctr : Attribute = None,
        bnd_curvature : Attribute = None,
        verbose:bool=True, 
        **kwargs):
        """
        Args:
            mesh (SurfaceMesh): the input surface. Should be a triangulation of a topological disk.
            bnd_scale_fctr (Attribute, optional): Scale factor on the boundary. Ignores values for interior vertices. Defaults to None. If provided, will automatically compute boundary curvature and ignore the `bnd_curvature`a argument.
            bnd_curvature (Attribute, optional): Geodesic curvature on the boundary. Ignores values for interior vertices. Defaults to None. If provided (and no `bnd_scale_fctr` is provided), will automatically compute boundary scale factors.
            verbose (bool, optional): verbose mode. Defaults to True.
        
        Keyword Args:
            save_on_corners (bool, optional): whether to store the results on face corners or vertices. Defaults to True
            use_cotan (bool, optional): whether to use cotan Laplacian or connectivity Laplacian. Defaults to True.
            hilbert_transform (bool, optional): whether to extend the boundary values with the Hilbert transform 
            (favors conformality instead of boundary preservation) or using harmonic extensions 
            (exact boundary but further from truly conformal). Defaults to True.

        Raises:
            Exception: if the mesh is not the triangulation of a topological disk

        Note:
            If neither of `bnd_scale_fctr` and `bnd_curvature` are provided, the algorithm will run in default mode with scale factors = 0 on the boundary. Otherwise, provided scale factors have priority over provided curvatures.
        """

        super().__init__("BFF", mesh, verbose, **kwargs)
        self._Bscale_fctr = bnd_scale_fctr
        self._Bcurvature = bnd_curvature
        self.use_cotan = kwargs.get("use_cotan", True)
        self.hilbert_transform = kwargs.get("hilbert_transform", True)

        self._curv : ArrayAttribute = None
        self.NV,self.NB = None,None
        self._L : sp.csc_matrix = None # Laplacian operator
        self._Lii, self._Lib, self._Lbb = None, None, None # Laplacian submatrices
        
        self._vertex_permutation = None
        self._Bedges = None # list of edges (in order) that make the boundary
        self._Blengths = None
        self._Btarget_lenghts = None

    def run(self) -> SurfaceMesh:
        """Run the algorithm

        Raises:
            Exception: If the mesh is not a triangulation of a topological disk

        Returns:
            SurfaceMesh: a copy of the original mesh with reordered vertices and computed uv-coordinates
        """
        if not (len(self.mesh.boundary_vertices)>1 and euler_characteristic(self.mesh)==1):
            raise Exception("Mesh is not a topological disk. Cannot run parametrization.")
        if not self.mesh.is_triangular():
            raise Exception("Mesh is not a triangulation. Cannot run parametrization.")

        self.log("Reorder mesh vertices")
        self._Bedges = self._reorder_vertices()
        self.NV, self.NB = len(self.mesh.vertices), len(self.mesh.boundary_vertices)

        self.log("Compute Gaussian and geodesic curvatures")
        self._curv  = attributes.angle_defects(self.mesh).as_array(self.NV) # Gaussian curvature

        self.log("Compute Laplacian operator")
        self._L = laplacian(self.mesh, cotan = self.use_cotan)

        self._Lii = self._L[ self.NB:,:][:, self.NB:]
        self._Lib = self._L[ self.NB:,:][:,:self.NB ]
        self._Lbb = self._L[:self.NB ,:][:,:self.NB ]

        self.log("Initialize Boundary scale factor and curvature")
        if self._Bscale_fctr is None and self._Bcurvature is None:
            # Nothing was provided. By default, we run with zero scale factor on the boundary
            self._Bscale_fctr = np.zeros(self.NB)
        
        if self._Bscale_fctr is not None:
            self._Bscale_fctr = np.array([self._Bscale_fctr[v] for v in self._vertex_permutation[:self.NB]])
            # Curvature was provided -> extrapolate scale factors via Neumann to Dirichlet
            self._Bcurvature = self._dirichlet_to_neumann(-self._curv, self._Bscale_fctr)
        elif self._Bcurvature is not None:
            # Scale factors were provided -> extrapolate curvature via Dirichlet to Neumann
            self._Bscale_fctr = self._neumann_to_dirichlet()
        else:
            raise Exception("Should not happen.")
                
        self.log("Fit best boundary curve")
        Ub,Vb = self._fit_boundary_curve()

        self.log("Extend curve values inside the surface")
        U,V = self._extend_boundary_values(Ub,Vb)

        self.log("Write UV-coordinates")
        if self.save_on_corners:
            self.uvs = self.mesh.face_corners.create_attribute("uv_coords", float, 2, dense=True)
            for ic,v in enumerate(self.mesh.face_corners):
                self.uvs[ic] = Vec(U[v], V[v])
        else:
            self.uvs = self.mesh.vertices.register_array_as_attribute("uv_coords", np.vstack((U,V)).T)
        return self.mesh

    def _reorder_vertices(self):
        v_border, _ = extract_border_cycle(self.mesh)
        self._vertex_permutation = v_border + self.mesh.interior_vertices
        self.mesh = reorder_vertices(self.mesh, self._vertex_permutation)
        _, e_border = extract_border_cycle(self.mesh)
        return e_border

    def _dirichlet_to_neumann(self, phi, U):
        a = sp.linalg.spsolve(self._Lii, phi[self.NB:] - self._Lib @ U)
        return phi[:self.NB] - self._Lib.transpose() @ a - self._Lbb @ U

    def _neumann_to_dirichlet(self):    
        raise NotImplementedError

    def _fit_boundary_curve(self):
        self._Blengths = np.zeros(len(self.mesh.boundary_edges))
        self._Btarget_lenghts = np.zeros(len(self.mesh.boundary_edges))
        for ie,e in enumerate(self._Bedges):
            A,B = self.mesh.edges[e]
            uA,uB = self._Bscale_fctr[A], self._Bscale_fctr[B]
            self._Blengths[ie] = geom.distance( self.mesh.vertices[A], self.mesh.vertices[B])
            self._Btarget_lenghts[ie] = self._Blengths[ie] * np.exp((uA+uB)/2)
        
        T = np.zeros((self.NB,2))
        direction = 0.
        for i in range(self.NB):
            direction += self._Bcurvature[i]
            T[i] = geom.Vec.from_complex(cmath.rect(1, direction))
        N = sp.diags(self._Blengths)
        final_lengths = self._Btarget_lenghts - N @ T @ np.linalg.inv(T.transpose() @ N @ T) @ T.transpose() @ self._Btarget_lenghts
        bnd_pts = np.zeros((self.NB,2))
        for i in range(1,self.NB):
            bnd_pts[i] = bnd_pts[i-1] + final_lengths[i-1] * T[i-1]
        return bnd_pts[:,0], bnd_pts[:,1]
    
    def _extend_boundary_values(self, Ub, Vb):
        if self.hilbert_transform:
            Ui = sp.linalg.spsolve(self._Lii, - self._Lib @ Ub) # extending u coordinate
            U = np.concatenate((Ub,Ui))
            h = np.zeros(self.NV)
            for i in range(self.NB):
                h[i] = (Ub[(i+1)%self.NB] - Ub[(i-1)%self.NB])/2
            V = sp.linalg.spsolve(self._L, h)
        else:
            Ui = sp.linalg.spsolve(self._Lii, - self._Lib @ Ub) # extending u coordinate
            Vi = sp.linalg.spsolve(self._Lii, -self._Lib @ Vb) # extending v coordinate
            U,V = np.concatenate((Ub,Ui)), np.concatenate((Vb,Vi))    
        return U,V