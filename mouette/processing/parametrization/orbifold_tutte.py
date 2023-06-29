from .base import BaseParametrization
from ...mesh.datatypes import *
from ...geometry import Vec
from ... import operators
from ..border import extract_border_cycle
from ...attributes.glob import euler_characteristic
from .. import SingularityCutter

import numpy as np
import cmath
from math import pi
from scipy.sparse import linalg


class OrbifoldTutteEmbedding(BaseParametrization):
    """
    Orbifold Tutte's embedding generalizes Tutte's embedding to spherical topologies. Using periodic conditions on a virtually defined boundary, it creates a parametrization that is a tiling of the plane, leading to a seamless parametrization of spheres.

    References:
        [1] Orbifold Tutte Embeddings, Noam Aigerman and Yaron Lipman, ACM Transaction on Graphics, 2015
    """

    def __init__(self, mesh:SurfaceMesh, singularities = None, use_cotan:bool=False, verbose:bool=True, **kwargs):
        """
        Initializer of the Orbifold Tutte's embedding method.

        Args:
            mesh (SurfaceMesh): the mesh to embed. Should be a surface with disk topology.
            singularities (iterable, optional): Three singularity points that define the virtual boundary. Need to be valid vertex indices. If not specified, will be generated automatically. Defaults to None.
            use_cotan (bool, optional): whether to use Tutte's original barycentric embedding [1], or use cotangents as weights in the laplacian matrix ([2]). Defaults to False.
            verbose (bool, optional): verbose mode. Defaults to True.
            save_on_corners (bool, optional): whether to store the results on face corners or vertices. Defaults to True
        """
        super().__init__("OrbifoldTutte", mesh, verbose, **kwargs)
        self._use_cotan : bool = use_cotan
        self._singus = None
        if singularities is not None:
            try:
                p1,p2,p3 = singularities
                assert all((isinstance(p,int) for p in (p1,p2,p3)))
                self._singus = p1,p2,p3
            except Exception as e:
                self.log("Invalid argument 'singularities' received. Generating ")
        if self._singus is None:
            self._singus = self._define_singularities()

    def _define_singularities(self):
        return 1,2,3

    def run(self) :
        if not (len(self.mesh.boundary_vertices)==0 and euler_characteristic(self.mesh)==2):
            raise Exception("Mesh is not a topological sphere. Cannot run parametrization.")

        ## Define a cut mesh along the shortest path towars the boundary
        cutter = SingularityCutter(self.mesh, self._singus)

        Ubnd,Vbnd = self._initialize_boundary(self._bnd_mode)

        lap = operators.laplacian(self.mesh, cotan=self._use_cotan)
        freeInds = self.mesh.interior_vertices
        bndInds, _ = extract_border_cycle(self.mesh)

        LI = lap[freeInds, :][:, freeInds]
        LB = lap[freeInds, :][:, bndInds]

        U = linalg.spsolve(LI, -LB.dot(Ubnd))
        V = linalg.spsolve(LI, -LB.dot(Vbnd))

        if self.save_on_corners:
            self.uvs = self.mesh.face_corners.create_attribute("uv_coords", float, 2, dense=True)
            for i,v in enumerate(freeInds):
                for c in self.mesh.connectivity.vertex_to_corner(v):
                    self.uvs[c] = Vec(U[i], V[i])
            for i,v in enumerate(bndInds):
                for c in self.mesh.connectivity.vertex_to_corner(v):
                    self.uvs[c] = Vec(Ubnd[i], Vbnd[i])
        else:
            self.uvs = self.mesh.vertices.create_attribute("uv_coords", float, 2, dense=True)
            for i,v in enumerate(freeInds):
                self.uvs[v] = Vec(U[i], V[i])
            for i,v in enumerate(bndInds):
                self.uvs[v] = Vec(Ubnd[i], Vbnd[i])

    def _initialize_boundary(self, boundary_mode):
        n = len(self.mesh.boundary_vertices)
        U, V = np.zeros(n), np.zeros(n)
        if boundary_mode == TutteEmbedding.BoundaryMode.CUSTOM:
            U,V = self._custom_bnd[:,0], self._custom_bnd[:,1]
        elif boundary_mode == TutteEmbedding.BoundaryMode.CIRCLE:
            for i in range(n):
                rt = cmath.rect(1., 2*pi*i/n)
                U[i] = rt.real
                V[i] = rt.imag
        elif boundary_mode == TutteEmbedding.BoundaryMode.SQUARE:
            corners = [0, n//4, n//2, (3*n)//4]
            U[corners[0]], V[corners[0]] = 0,0
            U[corners[1]], V[corners[1]] = 1,0
            U[corners[2]], V[corners[2]] = 1,1
            U[corners[3]], V[corners[3]] = 0,1
            for i in range(1, n//4):
                U[i] = 4*i/n
                # V = 0
            for i,v in enumerate(range(n//4+1, n//2)):
                U[v] = 1
                V[v] = 4*i/n
            for i,v in enumerate(range(n//2+1, (3*n)//4)):
                U[v] = 1-4*i/n
                V[v] = 1
            for i,v in enumerate(range((3*n)//4+1, n)):
                # U = 0
                V[v] = 1-4*i/n
        return U, V