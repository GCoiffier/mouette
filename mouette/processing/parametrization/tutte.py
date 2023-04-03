from .base import BaseParametrization
from ...mesh.datatypes import *
from ...geometry import Vec
from ... import operators
from ..border import extract_border_cycle
from ...attributes.glob import euler_characteristic
from ...utils.argument_check import check_argument

import numpy as np
from enum import Enum
import cmath
from math import pi
from scipy.sparse import linalg


class TutteEmbedding(BaseParametrization):
    """
    Tutte's embedding parametrization method for a disk inside a fixed boundary.
    The parametrization is locally injective (Floater, 1997) provided the boundary is convex.

    References:
        How to draw a graph, Tutte, 1963.
    """

    class BoundaryMode(Enum):
        """
        Enum that represents the shape of the boundary curve.
        """
        CIRCLE = 0
        SQUARE = 1

        @staticmethod
        def from_string(s :str):
            if "circle" in s.lower():
                return TutteEmbedding.BoundaryMode.CIRCLE
            if "square" in s.lower():
                return TutteEmbedding.BoundaryMode.SQUARE
            raise Exception(f"'{s}' does correspond to any boundary mode. Choices are : 'circle', 'square'")

    def __init__(self, mesh:SurfaceMesh, boundary_mode:str = "circle", verbose:bool=True, **kwargs):
        """
        Initializer of the Tutte's embedding tool.

        Args:
            mesh (SurfaceMesh): the mesh to embed. Should be a surface with disk topology.
            boundary_mode (str, optional): Shape of the boundary. Possible choices are ["square", "circle"]. Defaults to "circle".
            verbose (bool, optional): verbose mode. Defaults to True.
            save_on_corners (bool, optional): whether to store the results on face corners or vertices. Defaults to True
        

        Raises:
            InvalidArgumentValueError : if 'boundary_mode' is not "square" or "circle".
        """
        check_argument("boundary_mode", boundary_mode, str, ["square", "circle"])
        super().__init__("Tutte", mesh, verbose, **kwargs)
        self._bnd_mode : TutteEmbedding.BoundaryMode = TutteEmbedding.BoundaryMode.from_string(boundary_mode)

    def run(self) :
        if euler_characteristic(self.mesh)!=1:
            raise Exception("Mesh is not a topological disk. Cannot run parametrization.")

        Ubnd,Vbnd = self._initialize_boundary(self._bnd_mode)

        lap = operators.laplacian(self.mesh, cotan=False)
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
        if boundary_mode == TutteEmbedding.BoundaryMode.CIRCLE:
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