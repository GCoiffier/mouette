from ...mesh.datatypes import *
from ...mesh.mesh import copy
from ...mesh.mesh_attributes import Attribute, ArrayAttribute
from ...geometry import Vec
from ... import operators
from ..border import extract_border_cycle
from ..worker import Worker
from ...attributes.glob import euler_characteristic
from ...utils.argument_check import check_argument

import numpy as np
from enum import Enum
import cmath
from math import pi
from scipy.sparse import linalg


class TutteEmbedding(Worker):
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

    def __init__(self, mesh : SurfaceMesh, verbose : bool=True, boundary_mode : str = "circle", save_on_mesh : bool = True):
        """
        Initializer of the Tutte's embedding tool.

        Args:
            mesh (SurfaceMesh): the mesh to embed. Should be a surface with disk topology.
            verbose (bool, optional): verbose mode. Defaults to True.
            boundary_mode (str, optional): Shape of the boundary. Possible choices are ["square", "circle"]. Defaults to "circle".
            save_on_mesh (bool, optional): whether to write uv coordinates as an attribute on the mesh. Defaults to True.
        

        Raises:
            InvalidArgumentValueError : if 'boundary_mode' is not "square" or "circle".
        """
        check_argument("boundary_mode", boundary_mode, str, ["square", "circle"])
        super().__init__(self, verbose=verbose)
        self.mesh : SurfaceMesh = mesh
        self._bnd_mode : TutteEmbedding.BoundaryMode = TutteEmbedding.BoundaryMode.from_string(boundary_mode)
        self._save_on_mesh : bool = save_on_mesh
        self._flat_mesh : SurfaceMesh = None
        self.uvs : Attribute = None # attribute on vertices

    def run(self) :
        if euler_characteristic(self.mesh)!=1:
            raise Exception("Mesh is not a topological disk. Cannot run parametrization.")

        Ybnd = self._initialize_boundary(self._bnd_mode)

        lap = operators.laplacian(self.mesh, cotan=False)
        freeInds = self.mesh.interior_vertices
        bndInds, _ = extract_border_cycle(self.mesh)

        LI = lap[freeInds, :][:, freeInds]
        LB = lap[freeInds, :][:, bndInds]

        U = linalg.spsolve(LI, -LB.dot(Ybnd[:,0]))
        V = linalg.spsolve(LI, -LB.dot(Ybnd[:,1]))

        if self._save_on_mesh:
            self.uvs = self.mesh.vertices.create_attribute("uv_coords", float, 2, dense=True)
        else:
            self.uvs = ArrayAttribute(float, len(self.mesh.vertices), 2)
        for i,v in enumerate(freeInds):
            self.uvs[v] = Vec(U[i], V[i])
        for i,v in enumerate(bndInds):
            self.uvs[v] = Vec(Ybnd[i,:])

    def _initialize_boundary(self, boundary_mode):
        n = len(self.mesh.boundary_vertices)
        Ybnd = np.zeros((n,2))
        if boundary_mode == TutteEmbedding.BoundaryMode.CIRCLE:
            for i in range(n):
                rt = cmath.rect(1., 2*pi*i/n)
                Ybnd[i,0] = rt.real
                Ybnd[i,1] = rt.imag
        elif boundary_mode == TutteEmbedding.BoundaryMode.SQUARE:
            corners = [0, n//4, n//2, (3*n)//4]
            Ybnd[corners[0], :] = [0,0]
            Ybnd[corners[1], :] = [1,0]
            Ybnd[corners[2], :] = [1,1]
            Ybnd[corners[3], :] = [0,1]
            for i in range(1, n//4):
                Ybnd[i,:] = Vec(4*i/n, 0)
            for i,v in enumerate(range(n//4+1, n//2)):
                Ybnd[v,:] = Vec(1, 4*i/n)
            for i,v in enumerate(range(n//2+1, (3*n)//4)):
                Ybnd[v,:] = Vec(1-4*i/n, 1)
            for i,v in enumerate(range((3*n)//4+1, n)):
                Ybnd[v,:] = Vec(0, 1-4*i/n)
        return Ybnd

    @property
    def flat_mesh(self):
        """
        A flat representation of the mesh where uv-coordinates are copied to xy.

        Returns:
            SurfaceMesh: the flat mesh
        """
        if self.uvs is None:
            return None
        if self._flat_mesh is None:
            # build the flat mesh : vertex coordinates are uv of original mesh
            self._flat_mesh = copy(self.mesh)
            for v in self.mesh.id_vertices:
                self._flat_mesh.vertices[v] = Vec(self.uvs[v][0], self.uvs[v][1], 0)
        return self._flat_mesh