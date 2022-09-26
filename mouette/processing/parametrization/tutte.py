from ...mesh.datatypes import *
from ...mesh.mesh import copy
from ...mesh.mesh_attributes import Attribute, ArrayAttribute
from ...geometry import Vec
from ... import operators
from ..border import extract_border_cycle
from ..worker import Worker
from ...attributes.glob import euler_characteristic

import numpy as np
from enum import Enum
import cmath
from math import pi
from scipy.sparse import linalg


class TutteEmbedding(Worker):
    """
    from HOW TO DRAW A GRAPH, Tutte, 1963.

    Tutte's embedding garantees that the parametrization is locally injective (Floater, 1997) provided the boundary is convex.

    usage:
    ```
    tutte = TutteEmbedding(mesh)(<options>)
    ```

    or, alternatively:
    ```
    tutte = TutteEmbedding(mesh)
    tutte.run(<options>)

    options include the shape of the boundary curve, either:
    - 
    """

    class BoundaryMode(Enum):
        CIRCLE = 0
        SQUARE = 1

        @staticmethod
        def from_string(s :str):
            if "circle" in s.lower():
                return TutteEmbedding.BoundaryMode.CIRCLE
            if "square" in s.lower():
                return TutteEmbedding.BoundaryMode.SQUARE
            raise Exception(f"'{s}' does correspond to any boundary mode. Choices are : 'circle', 'square'")

    def __init__(self, mesh : SurfaceMesh, verbose:bool=True):
        super().__init__(self, verbose=verbose)
        self.mesh : SurfaceMesh = mesh
        self._flat_mesh : SurfaceMesh = None
        self.uvs : Attribute = None # attribute on vertices

    def run(self, boundary_mode : str = "circle", save_on_mesh : bool = True) :
        if euler_characteristic(self.mesh)!=1:
            raise Exception("Mesh is not a topological disk. Cannot run parametrization.")

        boundary_mode = TutteEmbedding.BoundaryMode.from_string(boundary_mode)
        Ybnd = self._initialize_boundary(boundary_mode)

        lap = operators.laplacian(self.mesh, cotan=False)
        freeInds = self.mesh.interior_vertices
        bndInds, _ = extract_border_cycle(self.mesh)

        LI = lap[freeInds, :][:, freeInds]
        LB = lap[freeInds, :][:, bndInds]

        U = linalg.spsolve(LI, -LB.dot(Ybnd[:,0]))
        V = linalg.spsolve(LI, -LB.dot(Ybnd[:,1]))

        if save_on_mesh:
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
        if self.uvs is None:
            return None
        if self._flat_mesh is None:
            # build the flat mesh : vertex coordinates are uv of original mesh
            self._flat_mesh = copy(self.mesh)
            for v in self.mesh.id_vertices:
                self._flat_mesh.vertices[v] = Vec(self.uvs[v][0], self.uvs[v][1], 0)
        return self._flat_mesh