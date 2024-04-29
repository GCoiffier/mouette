from ..worker import Worker

from ...mesh.datatypes import *
from ...mesh.mesh_attributes import ArrayAttribute
from ...mesh.mesh import copy

from ...geometry import Vec

from abc import abstractmethod

class BaseParametrization(Worker):

    def __init__(self, 
        name : str, 
        mesh : SurfaceMesh, 
        verbose : bool = True, 
        **kwargs
    ):
        super().__init__(name, verbose)
        self.mesh : SurfaceMesh = mesh
        self.save_on_corners : bool = kwargs.get("save_on_corners", True)
        self.uvs : ArrayAttribute = kwargs.get("uv_attr", None) # attribute on corners or vertices

        self._flat_mesh : SurfaceMesh = None

    @abstractmethod
    def run(self):
        pass

    @property
    def flat_mesh(self) -> SurfaceMesh:
        """
        A flat representation of the mesh where uv-coordinates are copied to xy.

        Returns:
            SurfaceMesh: the flat mesh
        """
        if self.uvs is None: return None
        if self._flat_mesh is None:
            # build the flat mesh : vertex coordinates are uv of original mesh
            self._flat_mesh = copy(self.mesh)
            for T in self.mesh.id_faces:
                for i,v in enumerate(self.mesh.faces[T]):
                    if self.save_on_corners:
                        self._flat_mesh.vertices[v] = Vec(self.uvs[3*T+i][0], self.uvs[3*T+i][1], 0.)
                    else:
                        self._flat_mesh.vertices[v] = Vec(self.uvs[v][0], self.uvs[v][1], 0.)
        return self._flat_mesh
