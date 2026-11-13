from ..data_container import DataContainer
from .base import Mesh

class PointCloud(Mesh):
    """
    A data structure for representing point clouds

    Attributes:
        vertices (DataContainer): the container for all vertices

        __str__: Representation of the object and its elements as a string.
    """

    def __init__(self, vertices: DataContainer):
        vertices = vertices or DataContainer(id="vertices")
        Mesh.__init__(self, vertices=vertices)

    def __str__(self) -> str:
        out = "mouette.mesh.PointCloud object\n"
        out += "| {} vertices\n".format(len(self.vertices))
        return out

    @property
    def id_vertices(self):
        """
        Shortcut for `range(len(self.vertices))`
        """
        return range(len(self.vertices))