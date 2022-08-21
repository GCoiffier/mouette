from ..mesh_data import RawMeshData
from .base import Mesh

class PointCloud(Mesh):

    def __init__(self, data : RawMeshData = None):
        Mesh.__init__(self, 0, data)

    def __str__(self):
        out = "mouette.mesh.PointCloud object\n"
        out += "| {} vertices\n".format(len(self.vertices))
        return out

    def append(self,x):
        # shortcut since we can only append elements in the 'vertices' container
        self.vertices.append(x)

    @property
    def id_vertices(self):
        """
        Shortcut for range(len(self.vertices))
        """
        return range(len(self.vertices))