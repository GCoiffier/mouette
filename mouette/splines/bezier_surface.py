from ..mesh.mesh_data import DataContainer, RawMeshData
from ..mesh.datatypes import *
from ..utils.argument_check import InvalidRangeArgumentError
from ..geometry import Vec
from .evaluate import de_casteljau
import numpy as np

class BezierPatch:
    def __init__(self, control_points):
        self.pts = [[Vec(x) for x in l] for l in control_points]

    @property
    def order(self):
        return (len(self.pts)-1, len(self.pts[0])-1)
    
    def evaluate_row(self,u):
        return [de_casteljau(self.pts[i],u) for i in range(len(self.pts))]

    def evaluate(self,u,v):
        return de_casteljau(self.evaluate_row(u),v)

    def as_surface(self,n1=20,n2=20):
        out = RawMeshData()
        U = np.linspace(0,1,n1)
        V = np.linspace(0,1,n2)
        uvs = out.vertices.create_attribute("uv_coords", float, 2)
        k = 0

        # evaluate vertices
        for i in range(n1):
            q = self.evaluate_row(U[i])
            for j in range(n2):
                out.vertices.append(de_casteljau(q,V[j]))
                uvs[k] = Vec(U[i],V[j])
                k+=1
                
        # add faces
        for i in range(n1-1):
            for j in range(n2-1):
                out.faces.append(( i*n1+j, i*n1+j+1, (i+1)*n1+j+1, (i+1)*n1+j))
        return SurfaceMesh(out)