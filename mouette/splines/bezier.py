from ..mesh.mesh_data import DataContainer, RawMeshData
from ..mesh.datatypes import *
from ..geometry import Vec
import numpy as np
from ..utils.argument_check import InvalidRangeArgumentError

def de_casteljau(P:list, t:float):
    """De Casteljau's algorithm for evaluating a Bezier curve B_P at some value of the parameter t

    Args:
        P (list): Control points of the Bezier curve
        t (float): parameter value

    Raises:
        InvalidRangeArgumentError: if t is not in the [0;1] interval

    Returns:
        Vec: the position in space of B_P(t) 
    """
    if not 0 <= t <= 1:
        raise InvalidRangeArgumentError("t", t, "in [0,1]")
    coeffs = [x for x in P]
    order = len(P)-1
    for j in range(order):
        for i in range(order - j):
            coeffs[i] = t*coeffs[i+1] + (1-t)*coeffs[i]
    return coeffs[0]


class BezierCurve:

    def __init__(self, control_points):
        self.pts = DataContainer([Vec(x) for x in control_points], id="control_points")

    @property
    def order(self):
        return len(self.pts)-1
    
    def evaluate(self, t):
        return de_casteljau(self.pts, t)
    
    def as_polyline(self, n_pts:int=100, custom_pos=None):
        if custom_pos is not None:
            points = custom_pos
        else:
            points = np.linspace(0,1,n_pts)
        out = RawMeshData()
        uvs = out.vertices.create_attribute("t",float)
        for it,t in enumerate(points):
            pt_t = self.evaluate(t)
            if pt_t.size == 2:
                out.vertices.append(Vec(pt_t[0], pt_t[1], 0.))
            elif pt_t.size == 3:
                out.vertices.append(pt_t)
            uvs[it] = t
        for i in range(n_pts-1):
            out.edges.append((i,i+1))
        return PolyLine(out)
        

class BezierPatch:
    def __init__(self, control_points):
        self.pts = [[Vec(x) for x in l] for l in control_points]

    @property
    def order(self) -> int:
        return (len(self.pts)-1, len(self.pts[0])-1)
    
    def _evaluate_row(self,u):
        return [de_casteljau(self.pts[i],u) for i in range(len(self.pts))]

    def evaluate(self,u,v) -> Vec:
        return de_casteljau(self._evaluate_row(u),v)

    def as_surface(self,n1=20,n2=20) -> SurfaceMesh:
        out = RawMeshData()
        U = np.linspace(0,1,n1)
        V = np.linspace(0,1,n2)
        uvs = out.vertices.create_attribute("uv_coords", float, 2)
        k = 0

        # evaluate vertices
        for i in range(n1):
            q = self._evaluate_row(U[i])
            for j in range(n2):
                out.vertices.append(de_casteljau(q,V[j]))
                uvs[k] = Vec(U[i],V[j])
                k+=1
                
        # add faces
        for i in range(n1-1):
            for j in range(n2-1):
                out.faces.append(( i*n1+j, i*n1+j+1, (i+1)*n1+j+1, (i+1)*n1+j))
        return SurfaceMesh(out)
