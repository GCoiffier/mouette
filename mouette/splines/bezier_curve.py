from ..mesh.mesh_data import DataContainer, RawMeshData
from ..mesh.datatypes import *
from ..geometry import Vec
import numpy as np
from .evaluate import de_casteljau

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
        
