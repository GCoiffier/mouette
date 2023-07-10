from . import Vec
import numpy as np

class AABB:
    """
    Axis Aligned Bounding Box
    """
    def __init__(self, pt:Vec, dim:Vec):
        self._x1 : Vec = pt
        self._x2 : Vec = pt + dim
        self._d  : Vec = dim

    @classmethod
    def of_mesh(cls, mesh : "Mesh", padding:float = 0.):
        pt_min = np.min(mesh.vertices._data, axis=0) - Vec(padding, padding, padding)
        pt_max = np.max(mesh.vertices._data, axis=0) + Vec(padding, padding, padding)
        return AABB(pt_min, pt_max-pt_min)

    @property
    def min_coords(self) -> Vec:
        return self._x1
    
    @property
    def max_coords(self) -> Vec:
        return self._x2
    
    @property
    def span(self) -> Vec:
        return self._d

    def contains_point(self,pt:Vec) -> bool:
        return (self._x1 <= pt).all() and (pt < self._x2).all()
    
    def is_empty(self)->bool:
        return np.any(self.min_coords >= self.max_coords)
    
class BB2D:
    def __init__(self, x1:float, y1:float, x2:float, y2:float):
        """
        Args:
            x1 (float): x coordinate of first point 
            y1 (float): y cooordinate of first point
            x2 (float): x coordinate of second point
            y2 (float): y coordinate of second point
        """
        self._x1 = Vec(min(x1,x2), min(y1,y2))
        self._x2 = Vec(max(x1,x2), max(y1,y2))
        self._d = self._x2 - self._x1
    
    @classmethod
    def of_mesh(cls, mesh : "Mesh", padding:float=0.):
        padding = max(0, padding) # only allow non-negative numbers
        pt_min = np.min(mesh.vertices._data, axis=0)
        pt_max = np.max(mesh.vertices._data, axis=0)
        return BB2D(pt_min[0] - padding, pt_min[1] - padding, pt_max[0] + padding, pt_max[1] + padding)

    @property
    def left(self): return self._x1.x
    @property
    def right(self): return self._x2.x
    @property
    def top(self): return self._x2.y
    @property
    def bottom(self): return self._x1.y

    @property
    def width(self): return self._d.x
    @property
    def height(self): return self._d.y

    @staticmethod
    def intersection(b1,b2):
        return BB2D(0.,0.,0.,0.)
    
    def __and__(self,other):
        return BB2D.intersection(self,other)

    @staticmethod
    def do_intersect(b1,b2) -> bool:
        return (b1.left <= b2.right and b1.right >= b2.left) and \
                (b1.bottom <= b2.top and b1.top >= b2.bottom)
    
    @staticmethod
    def union(b1,b2):
        pmin = np.min(b1._x1, b2._x1)
        pmax = np.max(b1._x2, b2._x2)
        return BB2D(pmin.x, pmin.y, pmax.x-pmin.x, pmax.y-pmin.y)
    
    def __or__(self, other):
        return BB2D.union(self,other)
    
    def contains_point(self,pt:Vec) -> bool:
        return self.left <= pt.x < self.right and self.bottom <= pt.y < self.top