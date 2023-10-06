from . import Vec
import numpy as np
    
class BB2D:
    """
    Axis aligned bounding box in 2 dimensions
    """
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
        """Computes the bounding box of all vertices of a mesh, projected into the xy-plane.

        Args:
            mesh (Mesh): input mesh. Can be of any type (only the 'vertices' container is accessed)
            padding (float, optional): slack to be added between the mesh and the box. Defaults to 0 for a tight bounding box.

        Returns:
            BB2D: bounding box of the vertices of the mesh
        """
        padding = max(0, padding) # only allow non-negative numbers
        pt_min = np.min(mesh.vertices._data, axis=0)
        pt_max = np.max(mesh.vertices._data, axis=0)
        return BB2D(pt_min[0] - padding, pt_min[1] - padding, pt_max[0] + padding, pt_max[1] + padding)

    @property
    def left(self) -> float:
        """
        Returns:
            float
        """
        return self._x1.x
    @property
    def right(self) -> float:
        """
        Returns:
            float
        """
        return self._x2.x
    @property
    def top(self) -> float:
        """
        Returns:
            float
        """
        return self._x2.y
    @property
    def bottom(self) -> float:
        """
        Returns:
            float
        """
        return self._x1.y
    @property
    def width(self) -> float:
        """
        Returns:
            float
        """
        return self._d.x
    @property
    def height(self) -> float:
        """
        Returns:
            float
        """ 
        return self._d.y
    @property
    def center(self) -> Vec:
        """Coordinates of the center point

        Returns:
            Vec: center point
        """
        return (self._x1 + self._x2)/2

    @staticmethod
    def intersection(b1,b2):
        """Computes the intersection bounding box of two bounding boxes

        Args:
            b1 (BB2D): first bounding box
            b2 (BB2D): second bounding box

        Returns:
            BB2D: a bounding box representing the intersection
        """
        if BB2D.do_intersect(b1,b2):
            if b1.left < b2.left:
                x1,x2 = b2.left, b1.right
            else:
                x1,x2 = b1.left, b2.right
            if b1.bottom < b2.bottom:
                y1,y2 = b2.bottom, b1.top
            else:
                y1,y2 = b1.bottom, b2.top
            return BB2D(x1,y1,x2,y2)
        return None
    
    def __and__(self,other):
        return BB2D.intersection(self,other)

    @staticmethod
    def do_intersect(b1,b2) -> bool:
        """
        Intersection test between two bounding boxes

        Args:
            b1 (BB2D): first bounding box
            b2 (BB2D): second bounding box

        Returns:
            bool: whether the two BB intersect
        """
        return (b1.left <= b2.right and b1.right >= b2.left) and \
                (b1.bottom <= b2.top and b1.top >= b2.bottom)
    
    @staticmethod
    def union(b1,b2):
        """Computes the union bounding box of two bounding boxes

        Args:
            b1 (BB2D): first bounding box
            b2 (BB2D): second bounding box

        Returns:
            BB2D: a bounding box representing the union
        """
        pmin = np.min((b1._x1, b2._x1), axis=0)
        pmax = np.max((b1._x2, b2._x2), axis=0)
        return BB2D(pmin[0], pmin[1], pmax[0], pmax[1])
    
    def __or__(self, other):
        return BB2D.union(self,other)
    
    def contains_point(self,pt:Vec) -> bool:
        """Point - bounding box intersection predicate.
        If the point is on the boundary of the box, the convention is as follows: inclusive if the point touches the min coordinates, exclusive for the max coordinates

        Args:
            pt (Vec): a query position

        Returns:
            bool: whether the point 'pt' is inside the bounding box.
        """
        return self.left <= pt[0] < self.right and self.bottom <= pt[1] < self.top
    
    def is_empty(self)->bool:
        """Tests if the bounding box encloses an empty domain

        Returns:
            bool: whether the bounding box is empty
        """
        return np.any(self._x1 >= self._x2)

class BB3D:
    """
    Axis Aligned Bounding Box
    """
    def __init__(self, *args):
        if len(args)==6:
            x1,y1,z1,x2,y2,z2 = args
        elif len(args)==2:
            x1,y1,z1 = args[0]
            x2,y2,z2 = args[1]
        else:
            raise Exception("BB3D expected either (x1,y1,z1,x2,y2,z2) or ((x1,y1,z1), (x2,y2,z2)) as argument.")
        self._x1 : Vec = Vec(x1,y1,z1)
        self._x2 : Vec = Vec(x2,y2,z2)
        self._d  : Vec = self._x2 - self._x1

    @classmethod
    def of_mesh(cls, mesh : "Mesh", padding:float = 0.):
        """Computes the bounding box of all vertices of a mesh

        Args:
            mesh (Mesh): input mesh. Can be of any type (only the 'vertices' container is accessed)
            padding (float, optional): slack to be added between the mesh and the box. Defaults to 0 for a tight bounding box.

        Returns:
            BB3D: bounding box of the vertices of the mesh
        """
        pad = Vec(padding, padding, padding)
        pt_min = np.min(mesh.vertices._data, axis=0) - pad
        pt_max = np.max(mesh.vertices._data, axis=0) + pad
        return BB3D(pt_min, pt_max)

    @property
    def min_coords(self) -> Vec:
        """Minimum coordinates of any point inside the box
        The box is an axis-aligned hexahedron which opposite corners are min_coords and max_coords

        Returns:
            Vec: minimum coordinates
        """
        return self._x1
    
    @property
    def max_coords(self) -> Vec:
        """Maximum coordinates of any point inside the box. 
        The box is an axis-aligned hexahedron which opposite corners are min_coords and max_coords

        Returns:
            Vec: maximum coordinates
        """
        return self._x2
    
    @property
    def span(self) -> Vec:
        """Three dimensions of the box

        Returns:
            Vec: dimensions of the box
        """
        return self._d

    @property
    def center(self) -> Vec:
        """Coordinates of the center point

        Returns:
            Vec: center point
        """
        return (self._x1 + self._x2)/2

    @staticmethod
    def intersection(b1,b2):
        """Computes the intersection bounding box of two bounding boxes

        Args:
            b1 (BB3D): first bounding box
            b2 (BB3D): second bounding box

        Returns:
            BB3D: a bounding box representing the intersection
        """
        if BB3D.do_intersect(b1,b2):
            pmin = Vec.zeros(3)
            pmax = Vec.zeros(3)
            for i in range(3):
                if b1.min_coords[i] < b2.min_coords[i]:
                    pmin[i], pmax[i] = b2.min_coords[i], b1.max_coords[i]
                else:
                    pmin[i], pmax[i] = b1.min_coords[i], b2.max_coords[i]
            return BB3D(pmin,pmax)
        return None
    
    def __and__(self,other):
        return BB3D.intersection(self,other)

    @staticmethod
    def do_intersect(b1,b2) -> bool:
        """
        Intersection test between two bounding boxes

        Args:
            b1 (BB3D): first bounding box
            b2 (BB3D): second bounding box

        Returns:
            bool: whether the two BB intersect
        """
        predicates = [b1.min_coords[i] <= b2.max_coords[i] and b1.max_coords[i] >= b2.min_coords[i] for i in range(3)]
        return np.all(predicates)
    
    @staticmethod
    def union(b1,b2):
        """Computes the union bounding box of two bounding boxes

        Args:
            b1 (BB3D): first bounding box
            b2 (BB3D): second bounding box

        Returns:
            BB3D: a bounding box representing the union
        """
        pmin = np.min((b1._x1, b2._x1), axis=0)
        pmax = np.max((b1._x2, b2._x2), axis=0)
        return BB3D(pmin, pmax)
    
    def __or__(self, other):
        return BB3D.union(self,other)

    def contains_point(self,pt:Vec) -> bool:
        """Point - bounding box intersection predicate.
        If the point is on the boundary of the box, the convention is as follows: inclusive if the point touches the min coordinates, exclusive for the max coordinates

        Args:
            pt (Vec): a query position

        Returns:
            bool: whether the point 'pt' is inside the bounding box.
        """
        return (self._x1 <= pt).all() and (pt < self._x2).all()
    
    def is_empty(self)->bool:
        """Tests if the bounding box encloses an empty domain

        Returns:
            bool: whether the bounding box is empty
        """
        return np.any(self.min_coords >= self.max_coords)
    
    