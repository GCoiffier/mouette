from . import Vec
import numpy as np
from ..utils.argument_check import check_argument
from .geometry import norm


class AABB:

    class IncompatibleDimensionError(Exception):
        def __init__(self, message):
            super().__init__(message)

    def __init__(self, p_min, p_max):
        """Axis Aligned Bounding Box in n dimensions.

        Args:
            p_min (Iterable): minimal values for each dimension
            p_max (Iterable): maximal values for each dimension

        Raises:
            Exception: fails if p_min and p_max have different sizes (inconsistent dimension)
        """
        self._p1 = Vec(p_min)
        self._p2 = Vec(p_max)
        if self._p1.size != self._p2.size:
            raise Exception("AABB: received two initial arrays of a different dimension!")
    
    def __repr__(self):
        return f"AABB | {self._p1} -> {self._p2}"

    @property
    def dim(self):
        """Dimension of the axis-aligned bounding box

        Returns:
            _type_: _description_
        """
        return self._p1.size

    @classmethod
    def unit_cube(cls, dim:int, centered:bool = False) -> "AABB":
        """Computes the unit bounding box [0,1]^n or [-0.5, 0.5]^n

        Args:
            dim (int): dimension of the BB to build
            centered (bool, optional): whether to generate [0,1]^n (False) or [-0.5;0.5]^n (True). Defaults to False.

        Returns:
            AABB: a hypercube of side length 1
        """
        if centered:
            return AABB(np.full(dim, -0.5), np.full(dim, 0.5))
        else:
            return AABB(np.zeros(dim, dtype=float), np.ones(dim, dtype=float))

    @classmethod
    def infinite(cls, dim:int) -> "AABB":
        """Computes a bounding box with bounds at infinity, containing the whole space R^n

        Args:
            dim (int): dimension of the BB to build

        Returns:
            AABB: A bounding box containing all of R^n
        """
        return AABB(np.full(dim, -np.inf), np.full(dim, np.inf))

    @classmethod
    def of_points(cls, points: np.ndarray, padding: float = 0.) -> "AABB":
        """Computes the bounding box of a set of nD points.

        Args:
            points (np.ndarray): input points
            padding (float, optional): slack to be added between the mesh and the box. Defaults to 0 for a tight bounding box.

        Returns:
            AABB: nD bounding box of the points
        """
        points = np.array(points)
        if len(points.shape) != 2:
            raise Exception("Expected an array of points of size (N,d) with N being the number of points and d their dimension")
        pad = np.full(points.shape[1], padding)
        pt_min = np.min(points, axis=0) - pad
        pt_max = np.max(points, axis=0) + pad
        return AABB(pt_min, pt_max)

    @classmethod
    def of_mesh(cls, mesh: "Mesh", padding: float = 0.) -> "AABB":
        """Computes the 3D bounding box of all vertices of a mesh

        Args:
            mesh (Mesh): input mesh. Can be of any type (only the 'vertices' container is accessed)
            padding (float, optional): slack to be added between the mesh and the box. Defaults to 0 for a tight bounding box.

        Returns:
            AABB: 3D bounding box of the vertices of the mesh
        """
        pad = Vec(padding, padding, padding)
        pt_min = np.min(mesh.vertices._data, axis=0) - pad
        pt_max = np.max(mesh.vertices._data, axis=0) + pad
        return AABB(pt_min, pt_max)

    @property
    def mini(self) -> Vec:
        """Minimum coordinates of any point inside the box
        The box is an axis-aligned hexahedron which opposite corners are mini and maxi

        Returns:
            Vec: minimum coordinates
        """
        return self._p1
    
    @property
    def maxi(self) -> Vec:
        """Maximum coordinates of any point inside the box. 
        The box is an axis-aligned hexahedron which opposite corners are mini and maxi

        Returns:
            Vec: maximum coordinates
        """
        return self._p2
    
    @property
    def span(self) -> Vec:
        """Dimensions of the box

        Returns:
            Vec: dimensions of the box
        """
        return self._p2 - self._p1

    @property
    def center(self) -> Vec:
        """Coordinates of the center point

        Returns:
            Vec: center point
        """
        return (self._p1 + self._p2)/2

    @staticmethod
    def intersection(b1: "AABB", b2: "AABB") -> "AABB":
        """
        Computes the intersection bounding box of two bounding boxes

        Args:
            b1 (AABB): first bounding box
            b2 (AABB): second bounding box

        Raises:
            AABB.IncompatibleDimensionError: fails if b1 and b2 have different dimensions

        Returns:
            AABB: a bounding box representing the intersection (may be empty).
        """
        if b1.dim != b2.dim: 
            raise AABB.IncompatibleDimensionError(f"Bounding boxes have different dimensions ({b1.dim} and {b2.dim}) : intersection impossible")
        
        return AABB(np.maximum(b1.mini, b2.mini), np.minimum(b1.maxi, b2.maxi))
    
    def __and__(self,other):
        return AABB.intersection(self,other)

    @staticmethod
    def do_intersect(b1: "AABB", b2: "AABB") -> bool:
        """
        Intersection test between two bounding boxes

        Args:
            b1 (AABB): first bounding box
            b2 (AABB): second bounding box

        Raises:
            AABB.IncompatibleDimensionError: fails if b1 and b2 have different dimensions
            
        Returns:
            bool: whether the two BB intersect
        """
        if b1.dim != b2.dim: 
            raise AABB.IncompatibleDimensionError(f"Bounding boxes have different dimensions ({b1.dim} and {b2.dim}): intersection impossible")
    
        predicates = [b1.mini[i] <= b2.maxi[i] and b1.maxi[i] >= b2.mini[i] for i in range(b1.dim)]
        return np.all(predicates)
    
    @staticmethod
    def union(b1: "AABB", b2: "AABB") -> "AABB":
        """Computes the union bounding box of two bounding boxes

        Args:
            b1 (AABB): first bounding box
            b2 (AABB): second bounding box

        Raises:
            AABB.IncompatibleDimensionError: fails if b1 and b2 have different dimensions
        
        Returns:
            AABB: a bounding box representing the union
        """
        if b1.dim != b2.dim: 
            raise AABB.IncompatibleDimensionError(f"Bounding boxes have different dimensions ({b1.dim} and {b2.dim}): union impossible")
        pmin = np.min((b1._p1, b2._p1), axis=0)
        pmax = np.max((b1._p2, b2._p2), axis=0)
        return AABB(pmin, pmax)
    
    def __or__(self, other):
        return AABB.union(self,other)

    def pad(self, pad:Vec):
        """Enlarges the bounding box by adding `pad` on each dimension on each side. Does nothing if the padding values are negative.

        Args:
            pad (float | iterable): Additional dimensions to be added. If a float is provided, will assume that padding is the same for each dimensions. If an array is provided, its size should match the dimension of the box. Values are clamped to be >=0.
        """
        if isinstance(pad,float):
            pad = np.full(self.dim, pad)
        else:
            pad = Vec(pad)
            if pad.size != self.dim: raise AABB.IncompatibleDimensionError(f"Padding vector has a different dimension ({pad.size}) than bounding box ({self.dim})") 
        pad = np.maximum(pad, 0)
        self._p1 -= pad
        self._p2 += pad

    def contains_point(self, pt: Vec) -> bool:
        """Point - bounding box intersection predicate.
        If the point is on the boundary of the box, the convention is as follows: inclusive if the point touches the min coordinates, exclusive for the max coordinates

        Args:
            pt (Vec): a query position
        
        Raises:
            AABB.IncompatibleDimensionError: fails if the point has a different dimension than the bounding box.

        Returns:
            bool: whether the point 'pt' is inside the bounding box.
        """
        pt = Vec(pt)
        if pt.size != self.dim:
            raise AABB.IncompatibleDimensionError(f"Point has a different dimension({pt.size}) than bounding box ({self.dim})")
        return (self._p1 <= pt).all() and (pt < self._p2).all()
    
    def project(self, pt:Vec) -> Vec:
        """Computes the closest point from point 'pt' in the bounding box in Euclidean distance

        Args:
            pt (Vec): the point to project
        
        Raises:
            AABB.IncompatibleDimensionError: fails if the point has a different dimension than the bounding box.

        Returns:
            Vec: the projected point
        """
        pt = Vec(pt)
        if pt.size != self.dim:
            raise AABB.IncompatibleDimensionError(f"Point has a different dimension({pt.size}) than bounding box ({self.dim})")
        if self.contains_point(pt): return pt # pt is its own projection
        return np.maximum(self.mini, np.minimum(self.maxi, pt))

    def distance(self, pt:Vec, which="l2") -> float:
        """Computes the distance from a point to the bounding box.

        Args:
            pt (Vec): coordinates of the point
            which (str, optional): which distance to consider. Choices are 'l2', 'l1' or 'linf'. Defaults to "l2".

        Raises:
            AABB.IncompatibleDimensionError: fails if the point has a different dimension than the bounding box.
            
        Returns:
            float: the distance from the point to the bounding box
        """
        pt = Vec(pt)
        if pt.size != self.dim:
            raise AABB.IncompatibleDimensionError(f"Point has a different dimension({pt.size}) than bounding box ({self.dim})")
        check_argument("which", which, str, ["l2", "l1", "linf"])
        vec = np.maximum(np.maximum(self.mini - pt, pt - self.maxi), 0.)
        return norm(vec,which)

    def is_empty(self)->bool:
        """Tests if the bounding box encloses an empty domain

        Returns:
            bool: whether the bounding box is empty
        """
        return np.any(self.mini >= self.maxi)