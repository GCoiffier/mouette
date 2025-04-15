from ..mesh.datatypes import *
from ..attributes import mean_edge_length
from ..utils import Logger
from .connection import SurfaceConnectionFaces
from ..operators import gradient, laplacian, area_weight_matrix

import numpy as np
import scipy.sparse as sp

class HeatMethodDistance(Logger):
    """
    Computation of distances from a set of points on a surface mesh, using the heat method.

    References:
        [1] _The Heat Method for Distance Computation_, Crane et al. (2017)

    Example:
        [https://github.com/GCoiffier/mouette/blob/main/examples/heat_method_distance.py](https://github.com/GCoiffier/mouette/blob/main/examples/heat_method_distance.py)
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
        mesh : SurfaceMesh, 
        save_on_mesh: bool = True, 
        diffuse_coeff : float = 1., 
        verbose = False, 
        **kwargs
    ):
        """
        Args:
            mesh (SurfaceMesh): The input surface mesh
            diffuse_coeff (float, optional): Coefficient used in the diffusion computation of the heat. Higher coefficient leads to more regular but less precise results. This coefficient is multiplied by the mean edge length squared to account for global scaling. Defaults to 1.
            verbose (bool, optional): verbose output. Defaults to False.

        Keyword Args:
            custom_connection (SurfaceConnectionFaces, optional): custom local bases provided to the solver. If not provided, bases will be computed automatically. Defaults to None.
        
        Attributes:
            conn (SurfaceConnectionFaces): the local bases used to compute gradients and differential operators
        
        Raises:
            AssertionError: Fails if the input mesh is not triangular
        """
        super().__init__("HeatMethod", verbose)
        assert mesh.is_triangular()
        self.mesh = mesh
        self.save_on_mesh : bool = save_on_mesh
        self.conn : SurfaceConnectionFaces = kwargs.get("custom_connection", None)
        self.diffuse_coeff = diffuse_coeff
        self.initialized : bool = False

        self._diffuse_solve = None
        self._lap_op = None # laplacian on the surface
        self._grad_op = None # gradient operator on the surface (vertices scalars -> faces vectors)
        self._div_op = None # divergence operator (faces vectors -> vertices scalars)

    def _initialize(self):
        self.log("Initialize")
        # Compute local bases in each triangles
        if self.conn is None:
            self.conn = SurfaceConnectionFaces(self.mesh)

        # Computes Laplacian and precomputes Cholesky decomposition for sparse solve
        self._lap_op = laplacian(self.mesh)
        M = area_weight_matrix(self.mesh)
        t = self.diffuse_coeff * mean_edge_length(self.mesh)**2
        self._diffuse_solve = sp.linalg.factorized(- M - t*self._lap_op)
        self._grad_op = gradient(self.mesh, self.conn).tocsc() # gradient operator
        self._div_op = self._grad_op.conjugate().transpose() #@ area_weight_matrix_faces(self.mesh)
        self.initialized = True

    def get_distance(self, source_points : list, return_gradients : bool = False) -> np.ndarray:
        """Runs the heat method solve to compute distances to points given.

        Args:
            source_points (list): a list of points for which the distance will be zero.
            return_gradients (bool, optional): whether to also return gradients per face. Defaults to False.

        Returns:
            np.ndarray: a array of size |V| containing the distance to the source points for each vertex. If `return_gradients` is True, will also return the gradients as a complex value per face.
        """
        if not self.initialized: self._initialize()

        self.log("Solve diffusion problem")
        heat0 = np.zeros(len(self.mesh.vertices))
        for v in source_points:
            heat0[v] = 1. # add a Diract delta at each source point
        heat_diffused = self._diffuse_solve(heat0)

        self.log("Compute gradients")
        gradients = self._grad_op @ heat_diffused
        for f in self.mesh.id_faces: # normalize gradients
            gradients[f] /= abs(gradients[f])

        self.log("Integrate gradients via a Poisson problem")
        freeInds = [x for x in self.mesh.id_vertices if x not in set(source_points)]   
        lapI = self._lap_op[freeInds,:][:,freeInds]
        distance = np.zeros(len(self.mesh.vertices))
        distance[freeInds] = sp.linalg.spsolve(lapI, np.real(self._div_op @ gradients)[freeInds])
        if return_gradients:
            return distance, gradients
        return distance
    
    def get_distance_to_boundary(self, return_gradients : bool = False) -> np.ndarray:
        """Computes the distance to the boundary for each vertex of the mesh. 
        A shortcut for `get_distance(mesh.boundary_vertices)`

        Args:
            return_gradients (bool, optional): whether to also return gradients per face. Defaults to False.

        Raises:
            Exception: Fails if the mesh has no boundary

        Returns:
            np.ndarray: a array of size |V| containing the distance to the source points for each vertex. If `return_gradients` is True, will also return the gradients as a complex value per face.
        """
        if len(self.mesh.boundary_vertices)==0:
            raise Exception("Mesh has no boundary")
        return self.get_distance(self.mesh.boundary_vertices, return_gradients)



# class VectorHeatMethod(Worker):

#     @allowed_mesh_types(SurfaceMesh)
#     def __init__(self,  mesh: SurfaceMesh, usigned: bool = False, verbose = False, **kwargs):
#         super().__init__("VectorHeatMethod", verbose)
#         self.mesh = mesh

#     def run(self, curve : Attribute, curve_normals : Attribute):
#         return
