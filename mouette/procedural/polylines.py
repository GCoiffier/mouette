from ..mesh import from_arrays
from ..mesh.datatypes import PolyLine
from ..mesh.mesh_data import RawMeshData
from ..utils import iterators
import numpy as np

def chain_of_vertices(vertices: np.ndarray, loop:bool = False) -> PolyLine:
    """Creates a polyline that links the provided vertices in order.

    Args:
        vertices (np.ndarray): vertex positions.
        loop (bool, optional): whether to link the last vertex with the first, creating a closed loop. Defaults to False.

    Returns:
        PolyLine: _description_
    """
    pl = from_arrays(vertices, raw=True)
    n = len(pl.vertices)
    if loop:
        pl.edges += [x for x in iterators.cyclic_pairs(range(n))]
    else:
        pl.edges += [x for x in iterators.consecutive_pairs(range(n))]
    return PolyLine(pl)


def vector_field(origins: np.ndarray, vectors: np.ndarray, length_mult: float = 1.) -> PolyLine:
    """Creates the representation of a vector field from an array of origin points and an array of vectors.

    Args:
        origins (np.ndarray): size (N,K) with N the number of points and K<=3 the dimension. Origin of the vectors
        vectors (np.ndarray): size (N,K) with N the number of points and K<=3 the dimension. Coordinates of each vector.
        length_mult (float, optional): factor multiplied to each vector to modulate their length for vizualisation purposes. Defaults to 1.

    Raises:
        Exception: fails if the two arrays (origins and vectors) have a different shape
        Exception: fails if one of the two arrays have dimension > 3

    Returns:
        PolyLine: the vector field represented as a polyline.
    """
    pl = RawMeshData()
    
    # Sanitize input arrays
    origins, vectors = np.array(origins), np.array(vectors)
    if origins.shape != vectors.shape:
        raise Exception(f"Origin points array and vector array have different shapes, ({origins.shape} and {vectors.shape}).")
    if origins.shape[1]<3:
        origins = np.pad(origins, ((0,0),(0,3-origins.shape[1])))
        vectors = np.pad(vectors, ((0,0),(0,3-vectors.shape[1])))
    elif origins.shape[1]!=3: 
        raise Exception("Vertex array should have shape (n,3)")
    
    n = origins.shape[0]
    for i in range(n):
        end_i = origins[i] + length_mult*vectors[i]
        pl.vertices += [origins[i], end_i]
        pl.edges.append((2*i,2*i+1))
    return PolyLine(pl)