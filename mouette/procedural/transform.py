from ..mesh.datatypes import *
from ..mesh.mesh_data import RawMeshData
from ..mesh.mesh import merge, _instanciate_raw_mesh_data

from ..attributes import face_barycenter

from .shapes import cylinder, icosphere
from ..attributes import mean_edge_length

def spherify_vertices(points : PointCloud, radius : float = 1e-2, n_subdiv=1) -> SurfaceMesh:
    """Transforms vertices of a point cloud as icospheres

    Args:
        points (PointCloud): the input point cloud
        radius (float, optional): radius of each sphere. Defaults to 1e-2.
        n_subdiv (int, optional): number of subdivisions of the icospheres. Defaults to 1.

    Returns:
        SurfaceMesh
    """
    spheres = []
    if isinstance(points, Mesh):
        what = points.vertices
    else:
        what = points
    for P in what:
        mp = icosphere(n_subdiv, P, radius)
        spheres.append(mp)
    return merge(spheres)


@forbidden_mesh_types(PointCloud)
def cylindrify_edges( mesh : PolyLine, radius: float = 5e-2, N=50) -> SurfaceMesh:
    """ Transforms edges of a polyline as cylinder surface

    Parameters:
        mesh (PolyLine): the input mesh
        radius (float, optional): Radius of the output cylinders. Defaults to 5e-2.
        N (int, optional): Number of points inside each circle of the cylinders. Defaults to 50.

    Returns:
        SurfaceMesh
    """
    if len(mesh.edges)==0: 
        return SurfaceMesh()

    L = mean_edge_length(mesh)
    cylinders = []
    for A,B in mesh.edges:
        pA,pB = mesh.vertices[A], mesh.vertices[B]
        cylinders.append(cylinder(pA,pB, L*radius, N, fill_caps=False))
    return merge(cylinders)