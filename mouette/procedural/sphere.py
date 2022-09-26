import numpy as np
from math import pi, cos, sin
from scipy.spatial import ConvexHull

from ..geometry import Vec,cross,dot
from ..mesh.mesh_data import RawMeshData
from ..mesh.datatypes import *
from ..mesh.mesh import merge, _instanciate_raw_mesh_data

from ..geometry import transform

from .platonic import icosahedron

def sphere_uv( n_lat : int, n_long : int, center : Vec = Vec(0.,0.,0.), radius : float = 1.) -> SurfaceMesh:
    """Generates a surface mesh using uv sampling of a sphere

    Parameters:
        n_lat (int): number of different latitudes for points
        n_long (int): number of different longitudes for points
        center (Vec, optional): Center position of the sphere. Defaults to Vec(0.,0.,0.).
        radius (float, optional): Radius of the sphere. Defaults to 1.

    Returns:
        SurfaceMesh: the sphere
    """
    sp = RawMeshData()
    # add two points at poles
    sp.vertices.append(center + radius*Vec(0.,0.,1.))
    sp.vertices.append(center + radius*Vec(0.,0.,-1.))

    # add other points
    theta = np.linspace(-pi/2,pi/2, n_long+2)[1:-1]
    phi = np.linspace(-pi, pi,n_lat+1)[:-1]
    for t in theta:
        for p in phi:
            sp.vertices.append(Vec(radius*cos(t)*cos(p), radius*cos(t)*sin(p), radius*sin(t)))

    # build surface as convex hull
    ch = ConvexHull(sp.vertices._data, qhull_options="QJ")
    
    # correct normals
    for face in ch.simplices:
        pA,pB,pC = (sp.vertices[_u] for _u in face)
        ray = (pA+pB+pC)/3 - center
        nrml = cross(pB-pA,pC-pA)
        if dot(ray,nrml)<0:
            sp.faces.append([face[0], face[2], face[1]])
        else:
            sp.faces.append(face)
    return _instanciate_raw_mesh_data(sp,2)

def icosphere(n_refine : int= 3, center : Vec = Vec(0.,0.,0.), radius : float = 1.) -> SurfaceMesh :
    ico = icosahedron(center)
    ico = transform.scale(ico, radius, center)
    # Subdivide
    for _ in range(n_refine):
        raise NotImplementedError
    return ico

def spherify_vertices(points : PointCloud, radius : float = 1e-2, n_subdiv=1) -> SurfaceMesh:
    spheres = []
    if isinstance(points, Mesh):
        what = points.vertices
    else:
        what = points
    for P in what:
        mp = icosphere(n_subdiv, P, radius)
        spheres.append(mp)
    return merge(spheres)

def sphere_fibonacci( n_pts : int , build_surface : bool = True) -> SurfaceMesh:
    """Generates a point cloud or a surface mesh using fibonacci sampling of a sphere.

    Parameters:
        n_pts (int): total number of vertices
        build_surface (bool, optional): If specified to True, the function will also compute a triangulation of the vertices. This is obtained through a convex hull algorithm (since points lay on a convex shape, the convex hull and the Delaunay triangulation are equivalent). Defaults to True.

    Returns:
        [SurfaceMesh | PointCloud]: the generated mesh
    """

    phi = 0.5 * (1. + np.sqrt(5.))

    theta = np.zeros(n_pts)
    sphi = np.zeros(n_pts)
    cphi = np.zeros(n_pts)

    points = []
    for i in range(n_pts):
        j = 2*i - (n_pts-1) 
        theta = 2.0 * np.pi *  j / phi
        sphi = j / n_pts
        cphi = np.sqrt( (n_pts + j ) * ( n_pts - j )) /  n_pts

        x = cphi * np.sin(theta)
        y = cphi * np.cos(theta)
        z = sphi
        points.append(Vec(x,y,z))

    data = RawMeshData()
    data.vertices += points
    if build_surface:
        # Triangulate points on a sphere : Delaunay is equivalent to convex hull since every points lay in the hull
        ch = ConvexHull(points, qhull_options="QJ")
        data.faces += list(ch.simplices)
    return _instanciate_raw_mesh_data(data)
