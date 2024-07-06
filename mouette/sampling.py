from .mesh.datatypes import *
from .mesh import from_arrays
from .attributes import face_normals, face_area, edge_length
import numpy as np
from numpy.random import random, choice
from .geometry import Vec, AABB
from .utils import check_argument

def sample_sphere(center : Vec, radius : float, n_pts : int, return_point_cloud : bool = False):
    """Samples points uniformly on the surface of a 3D sphere

    Args:
        center (Vec): center of the sphere
        radius (float): radius of the spheres
        n_pts (int): number of points to sample
        return_point_cloud (bool, optional): whether to compile the points in a PointCloud object or return the raw numpy array. Defaults to False.

    Returns:
        PointCloud | np.ndarray: a sampled point cloud of `n_pts` points
    """
    pts = np.vstack([
        np.random.normal(0.,1., size=n_pts),
        np.random.normal(0.,1., size=n_pts),
        np.random.normal(0.,1., size=n_pts)
    ]).T
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) # uniform distribution of the unit sphere
    pts = radius*pts + center
    if return_point_cloud:
        pointcloud = PointCloud()
        pointcloud.vertices += list(pts)
        return pointcloud
    else:
        return pts

def sample_ball(center : Vec, radius : float, n_pts : int, return_point_cloud : bool = False):
    """Samples points uniformly inside a 3D ball.

    Args:
        center (Vec): center of the ball
        radius (float): radius of the ball
        n_pts (int): number of points to sample
        return_point_cloud (bool, optional): whether to compile the points in a PointCloud object or return the raw numpy array. Defaults to False.

    Returns:
        PointCloud | np.ndarray: a sampled point cloud of `n_pts` points
    """
    pts = np.vstack([
        np.random.normal(0.,1., size=n_pts),
        np.random.normal(0.,1., size=n_pts),
        np.random.normal(0.,1., size=n_pts)
    ]).T
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    R = np.random.uniform(0, radius, n_pts).reshape((n_pts,1))
    R = np.cbrt(R) # R^{1/3} for uniform distribution
    pts = pts*R + center
    if return_point_cloud:
        pointcloud = PointCloud()
        pointcloud.vertices += list(pts)
        return pointcloud
    else:
        return pts

def sample_AABB(
        box : AABB, 
        n_pts : int,
        mode : str = "uniform",
        return_point_cloud : bool = False
    ):
    """Sample a point cloud uniformly at random inside an axis-aligned bounding box. Works by sampling the unit cube and applying an affine transformation.

    Args:
        box (AABB): the domain of sampling
        n_pts (int): number of points to sample. If mode is 'grid', the function may return a slightly smaller number of points (nearest perfect n-th root).
        mode (str): sampling mode. 'uniform' or 'grid'. Uniform takes points at random, while 'grid' generates a grid of regularly spaced points.
        return_point_cloud (bool, optional): whether to compile the points in a PointCloud object or return the raw numpy array. Is ignored if the bounding box has dimension >3. Defaults to False.

    Raises:
        Exception: fails if the bounding box is empty.
        ValueError: fails if the dimension of the box is > 3 and return_point_cloud is set to True, so that no PointCloud object with dim > 3 is created.

    Returns:
        PointCloud | np.ndarray: a sampled point cloud of `n_pts` points
    """
    check_argument("mode", mode, str, ["uniform", "grid"])
    if box.is_empty():
        raise Exception(f"Received AABB {box}, is empty. Cannot sample valid points.")
    if box.dim>3 and return_point_cloud:
        raise ValueError(f"No PointCloud can be generated for a bounding box of dimension {box.dim}>3.")
    if mode=="uniform":
        points = box.mini + box.span * random((n_pts,box.dim))
    elif mode=="grid":
        res = round(np.power(n_pts, 1/box.dim))
        Xdims = (np.linspace(0,1,res) for _ in range(box.dim))
        points = np.vstack(list(map(np.ravel, np.meshgrid(*Xdims)))).T
    if return_point_cloud:
        return from_arrays(points)
    else:
        return points

@allowed_mesh_types(PolyLine)
def sample_points_from_polyline(
    mesh : PolyLine, 
    n_pts : int,
    return_point_cloud : bool = False
    ):
    """
    Sample a point cloud uniformly at random from a polyline

    Args:
        mesh (PolyLine): the input polyline
        n_pts (int): number of points to sample
        return_point_cloud (bool, optional): whether to compile the points in a PointCloud object or return the raw numpy array. Defaults to False.

    Returns:
        PointCloud | np.ndarray:  a sampled point cloud of `n_pts` points
    """
    NE = len(mesh.edges)
    sampled_pts = np.zeros((n_pts, 3))
    if NE>1:
        lengths = edge_length(mesh, persistent=False).as_array()
        lengths /= np.sum(lengths)
        edges = choice(NE, size=n_pts, p=lengths)
    else:
        edges = [0]*n_pts
    for i,e in enumerate(edges):
        pA,pB = (mesh.vertices[_v] for _v in mesh.edges[e])
        t = np.random.random()
        sampled_pts[i,:] = t*pA + (1-t)*pB
    if return_point_cloud:
        pointcloud = PointCloud()
        pointcloud.vertices += list(sampled_pts)
        return pointcloud
    else:
        return sampled_pts

@allowed_mesh_types(SurfaceMesh)
def sample_points_from_surface(
    mesh : SurfaceMesh, 
    n_pts : int, 
    return_point_cloud : bool = False, 
    return_normals : bool=False
    ):
    """
    Sample a point cloud uniformly at random from a surface mesh

    Args:
        mesh (SurfaceMesh): input mesh
        n_pts (int): number of points to sample
        return_point_cloud (bool, optional): whether to compile the points in a PointCloud object or return the raw numpy array. Defaults to False.
        return_normals (bool, optional): wether to assign the normal of the faces to sampled points. Only has effect if return_point_cloud is set to True. Defaults to False.

    Returns:
        PointCloud | np.ndarray: a sampled point cloud of `n_pts` points
        np.ndarray : the associated normals (if sample_normals was set to True)
    """
    assert mesh.is_triangular()
    NF = len(mesh.faces)
    areas = face_area(mesh, persistent=False).as_array()
    areas /= np.sum(areas)
    sampled_pts = np.zeros((n_pts,3))
    
    sampled_faces = choice(NF, size=n_pts, p=areas) # faces on which we take the points (probability weighted by area)
    if return_normals:
        normals = face_normals(mesh,persistent=False)
        sampled_normals = np.array([normals[f] for f in sampled_faces])

    for i,f in enumerate(sampled_faces):
        pA,pB,pC = (mesh.vertices[_v] for _v in mesh.faces[f])
        u1,u2 = random(2)
        r1 = 1 - np.sqrt(u1)
        r2 = u2*(1-r1)
        sampled_pts[i,:] = pA+ r1*(pB-pA) + r2*(pC-pA)
    
    if return_point_cloud:
        pointcloud = PointCloud()
        pointcloud.vertices += list(sampled_pts)
        if return_normals: 
            pc_normals = pointcloud.vertices.create_attribute("normals", float, 3, dense=True)
            pc_normals._data = sampled_normals
        return pointcloud
    else:
        if return_normals:
            return sampled_pts,sampled_normals
        return sampled_pts