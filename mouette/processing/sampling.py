from ..mesh.datatypes import *
from ..attributes import face_normals, face_area, edge_length
import numpy as np
from numpy.random import random, choice
from ..geometry.aabb import BB2D, BB3D

def sample_bounding_box_2D(
        box : BB2D, 
        n_pts : int, 
        return_point_cloud : bool = False
    ) -> PointCloud | np.ndarray:
    """Sample a point cloud uniformly at random inside an axis-aligned 2D rectangle. Works by sampling the unit square and applying an affine transformation.

    Args:
        box (BB2D): the domain of sampling
        n_pts (int): number of points to sample
        return_point_cloud (bool, optional): whether to compile the points in a PointCloud object or return the raw numpy array. Defaults to False.

    Returns:
        PointCloud | np.ndarray: a sampled point cloud of `n_pts` points
    """
    points = np.random.random((n_pts,2))
    
    ### apply affine transform
    points[:,0] = box.left   + box.width  * points[:,0]
    points[:,1] = box.bottom + box.height * points[:,1]

    if return_point_cloud:
        points = np.pad(points, ((0,0), (0,1))) ### adds a z=0 to all points
        pointcloud = PointCloud()
        pointcloud.vertices += list(points)
        return pointcloud
    else:
        return points

def sample_bounding_box_3D(
        box : BB3D, 
        n_pts : int, 
        return_point_cloud : bool = False
    ) -> PointCloud | np.ndarray:
    """Sample a point cloud uniformly at random inside an axis-aligned 3D box. Works by sampling the unit cube and applying an affine transformation.

    Args:
        box (BB3D): the domain of sampling
        n_pts (int): number of points to sample
        return_point_cloud (bool, optional): whether to compile the points in a PointCloud object or return the raw numpy array. Defaults to False.

    Returns:
        PointCloud | np.ndarray: a sampled point cloud of `n_pts` points
    """
    points = box.min_coords + box.span * random((n_pts,3))
    if return_point_cloud:
        pointcloud = PointCloud()
        pointcloud.vertices += list(points)
        return pointcloud
    else:
        return points

@allowed_mesh_types(PolyLine)
def sample_points_from_polyline(
    mesh : PolyLine, 
    n_pts : int,
    return_point_cloud : bool = False
    ) -> PointCloud | np.ndarray:
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
    ) -> PointCloud | np.ndarray:
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
        rA,rB,rC = random(3)
        sampled_pts[i,:] = (rA*pA+rB*pB+rC*pC)/(rA+rB+rC)
    
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