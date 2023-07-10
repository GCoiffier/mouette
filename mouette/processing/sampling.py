from ..mesh.datatypes import *
from ..attributes import face_normals, face_area
import numpy as np
from numpy.random import randint, random, choice

@allowed_mesh_types(SurfaceMesh)
def sample_vertices_from_surface(mesh : SurfaceMesh, n_pts, sample_normals:bool=False) -> PointCloud:
    assert mesh.is_triangular()
    NF = len(mesh.faces)
    areas = face_area(mesh).as_array()
    areas /= np.sum(areas)
    pts = np.zeros((n_pts,3))
    
    faces = choice(NF, size=n_pts, p=areas) # faces on which we take the points (probability weighted by area)
    for i,f in enumerate(faces):
        pA,pB,pC = (mesh.vertices[_v] for _v in mesh.faces[f])
        rA,rB,rC = np.random.random(3)
        pts[i,:] = (rA*pA+rB*pB+rC*pC)/(rA+rB+rC)
    
    pointcloud = PointCloud()
    pointcloud.vertices += list(pts)
    if sample_normals:
        normals = face_normals(mesh) 
        pc_normals = pointcloud.vertices.create_attribute("normals", float, 3, dense=True)
        for i,f in enumerate(faces):
            pc_normals[i] = normals[f]
    return pointcloud