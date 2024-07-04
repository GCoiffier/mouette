import numpy as np

from ..geometry import Vec
from ..mesh.datatypes import *
from ..mesh.mesh import _instanciate_raw_mesh_data
from ..mesh.mesh_data import RawMeshData

def triangle(P0: Vec, P1: Vec, P2: Vec):
    """Generates a triangle from three vertices

    Args:
        P0,P1,P2 (Vec): three points

    Returns:
        SurfaceMesh: a triangle
    """
    out = RawMeshData()
    out.vertices += [P0, P1, P2]
    out.faces.append((0,1,2))
    return _instanciate_raw_mesh_data(out, 2)

def quad(P0: Vec, P1: Vec, P2: Vec, triangulate: bool = False):
    """Generates a quad from three vertices

       P1-------
      /        /
     /        / 
    P0-------P2  

    Args:
        P0,P1,P2 (Vec): coordinates of corners. The fourth point is deduced as P2 + P1 - 2*P0
        triangulate (bool): whether to output two triangles instead of a quad. Defaults to False.
    
    Returns:
        SurfaceMesh: a quad
    """
    P0, P1, P2 = Vec(P0), Vec(P1), Vec(P2)
    P3 = P2 + P1 - P0

    out = RawMeshData()
    out.vertices += [P0, P1, P3, P2]
    if triangulate:
        out.faces += [(0,1,2), (0,2,3)]
    else:
        out.faces += [(0,1,2,3)]
    return _instanciate_raw_mesh_data(out,2)

def unit_grid(nu: int, nv: int, triangulate: bool = False, generate_uvs: bool=False):
    out = RawMeshData()
    U = np.linspace(0,1,nu)
    V = np.linspace(0,1,nv)
    if generate_uvs:
        uv_attr = out.vertices.create_attribute("uv_coords",float,2)
    for i,u in enumerate(U):
        for j,v in enumerate(V):
            out.vertices.append(Vec(u,v,0))
            if generate_uvs:
                uv_attr[i*nu+j] = Vec(u,v)
            # generate faces
            if i<nu-1 and j<nv-1:
                if triangulate: # add two triangles
                    out.faces.append((i*nu+j, i*nu+j+1, (i+1)*nu+j))
                    out.faces.append((i*nu+j+1, (i+1)*nu+j+1, (i+1)*nu+j))
                else: # add a quad
                    out.faces.append((i*nu+j, i*nu+j+1, (i+1)*nu+j+1, (i+1)*nu+j))
    return _instanciate_raw_mesh_data(out, 2)


def unit_triangle(nu,nv, generate_uvs=False):
    out = RawMeshData()
    U = np.linspace(0,1,nu)
    V = np.linspace(1,0,nv)
    if generate_uvs:
        uv_attr = out.vertices.create_attribute("uv_coords",float,2)
    npt = 0
    for j,v in enumerate(V):
        for i,u in enumerate(U):
            if i>j: break
            out.vertices.append(Vec(u,v,0))
            if generate_uvs:
                uv_attr[npt] = Vec(u,v)
                npt += 1
    for j,v in enumerate(V):
        for i,u in enumerate(U):
            if i>j or j==nv-1: break
            kpt = j*(j+1)//2 + i
            if i<j: out.faces.append((kpt,kpt+j+2,kpt+1))
            out.faces.append((kpt,kpt+j+1,kpt+j+2))
    return _instanciate_raw_mesh_data(out, 2)