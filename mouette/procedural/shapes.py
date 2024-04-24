import numpy as np

from ..geometry import *
from ..attributes import mean_edge_length
from ..mesh.datatypes import *
from ..mesh.mesh_data import RawMeshData
from ..mesh.mesh import merge, _instanciate_raw_mesh_data

def cylinder(P1 : Vec, P2 : Vec, radius: float = 1., N=50, fill_caps=True) -> SurfaceMesh:
    cy = RawMeshData()
    axis = Vec.normalized(P2-P1)
    t = Vec(axis.y, -axis.x, 0.) # tangent vector
    if t.norm()<1e-6:
        t = Vec(0, axis.z, -axis.y) # another tangent vector
    t = Vec.normalized(t)
    for P in (P1,P2):
        for i in range(N):
            Pi = P + radius * rotate_around_axis(t, axis, 2*np.pi*i/N)
            cy.vertices.append(Pi)
    # caps
    if fill_caps:
        cy.vertices += [P1,P2]
        for i in range(N):
            cy.faces.append((i,(i+1)%N,2*N))
            cy.faces.append((i+N,2*N+1, (i+1)%N+N))
    # side of cylinder
    for i in range(N):
        cy.faces.append((i,  N+i, (i+1)%N))
        cy.faces.append((N+i, N+(i+1)%N, (i+1)%N))
    return SurfaceMesh(cy)

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


def torus(
    major_segments: int,
    minor_segments: int,
    major_radius: float,
    minor_radius: float,
    triangulate: bool = False
) -> SurfaceMesh:
    """Generates a torus
    From https://danielsieger.com/blog/2021/05/03/generating-primitive-shapes.html

    Args:
        major_segments (int): number of major segments
        minor_segments (int): number of minor segments
        major_radius (float): global radius of the torus
        minor_radius (float): thickness of the torus
        triangulate (bool, optional): whether to output a triangular or quadmesh. Defaults to False.

    Returns:
        SurfaceMesh: a torus
    """
    out = RawMeshData()
    # generate vertices
    for i in range(major_segments):
       for j in range(minor_segments):
            u = i / major_segments * 2 * np.pi 
            v = j / minor_segments * 2 * np.pi
            x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
            y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
            z = minor_radius * np.sin(v)
            out.vertices.append(Vec(x,y,z))

    for i in range(major_segments):
        i_next = (i+1)%major_segments
        for j in range(minor_segments):
            j_next = (j + 1) % minor_segments
            v0 = i * minor_segments + j
            v1 = i * minor_segments + j_next
            v2 = i_next * minor_segments + j_next
            v3 = i_next * minor_segments + j
            if triangulate:
                out.faces += [(v0,v1,v3), (v1,v2,v3)]
            else:
                out.faces.append((v0,v1,v2,v3))
    return _instanciate_raw_mesh_data(out, 2)