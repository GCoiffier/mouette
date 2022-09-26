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
    return _instanciate_raw_mesh_data(cy, 2)

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