from ..mesh.mesh_attributes import Attribute, ArrayAttribute
from ..mesh.datatypes import *
from .misc_corners import cotangent
from .. import geometry as geom
from ..geometry import Vec
import numpy as np

@forbidden_mesh_types(PointCloud)
def edge_length(mesh : Mesh, name = "length", persistent:bool=True, dense:bool = True) -> Attribute:
    """Compute edge lengths across the mesh.

    Parameters:
        mesh (Mesh): the input mesh. PointClouds are forbidden (no edges).
        name (str, optional): name of the attribute. Defaults to "length".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True
    """
    if persistent:
        length = mesh.edges.create_attribute(name, float, dense=True)
    else:
        length = ArrayAttribute(float, len(mesh.edges)) if dense else Attribute(float)
    for e,(a,b) in enumerate(mesh.edges):
        pA,pB = mesh.vertices[a], mesh.vertices[b]
        length[e] = geom.distance(pA,pB)
    return length

@allowed_mesh_types(SurfaceMesh)
def curvature_matrices(mesh : SurfaceMesh) -> Attribute:
    """
    Curvature matrix for each edge on the mesh, as defined in 'Restricted Delaunay Triangulations and Normal Cycle', David Cohen-Steiner and Jean-Marie Morvan, 2003

    Note:
        See Curvature Frame Fields for their aggregation on triangles or vertices
    """

    data = np.zeros((len(mesh.edges), 3,3), dtype=np.float64)
    for e, (A,B) in enumerate(mesh.edges):
        T1,T2 = mesh.connectivity.edge_to_faces(A,B)
        if T1 is not None and T2 is not None:
            _,_,n1 = geom.face_basis(*(mesh.vertices[u] for u in mesh.faces[T1]))
            _,_,n2 = geom.face_basis(*(mesh.vertices[u] for u in mesh.faces[T2]))
            # angle = geom.signed_angle_2vec3D(n1,n2,n2-n1)
            angle = geom.angle_2vec3D(n1,n2)
            edge = Vec.normalized(mesh.vertices[B] - mesh.vertices[A])
            data[e,:,:] = angle * np.outer(edge,edge)
    return data

@allowed_mesh_types(SurfaceMesh)
def cotan_weights(mesh : SurfaceMesh, name="cotan_weight", persistent:bool = True, dense:bool = True)-> Attribute:
    """ Compute the cotan weights of edges.
    The weight of an edge separating T1 and T2 is the sum of cotangent of opposite angles in T1 and T2

    Parameters:
        mesh (Mesh): the input mesh.
        name (str, optional): name of the attribute. Defaults to "cotan_weight".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
        dense (bool, optional): Is the attribute dense (numpy array) or not (dict). Defaults to True
    """
    if not mesh.is_triangular():
        raise Exception("Tried to compute cotan weights on a non-triangulated mesh")

    if persistent:
        cw = mesh.edges.create_attribute(name, float, dense=dense)
    else:
        cw = ArrayAttribute(float, len(mesh.edges)) if dense else Attribute(float)

    if mesh.face_corners.has_attribute("cotan"):
        cot = mesh.face_corners.get_attribute("cotan")
    else:
        cot = cotangent(mesh,persistent=persistent)

    for e,(A,B) in enumerate(mesh.edges):
        T,iA,iB = mesh.connectivity.direct_face(A,B,True)
        if T is not None:
            cnr = mesh.connectivity.face_to_first_corner(T)+3-iA-iB
            cw[e] += cot[cnr]/2
        T,iB,iA = mesh.connectivity.direct_face(B,A,True)
        if T is not None:
            cnr = mesh.connectivity.face_to_first_corner(T)+3-iA-iB
            cw[e] += cot[cnr]/2
    return cw 