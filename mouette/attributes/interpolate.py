import numpy as np
from ..mesh.datatypes import *
from ..mesh.mesh_attributes import Attribute
from .misc_corners import corner_angles
from .misc_faces import face_area

@allowed_mesh_types(SurfaceMesh)
def scatter_vertices_to_corners(
    mesh : Mesh,
    vattr : Attribute,
    cattr : Attribute
) -> Attribute :
    """
    Given an attribute on vertices, distributes its values onto corresponding corners.

    Args:
        mesh (Mesh): the input mesh
        vattr (Attribute): input vertex attribute
        cattr (Attribute): output face corner attribute

    Returns:
        Attribute: cattr
    """
    for c,v in enumerate(mesh.face_corners):
        cattr[c] = vattr[v]
    return cattr

@forbidden_mesh_types(PointCloud,PolyLine)
def interpolate_vertices_to_faces(
    mesh : Mesh,
    vattr : Attribute, 
    fattr : Attribute
) -> Attribute:
    
    """Given an attribute on vertices, interpolates its value onto faces

    Parameters:
        mesh (Union[SurfaceMesh, VolumeMesh]): the input mesh
        vattr (Attribute): input vertex attribute 
        fattr (Attribute): output face attribute
    Returns:
        Attribute: fattr
    """
    fattr.clear()
    for f,F in enumerate(mesh.faces):
        for v in F:
            fattr[f] = fattr[f] + vattr[v]
    for f,F in enumerate(mesh.faces):
        fattr[f] /= len(F)
    return fattr

@allowed_mesh_types(SurfaceMesh)
def interpolate_faces_to_vertices(
    mesh : SurfaceMesh,
    fattr : Attribute, 
    vattr : Attribute,
    weight : str = "uniform")-> Attribute:
    """Given an attribute on vertices, interpolates its value onto faces

    Parameters:
        mesh (SurfaceMesh): the mesh
        fattr (Attribute): input face attribute
        vattr (Attribute): output face attribute
        weight (str): the way attributes are weighted in the sum.
            three possibilities :

            - uniform: every face will have weight 1
            
            - area: face have a weight proportionnal to their area
            
            - angle: face contribute to a vertex depending on the interior angle at this vertex
    Returns:
        Attribute: modified vattr

    Raises:
        Exception: fails if 'weight' is not in {'uniform', 'area', 'angle'}
    """
    if weight not in {'uniform', 'area', 'angle'}:
        raise Exception(f"interpolate_vertices_to_faces: weight argument '{weight}' not recognized.\nPossibilites are {'uniform', 'area', 'angle'}.")

    if weight=="uniform":
        for v in mesh.id_vertices:
            for f in mesh.connectivity.vertex_to_faces(v):
                vattr[v] = vattr[v] + fattr[f]
        for v in mesh.id_vertices:
            vattr[v] /= len(mesh.connectivity.vertex_to_faces(v))
    
    elif weight=="area":
        if mesh.faces.has_attribute("area"):
            area = mesh.faces.get_attribute("area")
        else:
            area = face_area(mesh, persistent=False)
        total_area = np.zeros(len(mesh.vertices))
        for v in mesh.id_vertices:
            total_area[v] = 0.
            for f in mesh.connectivity.vertex_to_faces(v):
                vattr[v] = vattr[v] + fattr[f] * area[f]
                total_area[v] = total_area[v] + area[f]
        for v in mesh.id_vertices:
            vattr[v] /= total_area[v]

    elif weight=="angle":
        if mesh.face_corners.has_attribute("angles"):
            angles = mesh.face_corners.get_attribute("angles")
        else:
            angles = corner_angles(mesh, persistent=False)
        defects = np.zeros(len(mesh.vertices))
        for c, v in enumerate(mesh.face_corners):
            f = mesh.connectivity.corner_to_face(c)
            vattr[v] = vattr[v] + fattr[f] * angles[c]
            defects[v] = defects[v] + angles[c]
        for v in mesh.id_vertices:
            vattr[v] /= defects[v]
    return vattr

# @allowed_mesh_types(VolumeMesh)
# def interpolate_vertices_to_cells(mesh : Mesh, vattr : Attribute, cattr : Attribute)-> Attribute:
#     """
#     Given an attribute on vertices, interpolates its value onto cells

#     Parameters:
#         mesh (VolumeMesh): the input mesh
#         vattr (Attribute): input vertex attribute
#         cattr (Attribute): output cell attribute

#     Raises:
#         NotImplementedError: TODO
#     """
#     raise NotImplementedError