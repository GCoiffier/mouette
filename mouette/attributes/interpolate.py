import numpy as np
from ..mesh.datatypes import *
from ..mesh.mesh_attributes import Attribute
from .attr_corners import corner_angles
from .attr_faces import face_area
from ..utils import check_argument


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
    """Given an attribute on faces, interpolates its value onto vertices

    Parameters:
        mesh (SurfaceMesh): the mesh
        fattr (Attribute): input face attribute
        vattr (Attribute): output face attribute
        weight (str): the way attributes are weighted in the sum.
            four possibilities :

            - uniform: every face will have weight 1
            
            - area: face have a weight proportionnal to their area
            
            - angle: face contribute to a vertex depending on the interior angle at this vertex

            - sum : like uniform but does not divide

    Returns:
        Attribute: modified vattr

    Raises:
        Exception: fails if 'weight' is not in {'uniform', 'area', 'angle', 'sum'}
    """
    weight = weight.lower()
    check_argument("weight", weight, str, {'uniform', 'area', 'angle', 'sum'})

    if weight in ("uniform", 'sum'):
        for v in mesh.id_vertices:
            v2f = mesh.connectivity.vertex_to_faces(v)
            vattr[v] = sum([fattr[f] for f in v2f])
            if weight == "uniform":
                vattr[v] /= len(v2f)

    elif weight == "area":
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

    elif weight == "angle":
        if mesh.face_corners.has_attribute("angles"):
            angles = mesh.face_corners.get_attribute("angles")
        else:
            angles = corner_angles(mesh, persistent=False)
        defects = np.zeros(len(mesh.vertices))
        for c, v in enumerate(mesh.face_corners):
            f = mesh.face_corners.adj(c)
            vattr[v] = vattr[v] + fattr[f] * angles[c]
            defects[v] = defects[v] + angles[c]
        for v in mesh.id_vertices:
            vattr[v] /= defects[v]
    return vattr

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

@allowed_mesh_types(SurfaceMesh)
def average_corners_to_vertices(
    mesh : SurfaceMesh,
    cattr : Attribute, 
    vattr : Attribute,
    weight : str = "uniform")-> Attribute:
    """Given an attribute on corners, computes a vertex attribute that is the average per vertex

    Parameters:
        mesh (SurfaceMesh): the mesh
        cattr (Attribute): input corner attribute
        vattr (Attribute): output vertex attribute
        weight (str): the way attributes are weighted in the sum.
            three possibilities:
            - uniform: every corner will have weight 1/n
            - angle: corners contribute to a vertex depending on the interior angle at this vertex
            - sum: do not consider weights and just add the value at corners
            
    Returns:
        Attribute: modified vattr

    Raises:
        Exception: fails if 'weight' is not in {'uniform', 'angle', 'sum'}
    """
    weight = weight.lower()
    check_argument("weight", weight, str, {'uniform', 'angle', 'sum'})
    
    if weight == "uniform":
        count = np.zeros(len(mesh.vertices))
        for c,v in enumerate(mesh.face_corners):
            vattr[v] = vattr[v] + cattr[c]
            count[v] += 1
        for v in mesh.id_vertices:
            vattr[v] /= count[v]

    elif weight == "sum":
        for c,v in enumerate(mesh.face_corners):
            vattr[v] = vattr[v] + cattr[c]

    elif weight == "angle":
        if mesh.face_corners.has_attribute("angles"):
            angles = mesh.face_corners.get_attribute("angles")
        else:
            angles = corner_angles(mesh, persistent=False)
        defects = np.zeros(len(mesh.vertices))
        for c, v in enumerate(mesh.face_corners):
            vattr[v] = vattr[v] + angles[c]*cattr[c]
            defects[v] += angles[c]
        for v in mesh.id_vertices:
            vattr[v] /= defects[v]

    return vattr


@allowed_mesh_types(SurfaceMesh)
def scatter_faces_to_corners(
    mesh : SurfaceMesh,
    fattr : Attribute,
    cattr : Attribute
) -> Attribute :
    """
    Given an attribute on faces, distributes its values onto corresponding corners.

    Args:
        mesh (Mesh): the input mesh
        fattr (Attribute): input face attribute
        cattr (Attribute): output face corner attribute

    Returns:
        Attribute: cattr
    """
    for F in mesh.id_faces:
        for c in mesh.connectivity.face_to_corners(F):
            cattr[c] = fattr[F]
    return cattr


@allowed_mesh_types(SurfaceMesh)
def average_corners_to_faces(
    mesh : SurfaceMesh,
    cattr : Attribute, 
    fattr : Attribute,
    weight : str = "uniform")-> Attribute:
    """Given an attribute on corners, computes a face attribute that is the average per face

    Parameters:
        mesh (SurfaceMesh): the mesh
        cattr (Attribute): input corner attribute
        fattr (Attribute): output face attribute
        weight (str): the way attributes are weighted in the sum.
            three possibilities:
            - uniform: every corner will have weight 1/n
            - angle: corners contribute to a fae depending on the interior angle
            - sum: do not consider weights and just add the value at corners
            
    Returns:
        Attribute: modified fattr

    Raises:
        Exception: fails if 'weight' is not in {'uniform', 'angle', 'sum'}
    """
    weight = weight.lower()
    check_argument("weight", weight, str, ['uniform', 'angle', 'sum'])

    if weight == "uniform":
        for F in mesh.id_faces:
            cnrF = mesh.connectivity.face_to_corners(F)
            for c in cnrF:
                fattr[F] = fattr[F] + cattr[c]/len(cnrF)

    elif weight == "sum":
        for F in mesh.id_faces:
            fattr[F] = sum([cattr[c] for c in mesh.connectivity.face_to_corners(F)])

    elif weight == "angle":
        if mesh.face_corners.has_attribute("angles"):
            angles = mesh.face_corners.get_attribute("angles")
        else:
            angles = corner_angles(mesh, persistent=False)
        sum_angles = np.zeros(len(mesh.faces))
        for F in mesh.id_faces:
            for c in mesh.connectivity.face_to_corners(F):
                fattr[F] = fattr[F] + angles[c]*cattr[c]
                sum_angles[F] += angles[c]
        for F in mesh.id_faces:
            fattr[F] /= sum_angles[F]
    return fattr



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