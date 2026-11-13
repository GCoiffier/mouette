import numpy as np
from typing import Union

from ..mesh.datatypes import *
from ..mesh.mesh_attributes import Attribute, _BaseAttribute
from .attr_corners import corner_angles
from .attr_faces import face_area
from .attr_cells import cell_volume
from ..utils import check_argument
from .attribute_utils import _mean_per_element, _weighted_broadcast

@forbidden_mesh_types(PointCloud,PolyLine)
def interpolate_vertices_to_faces(
    mesh : Mesh,
    vattr : Union[Attribute, np.ndarray],
    name : str = "interpolated",
    persistent : bool = True
) -> Union[Attribute, np.ndarray]:
    """Given an attribute on vertices, interpolates its value onto faces

    Parameters:
        mesh (Union[SurfaceMesh, VolumeMesh]): input mesh.
        vattr (Union[Attribute, np.ndarray]): input vertex attribute.
        name (str, optional): name of the newly created attribute. Ignored if persistent is set to False. Defaults to "interpolated".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Returns:
        Union[Attribute, np.ndarray]: interpolated values (either stored as Attribute on the mesh or as independent numpy array)
    """
    v_values = vattr.as_array() if isinstance(vattr, _BaseAttribute) else vattr
    f_values = _mean_per_element(mesh.faces._data, v_values)
    if persistent:
        return mesh.faces.register_array_as_attribute(name, f_values)
    return f_values


@allowed_mesh_types(SurfaceMesh)
def interpolate_faces_to_vertices(
    mesh : SurfaceMesh,
    fattr : Union[Attribute, np.ndarray],
    name : str = "interpolated",
    weight : str = "uniform",
    persistent : bool = True
)-> Union[Attribute, np.ndarray]:
    """Given an attribute on faces, interpolates its value onto vertices

    Parameters:
        mesh (SurfaceMesh): the mesh
        fattr (Union[Attribute, np.ndarray]): input face attribute
        name (str, optional): name of the newly created attribute. Ignored if persistent is set to False. Defaults to "interpolated".
        weight (str): the way attributes are weighted in the sum.
            four possibilities :  
            - uniform: every face will have weight 1  
            - area: face have a weight proportional to their area  
            - angle: face contribute to a vertex depending on the interior angle at this vertex  
            - sum : like uniform but does not divide  
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Returns:
        Union[Attribute, np.ndarray]: interpolated values (either stored as Attribute on the mesh or as independent numpy array)

    Raises:
        Exception: fails if 'weight' is not in {'uniform', 'area', 'angle', 'sum'}
    """
    weight = weight.lower()
    check_argument("weight", weight, str, {'uniform', 'area', 'angle', 'sum'})
    f_values = fattr.as_array() if isinstance(fattr, _BaseAttribute) else fattr
    v_size = 1 if len(f_values.shape)==1 else f_values.shape[1]
    v_values = np.zeros((len(mesh.vertices), v_size))

    ### Compute the weighting array   
    nf = mesh.faces._data.shape[1]
    if weight in ("uniform", "sum"):
        W = np.ones_like(mesh.faces._data, dtype=float)

    elif weight == "area":
        if mesh.faces.has_attribute("area"):
            area = mesh.faces.get_attribute("area").as_array()
        else:
            area = face_area(mesh, persistent=False)
        W = np.repeat(area, nf).reshape((-1,nf))

    elif weight == "angle":
        if mesh.face_corners.has_attribute("angles"):
            angles = mesh.face_corners.get_attribute("angles").as_array()
        else:
            angles = corner_angles(mesh, persistent=False)
        W = angles.reshape((-1, nf))

    v_values = _weighted_broadcast(mesh.faces._data, f_values, v_values, W, normalize = not (weight=="sum"))
    if persistent:
        return mesh.vertices.register_array_as_attribute(name, v_values)
    return v_values


@allowed_mesh_types(SurfaceMesh)
def interpolate_corners_to_vertices(
    mesh : SurfaceMesh,
    cattr : Union[Attribute, np.ndarray], 
    name : str = "interpolated",
    weight : str = "uniform",
    persistent : bool = True
)-> Union[Attribute, np.ndarray]:
    """Given an attribute on corners, computes a vertex attribute that is the average per vertex

    Parameters:
        mesh (SurfaceMesh): the mesh
        cattr (Union[Attribute, np.ndarray]): input corner attribute
        name (str, optional): name of the newly created attribute. Ignored if persistent is set to False. Defaults to "interpolated".
        weight (str): the way attributes are weighted in the sum.
            three possibilities:
            - uniform: every corner will have weight 1/n  
            - angle: corners contribute to a vertex depending on the interior angle at this vertex  
            - sum: do not consider weights and just add the value at corners  
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
            
    Returns:
        Union[Attribute, np.ndarray]: the interpolated attribute on vertices

    Raises:
        Exception: fails if 'weight' is not in {'uniform', 'angle', 'sum'}
    """
    weight = weight.lower()
    check_argument("weight", weight, str, {'uniform', 'angle', 'sum'})

    c_values = cattr.as_array() if isinstance(cattr, _BaseAttribute) else cattr
    v_size = 1 if len(c_values.shape)==1 else c_values.shape[1]
    v_values = np.zeros((len(mesh.vertices), v_size))
    
    nf = mesh.faces._data.shape[1]
    if weight in ("uniform", "sum"):
        W = np.ones_like(mesh.face_corners._elem, dtype=float)
    elif weight == "area":
        if mesh.faces.has_attribute("area"):
            area = mesh.faces.get_attribute("area").as_array()
        else:
            area = face_area(mesh, persistent=False)
        W = np.repeat(area, nf)

    elif weight == "angle":
        if mesh.face_corners.has_attribute("angles"):
            W = mesh.face_corners.get_attribute("angles").as_array()
        else:
            W = corner_angles(mesh, persistent=False)
    v_values = _weighted_broadcast(mesh.face_corners._elem.reshape((-1,1)), c_values, v_values, W.reshape((-1,1)), normalize = not (weight=="sum"))
    if persistent:
        return mesh.vertices.register_array_as_attribute(name, v_values)
    return v_values



@allowed_mesh_types(SurfaceMesh)
def interpolate_corners_to_faces(
    mesh : SurfaceMesh,
    cattr : Union[Attribute, np.ndarray], 
    name : str = "interpolated",
    weight : str = "uniform",
    persistent : bool = True
)-> Union[Attribute, np.ndarray]:
    """Given an attribute on corners, computes a face attribute that is the average per face.

    Parameters:
        mesh (SurfaceMesh): the mesh
        cattr (Union[Attribute, np.ndarray]): input corner attribute
        name (str, optional): name of the newly created attribute. Ignored if persistent is set to False. Defaults to "interpolated".
        weight (str): the way attributes are weighted in the sum.
            three possibilities:
            - uniform: every corner will have weight 1/n  
            - angle: corners contribute to a vertex depending on the interior angle at this vertex  
            - sum: do not consider weights and just add the value at corners 
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.
            
    Returns:
        Union[Attribute, np.ndarray]: the interpolated attribute on faces

    Raises:
        Exception: fails if 'weight' is not in {'uniform', 'angle', 'sum'}
    """
    weight = weight.lower()
    check_argument("weight", weight, str, {'uniform', 'angle', 'sum'})
    c_values = cattr.as_array() if isinstance(cattr, _BaseAttribute) else cattr
    f_size = 1 if len(c_values.shape)==1 else c_values.shape[1]
    f_values = np.zeros((len(mesh.faces), f_size))
    if weight in ("uniform", 'sum'):
        W = np.ones_like(mesh.face_corners._adj, dtype=float)
    elif weight=="angle":
        if mesh.face_corners.has_attribute("angles"):
            W = mesh.face_corners.get_attribute("angles").as_array()
        else:
            W = corner_angles(mesh, persistent=False)
    f_values = _weighted_broadcast(mesh.face_corners._adj.reshape((-1,1)), c_values, f_values, W.reshape((-1,1)), normalize = not (weight=="sum"))
    if persistent:
        return mesh.faces.register_array_as_attribute(name, f_values)
    return f_values


@allowed_mesh_types(SurfaceMesh)
def scatter_vertices_to_corners(
    mesh : Mesh,
    vattr : Union[Attribute, np.ndarray],
    name : str = "scattered",
    persistent : bool = True
) -> Union[Attribute, np.ndarray]:
    """
    Given an attribute on faces, creates an attribute over the face corners so that each corner has the value of its corresponding vertex.
    Args:
        mesh (Mesh): the input mesh
        vattr (Union[Attribute, np.ndarray]): input vertex attribute
        name (str, optional): name of the newly created attribute. Ignored if persistent is set to False. Defaults to "interpolated".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Returns:
        Attribute: cattr
    """
    v_values = vattr.as_array() if isinstance(vattr, _BaseAttribute) else vattr
    c_values = v_values[mesh.face_corners._elem]
    if persistent:
        return mesh.face_corners.register_array_as_attribute(name, c_values)
    return c_values

@allowed_mesh_types(SurfaceMesh)
def scatter_faces_to_corners(
    mesh : SurfaceMesh,
    fattr : Union[Attribute, np.ndarray],
    name  : str = "scattered",
    persistent : bool = True
) -> Union[Attribute, np.ndarray]:
    """Given an attribute on faces, creates an attribute over the face corners so that each corner has the value of its corresponding face.

    Args:
        mesh (SurfaceMesh): input mesh.
        fattr (Union[Attribute, np.ndarray]): input face attribute.
        name (str, optional): name of the newly created attribute. Ignored if persistent is set to False. Defaults to "interpolated".
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Returns:
        Union[Attribute, np.ndarray]: _description_
    """
    f_values = fattr.as_array() if isinstance(fattr, _BaseAttribute) else fattr
    c_values = f_values[mesh.face_corners._adj]
    if persistent:
        return mesh.face_corners.register_array_as_attribute(name, c_values)
    return c_values
    

@allowed_mesh_types(SurfaceMesh)
def interpolate_cells_to_vertices(
    mesh : SurfaceMesh,
    cattr : Union[Attribute, np.ndarray],
    name : str = "interpolated",
    weight : str = "uniform",
    persistent : bool = True
)-> Union[Attribute, np.ndarray]:
    """Given an attribute on cells, interpolates its value onto vertices

    Parameters:
        mesh (SurfaceMesh): the mesh
        cattr (Union[Attribute, np.ndarray]): input cell attribute
        name (str, optional): name of the newly created attribute. Ignored if persistent is set to False. Defaults to "interpolated".
        weight (str): the way attributes are weighted in the sum.
            four possibilities :  
            - uniform: every face will have weight 1  
            - volume: cell have a weight proportional to their volume  
            - sum : like uniform but does not divide  
        persistent (bool, optional): If the attribute is persistent (stored in the mesh object) or not. Defaults to True.

    Returns:
        Union[Attribute, np.ndarray]: interpolated values (either stored as Attribute on the mesh or as independent numpy array)

    Raises:
        Exception: fails if 'weight' is not in {'uniform', 'volume', 'sum'}
    """
    weight = weight.lower()
    check_argument("weight", weight, str, {'uniform', 'volume', 'sum'})
    c_values = cattr.as_array() if isinstance(cattr, _BaseAttribute) else cattr
    v_size = 1 if len(c_values.shape)==1 else c_values.shape[1]
    v_values = np.zeros((len(mesh.vertices), v_size))

    ### Compute the weighting array   
    nc = mesh.cells._data.shape[1]
    if weight in ("uniform", "sum"):
        W = np.ones_like(mesh.cells._data, dtype=float)

    elif weight == "volume":
        if mesh.cells.has_attribute("volume"):
            volume = mesh.faces.get_attribute("volume").as_array()
        else:
            volume = cell_volume(mesh, persistent=False)
        W = np.repeat(volume, nc).reshape((-1,nc))

    v_values = _weighted_broadcast(mesh.cells._data, c_values, v_values, W, normalize = not (weight=="sum"))
    if persistent:
        return mesh.vertices.register_array_as_attribute(name, v_values)
    return v_values


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
