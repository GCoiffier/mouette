import numpy as np
from ..mesh.datatypes import *
from ..mesh.mesh_attributes import Attribute

@allowed_mesh_types(SurfaceMesh)
def generate_uv_colormap_vertices(
    mesh : SurfaceMesh, 
    vattr : Attribute, 
    vmin : float = None, 
    vmax : float = None
):
    """Generates the visualization of an attribute as a 1D UV-map onto the mesh, 
    so that a colormap can be used as a texture to visualize the values.
    Version for an attribute on vertices.

    Args:
        mesh (SurfaceMesh): considered mesh
        vattr (Attribute): the attribute on vertices
        vmin (float, optional): Minimum value for the map. If not provided, taken as the minimum value of the attribute. Defaults to None.
        vmax (float, optional): Maximum value for the map. If not provided, taken as the maximum value of the attribute. Defaults to None.
        res (int, optional): Resolution of the map. Defaults to 1000.
    """
    vmin = np.amin(vattr) if vmin is None else vmin
    vmax = np.amax(vattr) if vmax is None else vmax
    assert vmin <= vmax
    span = vmax - vmin
    uvs = mesh.face_corners.create_attribute("uv_coords", float, 2)

    for c,v in enumerate(mesh.face_corners):
        valv = max(min(vattr[v],vmax), vmin)
        valc = (valv - vmin) / span
        valc = max(min(valc, 0.999), 0.001)
        uvs[c] = [0, valc]
    return uvs

@allowed_mesh_types(SurfaceMesh)
def generate_uv_colormap_corners(
    mesh : SurfaceMesh, 
    cattr : Attribute, 
    vmin : float = None, 
    vmax : float = None
):
    """Generates the visualization of an attribute as a 1D UV-map onto the mesh, 
    so that a colormap can be used as a texture to visualize the values.
    Version for an attribute on face corners.

    Args:
        mesh (SurfaceMesh): considered mesh
        cattr (Attribute): the attribute on face corners
        vmin (float, optional): Minimum value for the map. If not provided, taken as the minimum value of the attribute. Defaults to None.
        vmax (float, optional): Maximum value for the map. If not provided, taken as the maximum value of the attribute. Defaults to None.
    """
    vmin = np.amin(cattr) if vmin is None else vmin
    vmax = np.amax(cattr) if vmax is None else vmax
    assert vmin <= vmax
    span = vmax - vmin
    uvs = mesh.face_corners.create_attribute("uv_coords", float, 2)

    for c in mesh.id_corners:
        valc = max(min(cattr[c],vmax), vmin)
        valc = (cattr[c] - vmin) / span
        uvs[c] = [0, valc]
    return uvs


@allowed_mesh_types(SurfaceMesh)
def generate_uv_colormap_faces(
    mesh : SurfaceMesh, 
    fattr : Attribute, 
    vmin : float = None,
    vmax : float = None
):
    """Generates the visualization of an attribute as a 1D UV-map onto the mesh, 
    so that a colormap can be used as a texture to visualize the values.
    Version for an attribute on faces.

    Args:
        mesh (SurfaceMesh): considered mesh
        cattr (Attribute): the attribute on face corners
        vmin (float, optional): Minimum value for the map. If not provided, taken as the minimum value of the attribute. Defaults to None.
        vmax (float, optional): Maximum value for the map. If not provided, taken as the maximum value of the attribute. Defaults to None.
    """
    vmin = np.amin(fattr) if vmin is None else vmin
    vmax = np.amax(fattr) if vmax is None else vmax
    assert vmin <= vmax
    span = vmax - vmin
    uvs = mesh.face_corners.create_attribute("uv_coords", float, 2)

    for c in mesh.id_corners:
        valf = max(min(fattr[mesh.connectivity.corner_to_face(c)],vmax), vmin)
        valc = (valf - vmin) / span
        uvs[c] = [0, valc]
    return uvs