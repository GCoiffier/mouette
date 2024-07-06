from scipy.spatial.transform import Rotation
from ..mesh.datatypes import *
from . import Vec
from .aabb import AABB
import numpy as np

def translate(mesh : Mesh, tr : Vec) -> Mesh:
    """
    Translates all vertices of the mesh by a fixed vector

    Parameters:
        mesh (Mesh): the input mesh. Can be any mesh data structure
        tr (Vec): the translation vector

    Returns:
        Mesh: the translated mesh
    """
    for i in mesh.id_vertices:
        mesh.vertices[i] += tr
    return mesh

def rotate(mesh : Mesh, rot : Rotation, orig : Vec = None) -> Mesh:
    """Rotates all vertices of the mesh by a given rotation around a given origin.

    Parameters:
        mesh (Mesh): the input mesh. Can be any mesh data structure
        rot (scipy.spatial.transform.Rotation): The rotation
        orig (Vec, optional): The origin of the rotation. If not provided, the function rotates around (0,0,0). Defaults to None.

    Returns:
        Mesh: _description_
    """
    if isinstance(rot, np.ndarray):
        assert rot.shape == (3,3)
        rot =  Rotation.from_matrix(rot)
    elif isinstance(rot, list) or isinstance(rot, tuple):
        assert len(rot)==3
        rot = Rotation.from_euler("xyz", rot)
    else:
        assert isinstance(rot, Rotation)

    if orig is None:
        orig = Vec.zeros(3)

    for i in mesh.id_vertices:
        mesh.vertices[i] = orig + rot.apply(mesh.vertices[i] - orig)
    return mesh

def scale(mesh : Mesh, factor : float, orig : Vec = None) -> Mesh:
    """Scales the mesh by a given factor around a given origin (fixed point)

    Parameters:
        mesh (Mesh): the input mesh. Can be any mesh data structure
        factor (float): scale factor. If its magnitude is < 1e-8, will print a warning.
        orig (Vec, optional): Fixed point of the scaling. If not provided, it is set at (0,0,0). Defaults to None.

    Returns:
        Mesh: the scaled input mesh.
    """
    # if abs(factor)<1e-8:
    #     print("Warning: mesh will be scaled with a very small factor ({})".format(factor))
    if orig is None:
        orig = Vec.zeros(3)
    for i in mesh.id_vertices:
        mesh.vertices[i] = orig + factor*(mesh.vertices[i] - orig)
    return mesh

def scale_xyz(mesh : Mesh, fx : float = 1., fy : float = 1., fz : float = 1., orig : Vec = None) -> Mesh:
    """Scales the mesh independently along three axes.
    If any factor as a magnitude < 1e-8, will print a warning.

    Parameters:
        mesh (Mesh): the input mesh. Can be any mesh data structure
        fx (float, optional): Scale factor along x. Defaults to 1.
        fy (float, optional): Scale factor along y. Defaults to 1.
        fz (float, optional): Scale factor along z. Defaults to 1.
        orig (Vec, optional): Fixed point of the scaling. If not provided, it is set at (0,0,0). Defaults to None.

    Returns:
        Mesh: the scaled mesh.
    """
    # for f in (fx,fy,fx):
    #     if abs(f)<1e-8:
    #         print(f"[mouette.scale] Mesh will be scaled with a very small factor ({f})")
    if orig is None:
        orig = mesh.vertices[0]
    for i in mesh.id_vertices:
        Pi = mesh.vertices[i]
        mesh.vertices[i] = orig + Vec( fx*(Pi.x - orig.x), fy*(Pi.y - orig.y), fz *(Pi.z - orig.z))
    return mesh

def fit_into_unit_cube(mesh : Mesh) -> Mesh:
    """Applies translation and global scaling for the mesh to fit inside a cube [0;1]^3

    Args:
        mesh (Mesh): the input mesh. Can be any mesh data structure

    Returns:
        Mesh: the scaled and translated mesh
    """
    bounding = AABB.of_mesh(mesh)
    sc = 1/np.max(bounding.span)
    tr = bounding.mini
    return scale(translate(mesh, -tr), sc)

def translate_to_origin(mesh : Mesh) -> Mesh:
    """Translates all vertices of the mesh so that its barycenter is at origin (0., 0., 0.)

    Parameters:
        mesh (Mesh): the input mesh

    Returns:
        Mesh: the translated mesh
    """
    return translate(mesh, -sum(mesh.vertices)/len(mesh.vertices))

def flatten(mesh : Mesh, dim : int = None) -> Mesh:
    """
    Snaps to 0 one dimension to retrieve a flat 2D mesh.

    Parameters:
        mesh (Mesh): input mesh
        dim (int, optional): The dimension to flatten. If None, the function chooses the dimension which has the smallest variance.
    Returns:
        (Mesh): the modified input mesh
    """
    if dim is None:
        variances = []
        for i in range(3):
            variances.append(np.var([p[i] for p in mesh.vertices]))
        dim = np.argmin(variances)
    for i in mesh.id_vertices:
        mesh.vertices[i][dim] = 0.
    return mesh