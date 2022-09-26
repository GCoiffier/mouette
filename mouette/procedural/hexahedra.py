import numpy as np

from ..geometry import Vec
from .platonic import hexahedron
from ..mesh.datatypes import *

def axis_aligned_cube(colored=False) -> SurfaceMesh:
    """generated an axis aligned cube as 6 quad faces.

       7--------6
      /|       /|
     / |      / |
    4--------5  |
    |  |     |  |
    |  3-----|--2
    | /      | /
    |/       |/
    0--------1

    Parameters:
        colored (bool, optional): if set to true, will add a colo rattribute on faces to determine. Defaults to False.

    Returns:
        SurfaceMesh: a cube
    """
    v0 = Vec(-0.5,-0.5,-0.5)
    v1 = Vec(0.5,-0.5,-0.5)
    v2 = Vec(0.5,0.5,-0.5)
    v3 = Vec(-0.5,0.5,-0.5)

    v4 = Vec(-0.5,-0.5,0.5)
    v5 = Vec(0.5,-0.5,0.5)
    v6 = Vec(0.5,0.5,0.5)
    v7 = Vec(-0.5,0.5,0.5)
    return hexahedron(v0,v1,v2,v3,v4,v5,v6,v7, colored)

def hexahedron_4pts(P1 : Vec, P2 : Vec, P3 : Vec, P4 : Vec, colored=False, volume=False) -> SurfaceMesh:
    """Generate an hexahedron given by an absolute position and three points building a basis.

    4
    |
    |  3
    | /
    |/
    1--------2

    Parameters:
        P1 to P4 (Vec): coordinates of vertices
        colored (bool, optional): if set to true, will add a color attribute on faces to determine. Defaults to False.
        volume (bool, optional): if set to true, will also generate three tetrahedra to fill the volume. Defaults to False.

    Returns:
        SurfaceMesh: [description]
    """
    X,Y = P2-P1, P3-P1
    return hexahedron(P1, P1+X, P1+X+Y, P1+Y, P4, P4+X, P4+X+Y, P4+Y, colored, volume)