from ..mesh.datatypes import *
from ..mesh.mesh import _instanciate_raw_mesh_data
from ..mesh.mesh_data import RawMeshData
from ..geometry import Vec
from math import sqrt

def tetrahedron(P1,P2,P3,P4, volume=False) -> SurfaceMesh:
    tet = RawMeshData()
    tet.vertices += [P1,P2,P3,P4]
    tet.faces += [(1,2,3), (0,2,3),(0,1,3),(0,1,2)]
    if volume: tet.cells.append((0,1,2,3))
    return _instanciate_raw_mesh_data(tet)

def hexahedron(P1, P2, P3, P4, P5, P6, P7, P8, colored=False, volume=False) -> SurfaceMesh:
    """Generate an hexahedron in arbitrary configuration given 8 points. Order and connectivity of points is:

       8--------7
      /|       /|
     / |      / |
    5--------6  |
    |  |     |  |
    |  4-----|--3
    | /      | /
    |/       |/
    1--------2

    Parameters:
        P1 to P8 (Vec) : coordinates of height vertices
        colored (bool, optional): if set to true, will add a color attribute on faces. Defaults to False.
        volume (bool, optional): if set to true, will also generate three tetrahedra to fill the volume. Defaults to False.

    Returns:
        SurfaceMesh: a cube
    """
    hexa = RawMeshData()
    hexa.vertices += [P1,P2,P3,P4,P5,P6,P7,P8]

    if volume:
        #hexa.cells += [(0,1,2,3),(0,5,7,4), (2,6,7,5)]
        hexa.cells += [(0,1,2,3,4,5,6,7)]
    else:
        hexa.faces += [
            (0,2,1), (0,3,2),
            (0,1,5), (0,5,4),
            (1,2,6), (1,6,5),
            (2,3,7), (2,7,6),
            (3,0,4), (3,4,7),
            (4,5,6), (4,6,7)
        ]
        if colored:
            col = hexa.faces.create_attribute("color", float, 3)
            RED,GREEN,BLUE = Vec(1.,0.,0), Vec(0.,1.,0.), Vec(0.,0.,1.)
            col[0] = RED
            col[1] = RED
            col[10] = RED
            col[11] = RED

            col[2] = GREEN
            col[3] = GREEN
            col[6] = GREEN
            col[7] = GREEN
            
            col[4] = BLUE
            col[5] = BLUE
            col[8] = BLUE
            col[9] = BLUE
    return _instanciate_raw_mesh_data(hexa)

def octahedron():
    raise NotImplementedError

def icosahedron(center : Vec = Vec(0,0,0), uv=False):
    phi = (1 + sqrt(5)) / 2
    m = RawMeshData()

    m.vertices += [ a+center for a in 
    [
        Vec(-1, phi,0),
        Vec(1, phi, 0),
        Vec(-1, -phi, 0),
        Vec(1, -phi, 0),

        Vec(0, -1, phi),
        Vec(0, 1, phi),
        Vec(0, -1, -phi),
        Vec(0, 1, -phi),

        Vec(phi, 0, -1),
        Vec(phi, 0, 1),
        Vec(-phi, 0, -1),
        Vec(-phi, 0, 1),
    ]]

    m.faces += [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1)
    ]
    return _instanciate_raw_mesh_data(m, 2)

def dodecahedron() -> SurfaceMesh:
    raise NotImplementedError