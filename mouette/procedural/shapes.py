import numpy as np

from ..geometry import *
from ..mesh.datatypes import *
from ..mesh.mesh_data import RawMeshData
from ..mesh.mesh import _instanciate_raw_mesh_data
from ..mesh.subdivision import SurfaceSubdivision
from .dual import dual_mesh
from scipy.spatial import ConvexHull

from math import sqrt, pi, cos, sin

def tetrahedron(P1: Vec, P2: Vec, P3: Vec, P4: Vec, volume: bool=False) -> SurfaceMesh:
    """Simple tetrahedron from four points

    Args:
        P1 (Vec): first point
        P2 (Vec): second point
        P3 (Vec): third point
        P4 (Vec): fourth point
        volume (bool, optional): whether to generate a cell or just a surface. Defaults to False.

    Returns:
        SurfaceMesh: _description_
    """
    tet = RawMeshData()
    tet.vertices += [P1,P2,P3,P4]
    tet.faces += [(1,2,3), (0,2,3),(0,1,3),(0,1,2)]
    if volume: tet.cells.append((0,1,2,3))
    return _instanciate_raw_mesh_data(tet)

def hexahedron(
        P1: Vec, P2: Vec, P3: Vec, P4: Vec, P5: Vec, P6: Vec, P7: Vec, P8: Vec, 
        colored: bool = False, triangulate: bool = False, volume: bool = False
    ) -> SurfaceMesh:
    """Generate an hexahedron in arbitrary configuration given 8 points. Order and connectivity of points is:

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
        P1 to P8 (Vec): coordinates of eight vertices
        colored (bool, optional): if set to true, will add a color attribute on faces. Defaults to False.
        triangulate (bool, optional): if set to true, will triangulate the faces. Defaults to False.
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
        if triangulate:
            hexa.faces += [
                (0,2,1), (0,3,2),
                (0,1,5), (0,5,4),
                (1,2,6), (1,6,5),
                (2,3,7), (2,7,6),
                (3,0,4), (3,4,7),
                (4,5,6), (4,6,7)
            ]
        else:
            hexa.faces += [
                (0,3,2,1),
                (0,1,5,4),
                (1,2,6,5),
                (2,3,7,6),
                (3,0,4,7),
                (4,5,6,7)
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

def axis_aligned_cube(colored: bool = False, triangulate: bool = False) -> SurfaceMesh:
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
        triangulate (bool, optional): if set to true, will triangulate the faces. Defaults to False.

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
    return hexahedron(v0,v1,v2,v3,v4,v5,v6,v7, colored=colored, triangulate=triangulate)

def hexahedron_4pts(P1: Vec, P2: Vec, P3: Vec, P4: Vec, colored=False, volume=False) -> SurfaceMesh:
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

def octahedron():
    """Generate a unit octahedron as the dual mesh of the unit hexahedron
    
    Reference:
        https://danielsieger.com/blog/2021/01/03/generating-platonic-solids.html
    """
    return dual_mesh(axis_aligned_cube())

def icosahedron(center : Vec = Vec(0,0,0), radius: float=1., uv=False):
    """Generate a unit icosahedron

    Args:
        center (Vec, optional): center position. Defaults to Vec(0,0,0).
        uv (bool, optional): whether to generate uv coordinates. Defaults to False.

    Returns:
        _type_: _description_
    """
    phi = (1 + sqrt(5)) / 2
    m = RawMeshData()

    m.vertices += [ radius*a+center for a in 
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
    return SurfaceMesh(m)

def dodecahedron() -> SurfaceMesh:
    """Generate a unit dodecahedron as the dual mesh of an icosahedron

    Returns:
        SurfaceMesh: a dodecahedron
    """
    return dual_mesh(icosahedron())

def cylinder(P1 : Vec, P2 : Vec, radius: float = 1., N=50, fill_caps=True) -> SurfaceMesh:
    """Generates a cylinder around the segment defined by P1 and P2

    Args:
        P1 (Vec): start of the cylinder (center of bottom face)
        P2 (Vec): end of the cylinder (center of top face)
        radius (float, optional): radius. Defaults to 1..
        N (int, optional): number of segments. Defaults to 50.
        fill_caps (bool, optional): whether to also generate faces at caps. Defaults to True.

    Returns:
        SurfaceMesh: a cylinder
    """
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

def sphere_uv( n_lat : int, n_long : int, center : Vec = Vec(0.,0.,0.), radius : float = 1.) -> SurfaceMesh:
    """Generates a sphere using classical spherical uv-coordinates 

    Parameters:
        n_lat (int): number of different latitudes for points
        n_long (int): number of different longitudes for points
        center (Vec, optional): Center position of the sphere. Defaults to Vec(0.,0.,0.).
        radius (float, optional): Radius of the sphere. Defaults to 1.

    Returns:
        SurfaceMesh: the sphere
    """
    sp = RawMeshData()
    # add two points at poles
    sp.vertices.append(center + radius*Vec(0.,0.,1.))
    sp.vertices.append(center + radius*Vec(0.,0.,-1.))

    # add other points
    theta = np.linspace(-pi/2,pi/2, n_long+2)[1:-1]
    phi = np.linspace(-pi, pi,n_lat+1)[:-1]
    for t in theta:
        for p in phi:
            sp.vertices.append(Vec(radius*cos(t)*cos(p), radius*cos(t)*sin(p), radius*sin(t)))

    # build surface as convex hull
    ch = ConvexHull(sp.vertices._data, qhull_options="QJ")
    
    # correct normals
    for face in ch.simplices:
        pA,pB,pC = (sp.vertices[_u] for _u in face)
        ray = (pA+pB+pC)/3 - center
        nrml = cross(pB-pA,pC-pA)
        if dot(ray,nrml)<0:
            sp.faces.append([face[0], face[2], face[1]])
        else:
            sp.faces.append(face)
    return _instanciate_raw_mesh_data(sp,2)

def icosphere(n_refine : int= 3, center: Vec = Vec(0.,0.,0.), radius: float = 1.) -> SurfaceMesh:
    """Generates an icosphere, that is a subdivision of the icosahedron

    Args:
        n_refine (int, optional): number of subdivisions. Defaults to 3.
        center (Vec, optional): center of the sphere. Defaults to Vec(0.,0.,0.).
        radius (float, optional): radius of the sphere. Defaults to 1..

    Returns:
        SurfaceMesh: _description_
    """
    ico = icosahedron(center, radius)
    # Subdivide
    with SurfaceSubdivision(ico,False) as subdiv:
        for _ in range(n_refine):
            subdiv.subdivide_triangles()
            for iv in subdiv.mesh.id_vertices:
                subdiv.mesh.vertices[iv] = center + radius*Vec.normalized(subdiv.mesh.vertices[iv]-center)
    return subdiv.mesh

def sphere_fibonacci( n_pts : int, radius:float =1., build_surface : bool = True) -> SurfaceMesh:
    """Generates a point cloud or a surface mesh using fibonacci sampling of a sphere.

    Parameters:
        n_pts (int): total number of vertices
        radius (float, optional): Radius of the sphere. Defaults to 1.
        build_surface (bool, optional): If specified to True, the function will also compute a triangulation of the vertices. This is obtained through a convex hull algorithm (since points lay on a convex shape, the convex hull and the Delaunay triangulation are equivalent). Defaults to True.

    Returns:
        [SurfaceMesh | PointCloud]: the generated mesh
    """

    phi = 0.5 * (1. + np.sqrt(5.))

    theta = np.zeros(n_pts)
    sphi = np.zeros(n_pts)
    cphi = np.zeros(n_pts)

    points = []
    for i in range(n_pts):
        j = 2*i - (n_pts-1) 
        theta = 2.0 * np.pi *  j / phi
        sphi = j / n_pts
        cphi = np.sqrt( (n_pts + j ) * ( n_pts - j )) /  n_pts

        x = cphi * np.sin(theta)
        y = cphi * np.cos(theta)
        z = sphi
        points.append(radius*Vec(x,y,z))

    data = RawMeshData()
    data.vertices += points
    if build_surface:
        # Triangulate points on a sphere : Delaunay is equivalent to convex hull since every points lay in the hull
        ch = ConvexHull(points, qhull_options="QJ")
        for (A,B,C) in ch.simplices:
            # Correctly orient the face with outward normal
            pA,pB,pC = (points[_x] for _x in (A,B,C))
            pcenter = (pA+pB+pC)/3
            normal = geom.cross(pB-pA,pC-pA)
            if geom.dot(pcenter,normal)<0:
                data.faces.append((A,C,B))
            else:
                data.faces.append((A,B,C))
    return _instanciate_raw_mesh_data(data)
