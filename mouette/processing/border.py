from ..mesh.mesh_data import RawMeshData
from ..utils import keyify
from ..mesh.datatypes import *
from ..mesh.mesh_attributes import Attribute

@allowed_mesh_types(SurfaceMesh)
def extract_border_cycle(mesh : SurfaceMesh, starting_point : int = None):
    """
    Extracts a list of vertices that are on the border of the mesh.

    Parameters:
        mesh (Mesh): The mesh
        starting_point (int): Origin point. Should be on border
    
    Returns:
        (list): list of vertices on border in order
        (list): list of edges on border also in order

    Raises:
        Exception: Fails if 'starting_point' not on the border
    """
    if len(mesh.boundary_vertices)==0 : return []
    if starting_point is None:
        starting_point = mesh.boundary_vertices[0]
    if not mesh.is_vertex_on_border(starting_point) : 
        raise Exception("Starting point (vertex {}) is not on mesh border".format(starting_point))

    vborder, eborder = [starting_point], []
    point1, point2 = starting_point, mesh.connectivity.vertex_to_vertices(starting_point)[0]
    nvisited = 0
    MAX_VISITED = len(mesh.vertices)
        
    while point2 != starting_point and nvisited < MAX_VISITED:
        # while we have not come back to origin
        vborder.append(point2)
        eborder.append(mesh.connectivity.edge_id(point1, point2))
        for v in mesh.connectivity.vertex_to_vertices(point2):
            if mesh.is_vertex_on_border(v) and v!=point1:
                point1, point2 = point2, v
                break
        nvisited += 1
    eborder.append(mesh.connectivity.edge_id(point1, point2)) # add last edge to close the cycle
    return vborder, eborder

@allowed_mesh_types(SurfaceMesh)
def extract_border_cycle_all(mesh : SurfaceMesh) -> list:
    """
    Extracts all the border cycles of a mesh and returns a list of list of vertices

    Parameters:
        mesh (Mesh): The mesh
    
    Returns:
        (list): list of list of vertices
    """
    visited = dict([(i,False) for i in mesh.boundary_vertices])
    borders = []
    for v in mesh.boundary_vertices:
        if not visited[v]:
            border_v, _ = extract_border_cycle(mesh, v)
            for v2 in border_v:
                visited[v2] = True
            borders.append(border_v)
    return borders

@allowed_mesh_types(SurfaceMesh)
def extract_curve_boundary(mesh : SurfaceMesh) -> PolyLine :
    """Returns the boundary of a surface mesh as a polyline

    Args:
        mesh (SurfaceMesh): input mesh

    Returns:
        PolyLine: boundary curves of the mesh
    """
    bound = PolyLine()
    visited = Attribute(bool)
    component = bound.vertices.create_attribute("component", int)
    map_v2v = dict() # vertices have a new and different index in the boundary mesh (so that indexes are [0, n])

    ind_component = 0
    ind_vertex = 0
    for v in mesh.boundary_vertices:
        if not visited[v]:
            cycle_v, cycle_e = extract_border_cycle(mesh, v)
            bound.edges += [mesh.edges[e] for e in cycle_e]
            for v2 in cycle_v:
                visited[v2] = True
                map_v2v[v2] = ind_vertex
                ind_vertex += 1
                component[v2] = ind_component
                bound.vertices.append(mesh.vertices[v2])
            ind_component += 1

    # re order edge indexes
    for e,(A,B) in enumerate(bound.edges):
        bound.edges[e] = keyify(map_v2v[A], map_v2v[B])
    return bound, map_v2v

@allowed_mesh_types(VolumeMesh)
def extract_surface_boundary(mesh : VolumeMesh) -> SurfaceMesh :
    bound = RawMeshData()
    map_m2b = dict() # vertices have a new and different index in the boundary mesh (so that indices are [0, n-1])
    map_b2m = dict() # boundary to mesh
    vertex_set = set()
    for iF in mesh.boundary_faces:
        bound.faces.append(iF)
        for v in mesh.faces[iF]:
            vertex_set.add(v)

    # re order vertices
    for i,v in enumerate(vertex_set):
        bound.vertices.append(mesh.vertices[v])
        map_m2b[v] = i
        map_b2m[i] = v
    # apply ordering to faces
    for i, iF in enumerate(bound.faces):
        bound.faces[i] = tuple(( map_m2b[v] for v in mesh.faces[iF]))
    bound.prepare() # ordering will be propagated to edges
    return SurfaceMesh(bound), map_m2b, map_b2m
