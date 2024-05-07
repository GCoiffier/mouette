from .. import geometry as geom
from ..mesh.datatypes import *
from ..utils import PriorityQueue
from ..utils.argument_check import *
from ..mesh.mesh_attributes import _BaseAttribute

def build_path(mesh : Mesh, paths):
    path_mesh = PolyLine()
    k = 0
    for l in paths.values():
        if len(l)>0:
            path_mesh.vertices.append(mesh.vertices[l[0]])
        if len(l)>1:
            for i in range(1, len(l)):
                path_mesh.vertices.append(mesh.vertices[l[i]])
                path_mesh.edges.append((k+i-1,k+i))
    return path_mesh

def _check_weight_argument(weights):
    if isinstance(weights, str):
        check_argument("weights", weights, str, ["one", "length"])
    elif not (isinstance(weights,dict) or isinstance(weights, _BaseAttribute)):
        raise InvalidArgumentTypeError("weights", type(weights), str, dict, _BaseAttribute)

@forbidden_mesh_types(PointCloud)
def shortest_path(mesh : Mesh, start : int, targets : list, weights = "length", export_path_mesh = False):
    """
    Computes the shortest path from 'start' vertex to all given targets.
    Uses Dijsktra's algorithm

    Parameters:
        mesh (Union[Polyline, SurfaceMesh, VolumeMesh]): The mesh
        start (int): Vertex index of starting point on the mesh
        targets (int | list | set): 
            Vertex index of the end point on the mesh or
            list of vertex indexes for ending points on the mesh
        weights (str | dict) : provided weights of each edge. Options are:
            - "one" : uniform weight = 1 for every edge
            - "length" : use the length of the edge
            - any dict : custom weights
            Defaults to "length".
        export_path_mesh (bool, optional): 
            If specified, will also return the path as a Polyline file. 
            Defaults to False.
    
    Returns:
        dict: target_id -> list of vertices to visit on shortest path from 'start' to 'target_id'
        If export_path_mesh is set to True, also returns a Polyline
    """

    if isinstance(targets, int):
        targets = {targets}
    else:
        targets = set(targets)

    _check_weight_argument(weights)

    # Initialize data
    if weights=="one":
        edge_length = lambda _ : 1.
    elif weights== "length":
        edge_length = lambda u,v : geom.distance(mesh.vertices[u], mesh.vertices[v])
    else:
        edge_length = lambda u,v : weights[mesh.connectivity.edge_id(u,v)]

    queue = PriorityQueue()
    visited = dict([(_i,False) for _i in mesh.id_vertices])
    path = dict([(_i,None) for _i in mesh.id_vertices])
    distance = dict([(_i,float("inf")) for _i in mesh.id_vertices])
    distance[start] = 0.

    # Run Dijkstra's algorithm
    queue.push(start, 0.)
    while not queue.empty():
        v = queue.get().x
        if visited[v] : continue
        visited[v] = True
        for nv in mesh.connectivity.vertex_to_vertices(v):
            d = distance[v] + edge_length(v,nv)
            if distance[nv] > d:
                distance[nv] = d
                path[nv] = v
            if not visited[nv]:
                queue.push(nv, distance[nv])
    
    # Build paths
    paths_list = dict([(t, []) for t in targets])
    for t in targets:
        v = t
        while v != start:
            paths_list[t].append(v)
            v = path[v]
        paths_list[t].append(start)
        paths_list[t].reverse()
    
    if export_path_mesh:
        path_mesh = build_path(mesh, paths_list)
        return paths_list, path_mesh
    return paths_list

@forbidden_mesh_types(PointCloud)
def shortest_path_to_vertex_set(mesh : PolyLine, start : int, targets : list, weights = "length", export_path_mesh = False):
    """Computes the shortest path from 'start' vertex to the closest vertex in the target set
    The idea is to add fictionnal edges between all vertices of targets to a representent vertex, with weight 0, and call Dijsktra's algorithm to reach this vertex.

    Parameters:
        mesh (Union[Polyline, SurfaceMesh, VolumeMesh]): The mesh
        start (int): Vertex index of starting point on the mesh
        targets (list): list of vertices to reach
        weights (str | dict) : provided weights of each edge. Options are:
            - "one" : uniform weight = 1 for every edge
            - "length" : use the length of the edge
            - any dict : custom weights
            Defaults to "length".
        export_path_mesh (bool, optional): 
            If specified, will also return the path as a Polyline file. 
            Defaults to False.
    
    Raises:
        Exception: No target provided if the target list is empty

    Returns:
        (int) : the index of the closest vertex from the targets set
        (list) : the list of vertices on the closest path
        If export_path_mesh is set to True, also returns a Polyline
    """
    _check_weight_argument(weights)

    if not isinstance(targets, list):
        targets = [x for x in targets]

    if len(targets)==0 :
        raise Exception("No target provided")

    TARGET = -1 # an additionnal vertex linked to every vertex in the target set

    if len(targets)==1 : 
        # nothing special to do in this case, just call shortest_path between two vertices
        if export_path_mesh:
            parent, mesh = shortest_path(mesh, start, TARGET, weights, export_path_mesh)
            return TARGET, parent[TARGET], mesh
        else:
            parent = shortest_path(mesh, start, TARGET, weights, export_path_mesh)[TARGET]
            return TARGET, parent

    # Initialize data
    # build a dict u -> (v -> d) with u and v vertices and d the weight between them
    connectivity = dict([ (u, dict()) for u in mesh.id_vertices ])
    connectivity[TARGET] = dict()
    
    if weights=="one":
        for (u,v) in mesh.edges:
            connectivity[u][v] = 1
            connectivity[v][u] = 1

    elif weights== "length":
        for (u,v) in mesh.edges:
            d = geom.distance(mesh.vertices[u], mesh.vertices[v])
            connectivity[u][v] = d
            connectivity[v][u] = d
    else:
        for e,(u,v) in enumerate(mesh.edges):
            connectivity[u][v] = weights[e]
            connectivity[v][u] = weights[e]

    for s in targets:
        # TARGET is targets[0]
        connectivity[s][TARGET] = 0
        connectivity[TARGET][s] = 0

    queue = PriorityQueue()
    visited = dict([(_i,False) for _i in mesh.id_vertices])
    parent = dict([(_i,None) for _i in mesh.id_vertices])
    distance = dict([(_i,float("inf")) for _i in mesh.id_vertices])
    visited[TARGET], parent[TARGET], distance[TARGET] = False, None, float("inf")
    distance[start] = 0.

    # Run Dijkstra's algorithm
    queue.push(start, 0.)
    while not queue.empty():
        v = queue.get().x
        if visited[v] : continue
        visited[v] = True
        for nv in connectivity[v]:
            d = distance[v] + connectivity[v][nv]
            if distance[nv] > d:
                distance[nv] = d
                parent[nv] = v
            if not visited[nv]:
                queue.push(nv, distance[nv])
    
    # Build paths
    path = []
    v = TARGET
    while v != start:
        v = parent[v]
        path.append(v)
    path.reverse()
    ind = start if not path else path[-1]

    if export_path_mesh:
        path_mesh = build_path(mesh, {TARGET : path})
        return ind, path, path_mesh

    return ind, path

@allowed_mesh_types(SurfaceMesh)
def shortest_path_to_border(mesh : SurfaceMesh, start : int, weights = "length", export_path_mesh = False):
    """
    Computes the shortest path from 'start' vertex to the boundary of the mesh.
    Call to shortest_path_to_vertex_set with the set of boundary vertices

    Parameters:
        mesh (SurfaceMesh): The mesh
        start (int): Vertex index of starting point on the mesh
        weights (str | dict | Attribute) : provided weights of each edge. Options are:
            - "one" : uniform weight = 1 for every edge
            - "length" : use the length of the edge
            - any dict or Attribute on edges : custom weights
            weights are set to 0 for edges on the boundary. Defaults to "length".
        export_path_mesh (bool, optional): 
            If specified, will also return the path as a Polyline file. 
            Defaults to False.
    Raises:
        Exception: "Mesh has no border"
            raised if the border of the mesh does not exist

    Returns:
       list: The list of vertices on shortest path
       If export_path_mesh is set to True, also returns a Polyline
    """
    if len(mesh.boundary_vertices)==0:
        raise Exception("Mesh has no border")
    
    return shortest_path_to_vertex_set(mesh, start, mesh.boundary_vertices, weights, export_path_mesh)[1:] # ignore first argument (id of vertex)