from .worker import Worker

from ..geometry import distance
from ..mesh.datatypes import *
from ..mesh.mesh_data import RawMeshData
from ..mesh.mesh_attributes import *
from ..utils import keyify, UnionFind, PriorityQueue, consecutive_pairs
from .paths import *
from .trees import FaceSpanningForest 
from .. import attributes, utils 
from collections import deque
from typing import Iterable
from enum import Enum

class SurfaceMeshCutter(Worker):

    def __init__(self, mesh: SurfaceMesh, verbose: bool = False, **kwargs):
        """
        Args:
            mesh (SurfaceMesh): input mesh
            verbose (bool, optional): verbose mode. Defaults to False.

        Attributes:
            cut_edges (set): indices of edges that were cut
            cut_mesh (SurfaceMesh): a copy of the mesh where specified edges have been cut
        """
        
        super().__init__(kwargs.get("name", "MeshCutter"), verbose)
        self.input_mesh : SurfaceMesh = mesh
        self.cut_mesh : SurfaceMesh = None
        
        self.cut_edges : set = None # ids of edges of input mesh that were cut 

        # /!\ global ordering of vertices/edges is likely to change. Only triangles remain the same
        self._ref_vertex : dict = None # split vertex -> original vertex
        self._cut_graph : PolyLine = None

    def cut(self, edges_to_cut: Iterable):
        """Cut the mesh. Alias for `SurfaceMeshCutter.run`"""
        return self.run(edges_to_cut)

    def run(self, edges_to_cut: Iterable):
        """
        Cut the mesh. Fills the attributes and builds `cut_mesh`, which is a copy of the input mesh with corresponding edges disconnected

        Args:
            edges_to_cut (Iterable): containers of all the indices of edges to be cut
        """
        self.cut_edges = set([x for x in edges_to_cut if not self.input_mesh.is_edge_on_border(*self.input_mesh.edges[x])])        
        self._ref_vertex = dict()

        self.cut_mesh = RawMeshData()
        self.cut_mesh.vertices += self.input_mesh.vertices
        self.cut_mesh.faces += self.input_mesh.faces

        ### Duplicate vertices adjacent to cut edges. N-1 copies of a vertex should be created if it is adjacent to N cuts
        cut_degree = dict()
        for e in self.cut_edges:
            A,B = self.input_mesh.edges[e]
            cut_degree[A] = cut_degree.get(A, -1)+1
            cut_degree[B] = cut_degree.get(B, -1)+1

        duplicates = dict()
        for v in cut_degree:
            if self.input_mesh.is_vertex_on_border(v): cut_degree[v] +=1
            if cut_degree[v]==0: continue # no copies needed (vertex at the end of a cut => 1-ring still connected)
            duplicates[v] = [v]
            for _ in range(cut_degree[v]):
                new_v = len(self.cut_mesh.vertices)
                self._ref_vertex[new_v] = v
                duplicates[v].append(new_v)
                self.cut_mesh.vertices.append(self.input_mesh.vertices[v])

        ### Reassign face indices
        for v in cut_degree:
            if cut_degree[v]==0: continue
            ind = 0
            ndup = len(duplicates[v])
            for c in self.input_mesh.connectivity.vertex_to_corners(v):
                # corners are ordered counter-clockwise aroud v
                current_v = duplicates[v][ind]
                F = self.input_mesh.face_corners.adj(c)
                self.cut_mesh.faces[F] = utils.replace_in_list(self.cut_mesh.faces[F], v, current_v)
                e = self.input_mesh.connectivity.edge_id(v,self.input_mesh.connectivity.corner_to_half_edge(c)[1])
                if e in self.cut_edges: # switch to next version of v upon passing a cut edge
                    ind = (ind+1)%ndup
        self.cut_mesh = SurfaceMesh(self.cut_mesh) # finalize creation

    def duplicated_vertices(self, v:int) -> set:
        """Given the index v of a vertex in the input mesh, returns the set of all copies of v in the cut mesh.

        Args:
            v (int): index of a vertex in the input mesh

        Returns:
            set[int]: all vertex indices corresponding to copies of v in the cut mesh
        """
        return set(self.cut_mesh.face_corners[c] for c in self.input_mesh.connectivity.vertex_to_corners(v))

    def ref_vertex(self, v_cut: int) -> int:
        """Given the index `v_cut` of a vertex in the cut mesh, returns the index of the corresponding vertex in the original mesh.

        Args:
            v_cut (int): index of a vertex in the cut mesh. If the index is invalid, returns None

        Returns:
            int: index of the reference vertex in the original mesh
        """
        if v_cut>len(self.cut_mesh.vertices): return None
        return self._ref_vertex.get(v_cut, v_cut)

    @property
    def cut_graph(self) -> PolyLine:
        """The graph formed by all cut edges as a Polyline object

        Returns:
            PolyLine: the cut graph
        """
        if self._cut_graph is None:
            self._build_cut_graph()
        return self._cut_graph

    def _build_cut_graph(self):
        # Build tree of potential cuts for debug purposes
        self._cut_graph = RawMeshData()
        new_v_id = dict()
        vid = 0
        for ie in self.cut_edges:
            v1,v2 = self.input_mesh.edges[ie]
            for v in (v1,v2):
                if v not in new_v_id:
                    new_v_id[v] = vid
                    vid += 1
                    self._cut_graph.vertices.append(self.input_mesh.vertices[v])
            e = keyify(new_v_id[v1], new_v_id[v2])
            self._cut_graph.edges.append(e)
        self._cut_graph = PolyLine(self._cut_graph)


class SingularityCutter(SurfaceMeshCutter):

    class Strategy(Enum):
        AUTO = 0
        SIMPLE = 1
        SHORTEST_PATHS=2
        SHORTEST_LIMITED=3
        FEATURES=4

        @classmethod
        def from_string(cls, txt : str):
            key = txt.lower()
            if "short" in key:
                return cls.SHORTEST_PATHS
            if "limited" in key:
                return cls.SHORTEST_LIMITED
            if "feat" in key:
                return cls.FEATURES
            if "simple" in key:
                return cls.SIMPLE
            return cls.AUTO
        
        def to_string(self) -> str:
            return {
                SingularityCutter.Strategy.AUTO : "auto",
                SingularityCutter.Strategy.SIMPLE : "simple",
                SingularityCutter.Strategy.SHORTEST_PATHS : "approx. shortest cuts",
                SingularityCutter.Strategy.FEATURES : "follow features",
                SingularityCutter.Strategy.SHORTEST_LIMITED : "limited shortest cuts"
            }[self]

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
            mesh : SurfaceMesh,
            singularities : list,
            strategy : str = "auto",
            verbose = False,
            **kwargs):
        """
        Args:
            mesh (SurfaceMesh): input mesh
            singularities (list): indices of the singular vertices
            strategy (str): which strategy to use. Choices are ["auto", "simple", "short", "feat", "limited"]
            verbose (bool, optional): verbose mode. Defaults to False.

        Keyword Args:
            features (FeatureEdgeDetector, optional): feature edge data structure. If provided, the cuts will follow the feature as much as possible. Defaults to None.
            debug (bool): debug mode. Computes additionnal outputs as mesh attributes. Defaults to False

        Attributes:
            cut_edges (set): indices of edges that were cut
            cut_mesh (SurfaceMesh): a copy of the mesh where specified edges have been cut
        """
        super().__init__(mesh, verbose, name="SingularityCutter")
        self._strategy = SingularityCutter.Strategy.from_string(strategy)            
        self._debug : bool = kwargs.get("debug", False)

        if isinstance(singularities, list):
            self.singularities = singularities
            self.singu_set = set(singularities)
        else:
            self.singularities : list = [_x for _x in singularities]
            self.singu_set = set(singularities)

        self._feat_detector : "FeatureEdgeDetector" = kwargs.get("features", None)
        
        if self._strategy == SingularityCutter.Strategy.FEATURES and self._feat_detector is None:
            self.warn("Please provide a FeatureEdgeDetector object to run the 'feature' cutting stragegy. Changing strategy.")
            self._strategy = SingularityCutter.Strategy.AUTO
        if self._strategy == SingularityCutter.Strategy.AUTO:
            self._strategy = self._choose_auto_strategy()
        self.log("Strategy:", self._strategy.to_string())

    def cut(self):
        self.run()
    
    def _choose_auto_strategy(self):
        if (self._feat_detector is not None 
            and not self._feat_detector.only_border 
            and len(self._feat_detector.feature_edges)>len(self.input_mesh.boundary_edges)):
            # If we have featur edges, we run the feature following heuristic
            return SingularityCutter.Strategy.FEATURES
        
        n_singus = len(self.singularities)
        if n_singus <= 10:
            # When the number of singularities is small, we can afford to compute all n(n-1)/2 shortest paths
            return SingularityCutter.Strategy.SHORTEST_PATHS
        
        if len(self.singularities)>100:
            # If the number of singularities is too large, we do not use a heuristic for performance purposes
            return SingularityCutter.Strategy.SIMPLE
        
        # otherwise, the limited heuristic is a good compromise
        return SingularityCutter.Strategy.SHORTEST_LIMITED

    def run(self):
        """
        Runs the cutting process
        """
        self.log(f"Cutting to link {len(self.singularities)} singularities and retrieve disk topology")
        
        ### Run heuristic cuts depending on the strategy
        if self._strategy == SingularityCutter.Strategy.SIMPLE:
            edge_flag = Attribute(bool) # False by default for all edges
            regions = None # no features
        
        elif self._strategy in (SingularityCutter.Strategy.SHORTEST_PATHS, SingularityCutter.Strategy.SHORTEST_LIMITED):
            self.log("Run heuristic spanning tree to flag candidate seams")
            limited = self._strategy == SingularityCutter.Strategy.SHORTEST_LIMITED
            edge_flag = self._heuristic_flag_edges_shortest_path(limited)
            regions = None # no features 
    
        elif self._strategy == SingularityCutter.Strategy.FEATURES:
            self.log(f"{len(self._feat_detector.feature_edges)} feature edges and {len(self._feat_detector.feature_vertices)} feature vertices provided")
            self.log("Run heuristic spanning tree to flag candidate seams")
            edge_flag = self._heuristic_flag_edges_features()
            self.log("Building feature region connectivity")
            regions = self._build_feature_regions(edge_flag) # face regions delimited by feature edges
            
        else:
            self.warn("Cutting strategy not recognized. Should not happen.")
            raise Exception("Aborting")


        self.log("BFS on dual graph")
        visited_edges = self._build_dual_tree(edge_flag, regions)

        self.log("Pruning non-traversed edges to extract cuts")
        seam_edges = self._prune_edge_tree(visited_edges)
        
        self.log("Building cut mesh")
        super().run(seam_edges)
 
    def _heuristic_flag_edges_shortest_path(self, limited) -> Attribute:
        # With the shortest path strategy, we first compute all shortest paths between pairs of singularities and restrain the cut graph to be a spanning tree on this path graph

        mesh_has_border = len(self.input_mesh.boundary_vertices)>0
        if self.input_mesh.edges.has_attribute("length"):
            edge_lengths = self.input_mesh.edges.get_attribute("length")
        else:
            edge_lengths = attributes.edge_length(self.input_mesh)

        BORDER = -1 # the index of border (>0 are vertices)
        singul = self.singularities + [BORDER] if mesh_has_border else self.singularities

        # edge selected for spanning tree
        edge_flags = self.input_mesh.edges.create_attribute("singularity_tree", bool) if self._debug else Attribute(bool) 
        if not self.singularities:
            # no singularities => no spanning tree and no constraints on edges
            return edge_flags
    
        # Compute all edge paths between singularities (and border)
        path_btw_singus = dict()
        for i,a in enumerate(self.singularities):
            if limited:
                paths_a = closest_n_vertices(self.input_mesh, a, 5, set(self.singularities),weights=edge_lengths)
            else:
                paths_a = shortest_path(self.input_mesh, a, set(self.singularities[i:]), weights=edge_lengths)
            for b in paths_a:
                e = keyify(a,b)
                if e in path_btw_singus: continue
                path_btw_singus[e] = paths_a[b]
            if mesh_has_border and not limited:
                path_btw_singus[(BORDER,a)] = shortest_path_to_border(self.input_mesh, a, weights=edge_lengths)

        # Compute path lengths utility function
        compute_path_length = lambda path : sum([edge_lengths[self.input_mesh.connectivity.edge_id(A,B)] for A,B in consecutive_pairs(path)])

        path_lengths = []
        for k in path_btw_singus:
            path_lengths.append((compute_path_length(path_btw_singus[k]), k))
        path_lengths.sort()

        # Build Minimum Spanning Tree using Kruskal's algorithm
        uf = UnionFind(singul)
        selected = []
        for (_,key) in path_lengths: # first elem in tuple is length (not useful here)
            a,b = key
            if not uf.connected(a,b):
                selected.append(key)
                uf.union(a,b)

        # Flag edges
        for (a,b) in selected:
            path_ab = path_btw_singus[(a,b)]
            for u,v in consecutive_pairs(path_ab):
                edge_flags[ self.input_mesh.connectivity.edge_id(u,v) ] = True
        return edge_flags

    def _heuristic_flag_edges_features(self) -> Attribute:
        # With the feature strategy we want the spanning tree to have the maximal possible intersection with the feature graph. This leads to prettier results

        edge_lengths = attributes.edge_length(self.input_mesh, persistent=False)
        # edge selected for spanning tree
        edge_flags = self.input_mesh.edges.create_attribute("singularity_tree", bool) if self._debug else Attribute(bool) 
        if len(self.singularities)==0:
            # no singularities => no spanning tree and no constraints on edges
            return edge_flags

        # First compute the closest point from singularities to feature graph
        # boundary vertices are included as feature vertices -> boundary is handled automatically
        closest_v = set()
        for v in self.singularities:
            id_feat, path = shortest_path_to_vertex_set(self.input_mesh, v, self._feat_detector.feature_vertices, weights=edge_lengths)
            closest_v.add(id_feat)
            for i in range(len(path)-1):
                u,v = path[i], path[i+1]
                e = self.input_mesh.connectivity.edge_id(u,v)
                edge_flags[e] = True

        # Then construct a tree on the feature graph linking all the previous points. Perform a BFS
        queue = deque([(v,None) for v in closest_v])
        visited = dict([(v, False) for v in self._feat_detector.feature_vertices])
        while len(queue)>0:
            v,prev = queue.popleft()
            if visited[v] : continue
            visited[v] = True
            if prev is not None:
                e = self.input_mesh.connectivity.edge_id(v,prev)
                edge_flags[e] = True
            for e in self.input_mesh.connectivity.vertex_to_edges(v):
                if e in self._feat_detector.feature_edges:
                    nv = self.input_mesh.connectivity.other_edge_end(e,v)
                    if not visited[nv]:
                        queue.append((nv,v))
        return edge_flags

    def _build_feature_regions(self, forbidden_edges : Attribute) -> UnionFind:
        ### Builds a UnionFind data structure over faces. Two faces are connected if they can be linked by dual edges that do not cross the feature graph.
        regions = UnionFind(self.input_mesh.id_faces)
        not_traversible = { _e for _e in forbidden_edges} | self._feat_detector.feature_edges
        tree = FaceSpanningForest(self.input_mesh, not_traversible)()
        for vertex,father in tree.traverse():
            if father is not None:
                regions.union(vertex,father)
        if self._debug:
            TriFlagAttr = self.input_mesh.faces.create_attribute("face_regions", int)
            for f in self.input_mesh.id_faces:
                TriFlagAttr[f] = regions.find(f)
        return regions

    def _build_dual_tree(self, forbidden_edges:Attribute = None, regions : UnionFind = None):
        ### Builds a spanning tree over dual edges, so that each face is visited
        # If forbidden_edges is provided, will avoid crossing those edges.
        # If regions is provided, will make sure that two adjacent regions are connected by only one edge to avoid creating additionnal seams
    
        # Dijsktra on faces (dual edges)
        fvisited = ArrayAttribute(bool, len(self.input_mesh.faces)) #self.input_mesh.faces.create_attribute("cut_visited", bool)
        path = [None for _ in self.input_mesh.id_faces]
        dist = [float("inf") for _ in self.input_mesh.id_faces]
        queue : PriorityQueue = PriorityQueue() # queue contains indexes of faces
        queue.push(0,0)
        dist[0] = 0

        barycenters = attributes.face_barycenter(self.input_mesh, persistent=False)

        def face_distance(f1,f2):
            # Heuristic for distance between two faces
            return distance(barycenters[f1], barycenters[f2])

        while not queue.empty():
            iF = queue.get().x
            if fvisited[iF] : continue
            fvisited[iF] = True                
            for e in self.input_mesh.connectivity.face_to_edges(iF):
                v1,v2 = self.input_mesh.edges[e]
                if forbidden_edges[e] : continue # edge is on the singularity spanning tree
                iF2 = self.input_mesh.connectivity.opposite_face(v1,v2,iF)
                if iF2 is None: continue
                blocked = (regions is not None) and (e in self._feat_detector.feature_edges and regions.connected(iF,iF2))
                # if regions is provided, will check if this region is already reached by another branch of the tree. It that is the case, stop here.
                if blocked: continue
                if regions is not None: regions.union(iF,iF2)
                d = face_distance(iF,iF2)
                if dist[iF2] > dist[iF] + d :
                    dist[iF2] = dist[iF] + d
                    path[iF2] = e
                if not fvisited[iF2] :
                    queue.push(iF2, dist[iF2])
        return {path[f] for f in self.input_mesh.id_faces if path[f] is not None }

    def _prune_edge_tree(self, visited_edges):
        # seams are a subset of edges that were **not** used during the dual search
        remaining_edges = set(self.input_mesh.id_edges) - visited_edges
        
        # build connectivity on the edge tree
        cut_adj = dict([(i,set()) for i in self.input_mesh.id_vertices])
        for e in remaining_edges:
            a,b = self.input_mesh.edges[e]
            cut_adj[a].add(b)
            cut_adj[b].add(a)

        # remove leaves of cut tree until we fall on singularities
        queue = deque()
        for i in self.input_mesh.id_vertices:
            # append all tree leaves on the queue
            d = len(cut_adj[i])
            if d==1 and i not in self.singularities:
                queue.append(i)
        while len(queue)>0:
            A = queue.popleft()
            for B in cut_adj[A]:
                cut_adj[B].remove(A)
                remaining_edges.remove(self.input_mesh.connectivity.edge_id(A,B))
                if len(cut_adj[B])==1 and B not in self.singularities:
                    queue.append(B)
            cut_adj[A] = set() # A is disconnected
        self.log("# Edges cut:", len(remaining_edges) - len(self.input_mesh.boundary_edges))
        return remaining_edges

    def _build_cut_graph_as_mesh(self):
        # Build tree of potential cuts for debug purposes
        self._cut_graph = RawMeshData()

        new_v_id = dict()
        vid = 0
        for ie in self.cut_edges:
            v1,v2 = self.input_mesh.edges[ie]
            for v in (v1,v2):
                if v not in new_v_id:
                    new_v_id[v] = vid
                    vid += 1
                    self._cut_graph.vertices.append(self.input_mesh.vertices[v])
            e = keyify(new_v_id[v1], new_v_id[v2])
            self._cut_graph.edges.append(e)

        singuls_attr = self._cut_graph.vertices.create_attribute("selection", bool)
        for x in self.singularities:
            singuls_attr[new_v_id[x]] = True
        self._cut_graph = PolyLine(self._cut_graph)