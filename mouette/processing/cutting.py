from .worker import Worker

from ..geometry import distance
from ..mesh.datatypes import *
from ..mesh.mesh_data import RawMeshData
from ..mesh.mesh_attributes import *
from ..utils import keyify, UnionFind, PriorityQueue
from .paths import shortest_path, shortest_path_to_border, shortest_path_to_vertex_set
from .trees import FaceSpanningForest
from .. import attributes
from collections import deque

class SingularityCutter(Worker):
    """
    Given some indexes in the mesh, performs optimal cuts connecting all these vertices with the boundary.
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
            mesh : SurfaceMesh,
            singularities : list,
            features : "FeatureEdgeDetector" = None,
            verbose = False):
        super().__init__("SingularityCutter", verbose)
        self.input_mesh : SurfaceMesh = mesh
        
        if isinstance(singularities, list):
            self.singularities = singularities
            self.singu_set = set(singularities)
        else:
            self.singularities : list = [_x for _x in singularities]
            self.singu_set = set(singularities)

        self.edge_lengths = attributes.edge_length(self.input_mesh, persistent=False)

        self.feat_detector : "FeatureEdgeDetector" = features
        self._has_features : bool = None

        self.cut_edges : set = None # ids of edges of input mesh that were cut 
        self.cut_adj : dict = None # vertex -> set of adjacent vertices in the cut graph

        # /!\ global ordering of vertices/edges is likely to change. Only triangles remain the same
        self.ref_vertex : dict = None # split vertex -> original vertex

        self._cut_graph : PolyLine = None
        self._output_mesh : SurfaceMesh = None
    
    @property
    def has_features(self):
        if self._has_features is None:
            self._has_features = False
            if self.feat_detector is not None:
                for e in self.feat_detector.feature_edges:
                    if not self.input_mesh.is_edge_on_border(*self.input_mesh.edges[e]):
                        self._has_features = True
                        break
        return self._has_features

    @property
    def output_mesh(self) -> SurfaceMesh:
        if self._output_mesh is None:
            self._build_mesh_with_cuts()
        return self._output_mesh
    
    @property
    def cut_graph(self) -> PolyLine:
        if self._cut_graph is None:
            self._build_cut_graph_as_mesh()
        return self._cut_graph

    def run(self):
        self.log("Cutting to link singularities and retrieve disk topology")
        self.log("# Singularities :", len(self.singularities))
        if self.has_features:
            self.log(f"{len(self.feat_detector.feature_edges)} feature edges and {len(self.feat_detector.feature_vertices)} provided")
            self._run_with_features()
        else:
            self._run_no_features()

    def _run_with_features(self):
        self.log("Step 1 : Minimal Spanning Tree to link singularities and border")
        edge_flag = self._build_singularity_spanning_tree_with_features()

        self.log("Step 2 : BFS on dual graph + BFS on non-traversed edges to extract cuts and homology")
        evisited = self._build_dual_tree_with_features(edge_flag)
        self._build_cut_edges_tree(evisited) # builds attributes self.cut_edges and self.cut_adj

        self.log("Step 3 : pruning homology tree")
        self._prune_edge_tree()
        self.log("Cutting done")

    def _run_no_features(self):
        
        self.log("Step 1 : Minimal Spanning Tree to link singularities and border")
        edge_flag = self._build_singularity_spanning_tree_no_features()
    
        self.log("Step 2 : BFS on dual graph + BFS on non-traversed edges to extract cuts and homology")
        evisited = self._build_dual_tree_no_features(edge_flag)
        self._build_cut_edges_tree(evisited) # builds attributes self.cut_edges and self.cut_adj

        self.log("Step 3 : pruning homology tree")
        self._prune_edge_tree()
        self.log("Cutting done")

    def _build_singularity_spanning_tree_no_features(self):
        """
        Returns:
            Attribute: the spanning tree given as a boolean attribute on edges
        """
        mesh_has_border = len(self.input_mesh.boundary_vertices)>0
        BORDER = -1 # the index of border (>0 are vertices)
        singul = self.singularities + [BORDER] if mesh_has_border else self.singularities

        #edge_flags = self.input_mesh.edges.create_attribute("singularity_tree", bool) 
        edge_flags = Attribute(bool) # edge selected for spanning tree
        
        if not self.singularities:
            # no singularities => no spanning tree and no constraints on edges
            return edge_flags

        def path_length(path):
            l = 0
            for i in range(1, len(path)):
                A,B = path[i-1], path[i]
                l += self.edge_lengths[self.input_mesh.connectivity.edge_id(A,B)]
            return l
            
        # Compute all edge paths between singularities (and border)
        path_btw_singus = dict()
        for i,a in enumerate(self.singularities):
            paths_a = shortest_path(self.input_mesh, a, set(self.singularities[i:]), weights=self.edge_lengths)
            for b in paths_a:
                path_btw_singus[keyify(a,b)] = paths_a[b]
            if mesh_has_border:
                path_btw_singus[(BORDER,a)] = shortest_path_to_border(self.input_mesh, a, weights=self.edge_lengths)
                
        # Compute path lengths
        path_lengths = []
        for k in path_btw_singus:
            path_lengths.append((path_length(path_btw_singus[k]), k))
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
            for i in range(1, len(path_ab)):
                u,v = path_ab[i-1], path_ab[i]
                edge_flags[ self.input_mesh.connectivity.edge_id(u,v) ] = True
        return edge_flags

    def _build_singularity_spanning_tree_with_features(self):
        """We do not apply the same algorithm as we want the spanning tree to have the maximal possible intersection with the feature graph. This leads to prettier results.

        Returns:
            Attribute: the spanning tree given as a boolean attribute on edges
        """
        edge_flags = self.input_mesh.edges.create_attribute("singularity_tree", bool) # edge selected for spanning tree
        if len(self.singularities)==0:
            # no singularities => no spanning tree and no constraints on edges
            return edge_flags


        # First compute the closest point from singularities to feature graph
        closest_v = set()
        for v in self.singularities:
            id_feat, path = shortest_path_to_vertex_set(self.input_mesh, v, self.feat_detector.feature_vertices, weights=self.edge_lengths)
            closest_v.add(id_feat)
            for i in range(len(path)-1):
                u,v = path[i], path[i+1]
                e = self.input_mesh.connectivity.edge_id(u,v)
                edge_flags[e] = True

        # Then construct a tree on the feature graph linking all the previous points. Perform a BFS
        closest_v = list(closest_v)
        queue = deque()
        visited = dict([(v, False) for v in self.feat_detector.feature_vertices])
        parent = dict([(v, None) for v in self.feat_detector.feature_vertices])
        for v in closest_v:
            queue.append((v,None))
        while len(queue)>0:
            v,prev = queue.popleft()
            if visited[v] : continue
            visited[v] = True
            parent[v] = prev
            if prev is not None:
                e = self.input_mesh.connectivity.edge_id(v,prev)
                edge_flags[e] = True
            for e in self.input_mesh.connectivity.vertex_to_edges(v):
                if e in self.feat_detector.feature_edges:
                    nv = self.input_mesh.connectivity.other_edge_end(e,v)
                    if not visited[nv]:
                        queue.append((nv,v))

        return edge_flags

    def _build_feature_regions(self, forbidden_edges):
        if not self.has_features: return None
        regions = UnionFind(self.input_mesh.id_faces)
        not_traversible = { _e for _e in forbidden_edges} | self.feat_detector.feature_edges
        tree = FaceSpanningForest(self.input_mesh, not_traversible)()
        for vertex,father in tree.traverse():
            if father is not None:
                regions.union(vertex,father)

        TriFlagAttr = ArrayAttribute(int, len(self.input_mesh.faces)) # self.input_mesh.faces.create_attribute("face_regions", int)
        for f in self.input_mesh.id_faces:
            TriFlagAttr[f] = regions.find(f)
        return regions

    def _build_dual_tree_no_features(self, forbidden_edges:Attribute):
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
                if iF2 is not None: 
                    d = face_distance(iF,iF2)
                    if dist[iF2] > dist[iF] + d :
                        dist[iF2] = dist[iF] + d
                        path[iF2] = e
                    if not fvisited[iF2] :
                        queue.push(iF2, dist[iF2])
        return {path[f] for f in self.input_mesh.id_faces if path[f] is not None }
    
    def _build_dual_tree_with_features(self, forbidden_edges:Attribute):
        regions = self._build_feature_regions(forbidden_edges) # face regions delimited by feature edges

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
                if iF2 is not None:
                    u,v = set(self.input_mesh.faces[iF]) & set(self.input_mesh.faces[iF2])
                    blocked = self.input_mesh.connectivity.edge_id(u,v) in self.feat_detector.feature_edges and regions.connected(iF,iF2)
                    if not blocked:
                        d = face_distance(iF,iF2)
                        if dist[iF2] > dist[iF] + d :
                            dist[iF2] = dist[iF] + d
                            path[iF2] = e
                        if not fvisited[iF2]:
                            queue.push(iF2, dist[iF2])
                            regions.union(iF,iF2)
        return {path[f] for f in self.input_mesh.id_faces if path[f] is not None }

    def _build_cut_edges_tree(self, evisited):
        # Build self._cut_edges with edges that were **not** used during the above search
        self.cut_edges = set(self.input_mesh.id_edges) - evisited
        # build connectivity on the edge tree
        self.cut_adj = dict([(i,set()) for i in self.input_mesh.id_vertices]) # connectivity lists of cut tree
        for e in self.cut_edges:
            a,b = self.input_mesh.edges[e]
            self.cut_adj[a].add(b)
            self.cut_adj[b].add(a)

    def _prune_edge_tree(self):
        # remove leaves of cut tree until we fall on singularities
        queue = deque()
        for i in self.input_mesh.id_vertices:
            d = len(self.cut_adj[i])
            if d==1 and i not in self.singularities:
                queue.append(i)
        while len(queue)>0:
            A = queue.popleft()
            for B in self.cut_adj[A]:
                self.cut_adj[B].remove(A)
                self.cut_edges.remove(self.input_mesh.connectivity.edge_id(A,B))
                if len(self.cut_adj[B])==1 and B not in self.singularities:
                    queue.append(B)
            self.cut_adj[A] = set() # A is disconnected
        self.log("# Edges cut:", len(self.cut_edges) - len(self.input_mesh.boundary_edges))

    def _build_cut_graph_as_mesh(self):
        # Build tree of potential cuts for debug purposes
        self._cut_graph = RawMeshData()
        hard_edges = self._cut_graph.edges.create_attribute("hard_edges", bool)

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
            hard_edges[len(hard_edges)] = True
            self._cut_graph.edges.append(e)

        singuls_attr = self._cut_graph.vertices.create_attribute("selection", bool)
        for x in self.singularities:
            singuls_attr[new_v_id[x]] = True
        self._cut_graph = PolyLine(self._cut_graph)

    def _build_mesh_with_cuts(self):
        self._output_mesh = RawMeshData()
        uf = UnionFind(range(3*len(self.input_mesh.faces)))
        duplicate_vertices = dict([(v, set()) for v in self.input_mesh.id_vertices])

        # At first, no faces are adjacent
        kF = 0
        for iF,F in enumerate(self.input_mesh.faces):
            nF = len(F)
            self._output_mesh.faces.append([kF+_i for _i in range(nF)])
            for iv,v in enumerate(F):
                pv = self.input_mesh.vertices[v]
                self._output_mesh.vertices.append(pv)
                duplicate_vertices[v].add(kF+iv)
            kF += nF

        # Fusion vertices along edges that are not a cut
        for e in self.input_mesh.interior_edges:
            a,b = self.input_mesh.edges[e]
            if e not in self.cut_edges:
                F1, iA1, iB1 = self.input_mesh.connectivity.direct_face(a,b,True)
                F2, iB2, iA2 = self.input_mesh.connectivity.direct_face(b,a,True)
                uf.union(self._output_mesh.faces[F1][iA1], self._output_mesh.faces[F2][iA2])
                uf.union(self._output_mesh.faces[F1][iB1], self._output_mesh.faces[F2][iB2])

        for i,F in enumerate(self._output_mesh.faces):
            self._output_mesh.faces[i] = [uf.find(v) for v in F]
            
        # Reorder vertices to [0,n-1] and get rid of non connected duplicates
        imap = dict()
        i=0
        for F in self._output_mesh.faces:
            for v in F:
                if v in imap: continue
                imap[v]=i # assign new index to first version of vertex u met
                i+=1

        for iF,F in enumerate(self._output_mesh.faces):
            self._output_mesh.faces[iF] = [imap[v] for v in F]
        order_verts = [None]*len(imap)

        for u in range(len(self._output_mesh.vertices)):
            if u not in imap: continue
            order_verts[imap[u]] = self._output_mesh.vertices[u]
        self._output_mesh.vertices.clear()
        self._output_mesh.vertices += order_verts
        self._output_mesh = SurfaceMesh(self._output_mesh)
        
        # get rid of duplicated and isolated vertices in map
        for v in duplicate_vertices:
            duplicate_vertices[v] = {imap[uf.find(u)] for u in duplicate_vertices[v]}

        # inverse duplicate_vertices to get self.ref_vertex
        self.ref_vertex = dict()
        for v in duplicate_vertices:
            for u in duplicate_vertices[v]:
                self.ref_vertex[u] = v

