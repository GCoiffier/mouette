from .worker import Worker

from ..mesh.mesh_attributes import Attribute
from ..mesh.datatypes import *
from ..mesh.mesh_data import RawMeshData
from .. import geometry
from ..attributes.misc_faces import face_normals
from ..attributes.misc_corners import corner_angles
from math import pi

class FeatureEdgeDetector(Worker):
    """
    Worker used to detect features on a surface mesh. Feature edges are of three types:
    
    - Marked edges from the input file (reading the "hard_edges" attribute on edges)
    
    - Boundary edges
    
    - Edges such that their dihedral angle is large (crease edges)

    The `FeatureEdgeDetector` object ican be given as a parameter in parametrization or frame field algorithms that need feature edge alignment.
    """

    def __init__(self, only_border : bool = False, flag_corners=True, corner_order:int = 4, compute_feature_graph=True, verbose=True):
        """
        Parameters:
            only_border (bool, optional): If set to True, will only consider border edges as features. Defaults to False.
            flag_corners (bool, optional): If set to True, will also compute a goal angle defect (multiple of pi/2) of each detected vertices. Defaults to True.
            corner_order (int, optional): For corner detection, considers corners of angle defect 2pi/corner_order. Defaults to 4 (corners of pi/2).
            compute_feature_graph (bool, optional): whether to compute a Polyline object representing the feature graph. For debug and visualization purposes. Defaults to True.
            verbose (bool, optional): Verbose mode. Defaults to True.
        
        Attributes:
            feature_vertices (set): indices of the vertices that are adjacent to at least one feature edge
            feature_edges (set): indices of feature edges
            feature_degree (Attribute): number of feature edges each vertex is adjacent to
            local_feat_edges (dict): features edges in the neighborhood of each vertex (in the order of `mesh.connectivity.vertex_to_edges`)
            corners (Attribute): the order of each detected vertex corners
        """
        self._feature_graph : PolyLine = None
        self.only_border: bool = only_border # whether to ignore every feature edge that is not a boundary
        self.flag_corners: bool = flag_corners # whether to also flag the angle value of each feature vertex
        self.corner_order: int = corner_order
        self.compute_feature_graph: bool = compute_feature_graph
        
        self.border_cycles : list = None # result of 'M.processing.extract_border_cycle_all'

        self.fnormals : Attribute = None # face normals

        # Following data structures will be filled
        self.feature_vertices : set = None # indexes of feature vertices
        self.feature_edges : set = None # indexes of feature edges
        self.feature_degrees : Attribute = None # degree of each feature vertex in the feature graph
        self.local_feat_edges : dict = None # vertex -> list of local indices of edges that are feature edges
        self.corners : Attribute = None # index of each detected corners (as an int k such that defect is close to k*pi/2)

        super().__init__("FeatureDetector", verbose)

    def clear(self):
        """Clears the data structures"""
        self.feature_vertices = set()
        self.feature_edges = set()
        self.feature_degrees = Attribute(int)
        self.local_feat_edges = dict() # vertex -> list of local indices of edges that are feature edges



##### Feature Graph #####

    @property
    def feature_graph(self) -> PolyLine:
        """the feature edges as a polyline object"""
        if self._feature_graph is None:
            self._compute_feature_graph()
        return self._feature_graph

    def _compute_feature_graph(self, mesh: SurfaceMesh):
        """Computes self._feature_mesh"""
        self._feature_graph = RawMeshData()
        fvert = list(self.feature_vertices)
        fvert = dict([(fvert[i],i) for i in range(len(self.feature_vertices))])
        degree_attr = self._feature_graph.vertices.create_attribute("degree", int)
        self._feature_graph.vertices += [mesh.vertices[k] for k in fvert.keys()]
        for v in self.feature_vertices:
            degree_attr[fvert[v]] = self.feature_degrees[v]
        for e in self.feature_edges:
            A, B = mesh.edges[e]
            self._feature_graph.edges.append((fvert[A],fvert[B]))
        self._feature_graph = PolyLine(self._feature_graph)
 
##### Feature detection subfunctions #####

    def _add_border_to_features(self, mesh : SurfaceMesh, feature_attr : Attribute) -> Attribute:
        if len(mesh.boundary_edges)==0 : return feature_attr # no border
        for e in mesh.boundary_edges:
            feature_attr[e] = True
        return feature_attr

    def _add_hard_edges_to_features(self, mesh : SurfaceMesh, feature_attr : Attribute) -> Attribute:
        """The mesh may already define a set of "hard" edges, for instance if it was imported from a file where edges were specified.
        These edges should be registered as features, but we filter them if their curvature is too small.

        Parameters:
            feature_attr (Attribute): the feature flag to fill in

        Returns:
            Attribute: the modified feature flag
        """
        if self.only_border : return feature_attr
        DOT_THRESHOLD = 0.2
        if mesh.edges.has_attribute("hard_edges"):
            for e in mesh.edges.get_attribute("hard_edges"):
                A,B = mesh.edges[e]
                T1,T2 = mesh.connectivity.edge_to_faces(A,B)
                if T1 is None or T2 is None : continue
                N1,N2 = self.fnormals[T1], self.fnormals[T2]
                # we filter hard edge also depending on their angles. if marked but flat, the edge is ignored
                if geometry.dot(N1,N2) < 1 - DOT_THRESHOLD and not mesh.is_edge_on_border(*mesh.edges[e]):
                    feature_attr[e] = True
        return feature_attr

    def _add_sharp_angles_to_features(self, mesh : SurfaceMesh, feature_attr : Attribute) -> Attribute:
        """Flags as features all edges which dihedral angle is smaller than pi/2 or greater than 3*pi/2

        Parameters:
            feature_attr (Attribute): the feature flag to fill in

        Returns:
            Attribute: the modified feature flag
        """
        if self.only_border : return feature_attr
        DOT_THRESHOLD = 0.5
        for e, (A,B) in enumerate(mesh.edges):
            T1,T2 = mesh.connectivity.edge_to_faces(A,B)
            if T1 is None or T2 is None : continue
            N1,N2 = self.fnormals[T1], self.fnormals[T2]
            if geometry.dot(N1,N2) < DOT_THRESHOLD:
                feature_attr[e] = True
        return feature_attr
    
    def _flag_corners(self, mesh : SurfaceMesh):
        if mesh.vertices.has_attribute("corners"):
            self.corners = mesh.vertices.get_attribute("corners")
        else:
            self.corners = mesh.vertices.create_attribute("corners", int)
        angles = corner_angles(mesh, persistent=False)
        for v in self.feature_vertices:
            angle_v = 0.
            for T in mesh.connectivity.vertex_to_faces(v):
                c = mesh.connectivity.vertex_to_corner_in_face(v,T)
                angle_v += angles[c]
            if abs(angle_v) <  2*pi/self.corner_order:
                self.corners[v] = 1 if angle_v>=0 else -1
            else:
                self.corners[v] = round(angle_v * self.corner_order / ( 2 * pi))

##### Main detect function #####

    @allowed_mesh_types(SurfaceMesh)
    def run(self, mesh : SurfaceMesh):
        self.clear()
        self.log("Initializing attributes")
        if mesh.faces.has_attribute("normals"):
            self.fnormals = mesh.faces.get_attribute("normals")
        else:
            self.fnormals = face_normals(mesh, persistent=False)

        # create the feature attribute
        if mesh.vertices.has_attribute("feature"):
            feat_v = mesh.vertices.get_attribute("feature")
            feat_v.clear()
        else:
            feat_v = mesh.vertices.create_attribute("feature", bool)
        
        if mesh.edges.has_attribute("feature"):
            feature = mesh.edges.get_attribute("feature")
            feature.clear()
        else:
            feature = mesh.edges.create_attribute("feature", bool)

        self.log("Detect Features")
        feature = self._add_hard_edges_to_features(mesh, feature)
        feature = self._add_sharp_angles_to_features(mesh, feature)
        feature = self._add_border_to_features(mesh, feature)

        self.log("Build set containers")
        for e in feature:
            A,B = mesh.edges[e]
            self.feature_edges.add(e)
            self.feature_vertices.add(A)
            self.feature_vertices.add(B)
        
        self.log("Build local feature edges indices")
        for v in self.feature_vertices:
            self.local_feat_edges[v] = []
            for i, ev in enumerate(mesh.connectivity.vertex_to_edges(v)):
                if feature[ev] :
                    self.local_feat_edges[v].append(i)

        for e in self.feature_edges:
            A,B = mesh.edges[e]
            self.feature_degrees[A] += 1
            self.feature_degrees[B] += 1

        if self.flag_corners:
            self.log("Flag corners")
            self._flag_corners(mesh)

        if self.compute_feature_graph:
            self.log("Compute feature graph")
            self._compute_feature_graph(mesh)

        self.log(f" # Feature edges: {len(self.feature_edges)}")

        for v in self.feature_vertices: feat_v[v] = True
        self.log(f" # Feature vertices: {len(self.feature_vertices)}")


    def detect(self, mesh : SurfaceMesh):
        """Runs the detection on a provided mesh.

        Args:
            mesh (SurfaceMesh): the input mesh
        """
        self.run(mesh)