from ..mesh.datatypes import *
from ..mesh.mesh_attributes import ArrayAttribute
from .. import attributes
from .features import FeatureEdgeDetector
from ..geometry import Vec
from .. import geometry as geom
from .. import utils

import numpy as np
import math
from abc import ABC, abstractmethod

class SurfaceConnection(ABC):
    """
    Abstract base class for surface connections.
    """
    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, feat : FeatureEdgeDetector = None ):
        self.mesh = mesh
        self.feat = feat or FeatureEdgeDetector(only_border=True, compute_feature_graph=False, verbose=False)(self.mesh)
        self._baseX : np.ndarray = None
        self._baseY : np.ndarray = None
        self._transport : dict = None
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        pass

    def transport(self, iA : int, iB : int) -> float:
        return self._transport[(iA,iB)]

    def base(self, i):
        return self._baseX[i], self._baseY[i]

class SurfaceConnectionVertices(SurfaceConnection):
    """
    Local bases and parallel transport defined between tangent planes of the mesh at the vertices

    Reference:
        Globally Optimal Direction Fields by Knöppel et al. (2013)
    
    Args:
        mesh (SurfaceMesh): the supporting mesh
        feat (FeatureEdgeDetector, optional): feature edges of the mesh. If not provided, will be initialized with a FeatureEdgeDetector object that flags only the boundary. Defaults to None.
    Additionnal args:
        vnormals (Attribute): An attribute representing the vertex normals. If not provided, will be computed at initialization
        angles (Attribute) : An attribute representing angles at corners of faces. If not provided, will be computed at initialization

    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh,  feat : FeatureEdgeDetector = None, **kwargs):
        self.vnormals = kwargs.get("vnormals", None)
        
        if self.vnormals is None: 
            if mesh.vertices.has_attribute("normals"):
                self.vnormals = mesh.vertices.get_attribute("normals")
            else :
                self.vnormals = attributes.vertex_normals(mesh)
        
        self.angles = kwargs.get("angles", None)
        if self.angles is None:
            if mesh.face_corners.has_attribute("angles"):
                self.angles = mesh.face_corners.get_attribute("angles")
            else:
                self.angles = attributes.corner_angles(mesh)
        
        self.total_angle = ArrayAttribute(float, len(mesh.vertices))
        for iC, C in enumerate(mesh.face_corners):
            self.total_angle[C] = self.total_angle[C] + self.angles[iC] 
        super().__init__(mesh, feat)

    def _initialize(self):
        n_vert = len(self.mesh.vertices)
        self._baseX = ArrayAttribute(float, n_vert, 3)
        self._baseY = ArrayAttribute(float, n_vert, 3)
        self._transport = dict()

        for u in range(n_vert):
            # find basis vector -> first edge
            P = Vec(self.mesh.vertices[u])
            N = self.vnormals[u]
            # extract basis edge
            v = self.mesh.connectivity.vertex_to_vertex(u)[0]
            E = self.mesh.vertices[v] - P
            X = Vec.normalized(E - np.dot(E,N)*N) # project on tangent plane
            self._baseX[u] = X
            self._baseY[u] = geom.cross(N,X)

            vert_u = self.mesh.connectivity.vertex_to_vertex(u)
            
            # initialize angles of every edge in this basis
            ang = 0.
            if u in self.feat.feature_vertices:
                dfct = self.feat.corners[u] * 2 * np.pi / self.feat.corner_order # target defect (multiple of pi/order)
                for v in vert_u:
                    T = self.mesh.half_edges.adj(u,v)[0]
                    self._transport[(u,v)] = ang * dfct / self.total_angle[u]
                    c = self.mesh.connectivity.vertex_to_corner_in_face(u,T)
                    if c is None: continue
                    ang += self.angles[c]
            else:
                # normal vertex in interior -> flatten to 2pi
                for v in vert_u :
                    T = self.mesh.half_edges.adj(u,v)[0]
                    self._transport[(u,v)] = ang * 2 * np.pi / self.total_angle[u]
                    c = self.mesh.connectivity.vertex_to_corner_in_face(u,T)
                    ang += self.angles[c]

            # ang = 0.
            # # if u in self.feat.feature_vertices:
            # if self.mesh.is_vertex_on_border(u) :
            #     fst, lst = vert_u[0], vert_u[-1]
            #     pfst, plst = (self.mesh.vertices[x] for x in (fst, lst))
            #     comp_angle = geom.signed_angle_3pts(plst,P,pfst, N) # complementary angle, ie "exterior" angle between two edges on the boundary
            #     comp_angle = 2*np.pi + comp_angle if comp_angle<0 else comp_angle
                
            #     for v in vert_u:
            #         T = self.mesh.half_edges.adj(u,v)[0]
            #         self._transport[(u,v)] = ang * 2 * np.pi / (self.total_angle[u] + comp_angle)
            #         if T is None : continue
            #         c = self.mesh.connectivity.vertex_to_corner_in_face(u,T)
            #         ang += self.angles[c]
            # else:
            #     for v in vert_u:
            #         T = self.mesh.half_edges.adj(u,v)[0]
            #         c = self.mesh.connectivity.vertex_to_corner_in_face(u,T)
            #         self._transport[(u,v)] = ang * 2 * np.pi / self.total_angle[u]
            #         ang += self.angles[c]

class SurfaceConnectionFaces(SurfaceConnection):
    """
    Local bases and parallel transport defined between tangent planes of the mesh at the vertices

    Reference:
        Globally Optimal Direction Fields by Knöppel et al. (2013)

    Args:
        mesh (SurfaceMesh): the supporting mesh
        feat (FeatureEdgeDetector, optional): feature edges of the mesh. If not provided, will be initialized with a FeatureEdgeDetector object that flags only the boundary. Defaults to None.
    """
    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, feat : FeatureEdgeDetector = None) -> None:
        super().__init__(mesh, feat)

    def _initialize(self):
        #### Initialize bases
        NF = len(self.mesh.faces)
        self._baseX, self._baseY = ArrayAttribute(float, NF, 3), ArrayAttribute(float, NF, 3)
        self._transport = dict()
        for id_face, (A,B,C) in enumerate(self.mesh.faces):
            feat = [self.mesh.connectivity.edge_id(_u,_v) in self.feat.feature_edges for (_u,_v) in [(A,B), (B,C), (C,A)]]
            if np.any(feat):
                # face has feature edge -> feature should be the X coordinate of the basis
                A,B,C = utils.offset([A,B,C], np.argmax(feat))
            pA,pB,pC = (self.mesh.vertices[_v] for _v in (A,B,C))
            X,Y,_ = geom.face_basis(pA,pB,pC) # local basis of the triangle (ignore normal)
            self._baseX[id_face] = X
            self._baseY[id_face] = Y

        #### Initialize Parallel Transport
        for e in self.mesh.interior_edges:
            A,B = self.mesh.edges[e]
            pA,pB = self.mesh.vertices[A], self.mesh.vertices[B]
            E = geom.Vec(pB-pA)
            T1,T2 = self.mesh.half_edges.edge_to_triangles(A,B)
            X1,Y1 = self._baseX[T1], self._baseY[T1]
            X2,Y2 = self._baseX[T2], self._baseY[T2]
            angle1 = math.atan2( geom.dot(E,Y1), geom.dot(E,X1))
            angle2 = math.atan2( geom.dot(E,Y2), geom.dot(E,X2))
            self._transport[(T1,T2)] = angle1 - angle2
            self._transport[(T2,T1)] = angle2 - angle1