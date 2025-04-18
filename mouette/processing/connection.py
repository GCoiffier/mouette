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
        self._baseX : ArrayAttribute = None
        self._baseY : ArrayAttribute = None
        self._transport : dict = None
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        pass

    def transport(self, iA : int, iB : int) -> float:
        """_summary_

        Args:
            iA (int): first vertex index
            iB (int): second vertex index

        Returns:
            float: angle needed to change from basis of iA to basis of iB
        """
        return self._transport[(iA,iB)]

    def base(self, i:int):
        """Local basis of element i

        Args:
            i (int): index

        Returns:
            (Vec,Vec): the X vector and Y vector of the tangent basis.

        Note:
            `cross(X,Y)` is the normal of element i
        """
        return self._baseX[i], self._baseY[i]

    @property
    def bX(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: local X vectors of bases as a 2D numpy array
        """
        return self._baseX.as_array()
    
    @property
    def bY(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: local Y vectors of bases as a 2D numpy array
        """
        return self._baseY.as_array()

    def project(self, V:Vec, i:int):
        return Vec(self._baseX[i].dot(V), self._baseY[i].dot(V))


class SurfaceConnectionVertices(SurfaceConnection):
    """
    Local bases and parallel transport defined between tangent planes of the mesh at the vertices

    Reference:
        _Globally Optimal Direction Fields_, Knöppel et al. (2013)
    
    Args:
        mesh (SurfaceMesh): the supporting mesh
        feat (FeatureEdgeDetector, optional): feature edges of the mesh. If not provided, will be initialized with a FeatureEdgeDetector object that flags only the boundary. Defaults to None.
    
    Keyword Args:
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
            vert_u = self.mesh.connectivity.vertex_to_vertices(u)[::-1]
            E = self.mesh.vertices[vert_u[0]] - P
            X = Vec.normalized(E - np.dot(E,N)*N) # project on tangent plane
            self._baseX[u] = X
            self._baseY[u] = geom.cross(N,X)

            # initialize angles of every edge in this basis
            ang = 0.
            if u in self.feat.feature_vertices:
                dfct = self.feat.corners[u] * 2 * np.pi / self.feat.corner_order # target defect (multiple of pi/order)
                for v in vert_u:
                    T = self.mesh.connectivity.direct_face(u,v)
                    self._transport[(u,v)] = ang * dfct / self.total_angle[u]
                    c = self.mesh.connectivity.vertex_to_corner_in_face(u,T)
                    if c is None: continue
                    ang += self.angles[c]
            else:
                # normal vertex in interior -> flatten to 2pi
                for v in vert_u:
                    T = self.mesh.connectivity.direct_face(u,v)
                    self._transport[(u,v)] = ang * 2 * np.pi / self.total_angle[u]
                    c = self.mesh.connectivity.vertex_to_corner_in_face(u,T)
                    ang += self.angles[c]

class FlatConnectionVertices(SurfaceConnection):
    """
    Surface connection on vertices where all local bases are taken as the canonical basis ([1 0 0], [0 1 0]).
    /!\\ This only makes sense if the considered surface is embedded in R^2, meaning that there is no curvature.

    Args:
        mesh (SurfaceMesh): the supporting mesh
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh):
        super().__init__(mesh, None)
        self._baseX = Vec.X()
        _,_,sample_normal = geom.face_basis(*(self.mesh.vertices[_u] for _u in self.mesh.faces[0][:3]))
        if sample_normal.z<0:
            self._baseY = -Vec.Y()
        else:
            self._baseY = Vec.Y()

    def _initialize(self):
        return # nothing to do

    def transport(self, iA,iB):
        E = self.mesh.vertices[iB] - self.mesh.vertices[iA]
        return np.arctan2(E.y,E.x)

    def base(self, i:int):
        # Same base everywhere
        return self._baseX, self._baseY

    def project(self, V:Vec, i:int):
        # projection does not depend on the considered vertex i
        return Vec(self._baseX.dot(V), self._baseY.dot(V))

class SurfaceConnectionFaces(SurfaceConnection):
    """
    Local bases and parallel transport defined between tangent planes of the mesh at the vertices

    Reference:
        _Globally Optimal Direction Fields_, Knöppel et al. (2013)

    Args:
        mesh (SurfaceMesh): the supporting mesh
        feat (FeatureEdgeDetector, optional): feature edges of the mesh. If not provided, will be initialized with a FeatureEdgeDetector object that flags only the boundary. Defaults to None.
    """
    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, feat : FeatureEdgeDetector = None):
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
            T1,T2 = self.mesh.connectivity.edge_to_faces(A,B)
            X1,Y1 = self._baseX[T1], self._baseY[T1]
            X2,Y2 = self._baseX[T2], self._baseY[T2]
            angle1 = math.atan2( geom.dot(E,Y1), geom.dot(E,X1))
            angle2 = math.atan2( geom.dot(E,Y2), geom.dot(E,X2))
            self._transport[(T1,T2)] = angle1 - angle2
            self._transport[(T2,T1)] = angle2 - angle1

class FlatConnectionFaces(SurfaceConnection):
    """
    Surface connection on faces where all local bases are taken as the canonical basis ([1 0 0], [0 1 0]).
    /!\\ This only makes sense if the considered surface is embedded in R^2, meaning that there is no curvature.

    Args:
        mesh (SurfaceMesh): the supporting mesh
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh):
        super().__init__(mesh, None)
        self._baseX = Vec.X()
        _,_,sample_normal = geom.face_basis(*(self.mesh.vertices[_u] for _u in self.mesh.faces[0][:3]))
        if sample_normal.z<0:
            self._baseY = -Vec.Y()
        else:
            self._baseY = Vec.Y()

    def _initialize(self):
        return # nothing to do

    def transport(self, iA,iB):
        return 0.

    def base(self, i:int):
        # Same base everywhere
        return self._baseX, self._baseY

    def project(self, V:Vec, i:int):
        # projection does not depend on the considered vertex i
        return Vec(self._baseX.dot(V), self._baseY.dot(V))
    


class SurfaceConnectionEdges(SurfaceConnection):
    """
    Local bases and parallel transport defined between tangent planes of the mesh at edges

    Reference:
        _A Heat Method for Generalized Signed Distance_, Feng and Crane (2024)
    
    Args:
        mesh (SurfaceMesh): the supporting mesh
    
    Keyword Args:
        enormals (Attribute): An attribute representing the edge normals. If not provided, will be computed at initialization
        angles (Attribute) : An attribute representing angles at corners of faces. If not provided, will be computed at initialization
    
    Raises:
        AssertionError: fails if the input mesh is not triangular
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, **kwargs):
        assert mesh.is_triangular()
        self.enormals : ArrayAttribute = kwargs.get("enormals", None)
        self.fnormals : ArrayAttribute = attributes.face_normals(mesh)
        if self.enormals is None:
            self.enormals = mesh.edges.create_attribute("normals", float, 3, dense=True)
        super().__init__(mesh)

    def _initialize(self):
        #### Initialize bases
        NE = len(self.mesh.edges)
        self._baseX, self._baseY = ArrayAttribute(float, NE, 3), ArrayAttribute(float, NE, 3)
        self._transport = dict()
        for id_edge,(A,B) in enumerate(self.mesh.edges):
            pA,pB = self.mesh.vertices[A], self.mesh.vertices[B]
            for adj_face in self.mesh.connectivity.edge_to_faces(A,B):
                if adj_face is None: continue
                self.enormals[id_edge] += self.fnormals[adj_face]
            self.enormals[id_edge] = Vec.normalized(self.enormals[id_edge])
            self._baseX[id_edge] = Vec.normalized(pB-pA)
            self._baseY[id_edge] = Vec.normalized(geom.cross(self.enormals[id_edge], self._baseX[id_edge]))

        #### Initialize Parallel Transport
        for F,(A,B,C) in enumerate(self.mesh.faces):
            e1,s1 = self.mesh.connectivity.edge_id(A,B), (1 if A<B else -1)
            e2,s2 = self.mesh.connectivity.edge_id(B,C), (1 if B<C else -1)
            e3,s3 = self.mesh.connectivity.edge_id(C,A), (1 if C<A else -1)
            
            X1,X2,X3 = (self._baseX[_e] for _e in (e1,e2,e3))

            self._transport[(e1,e2)] = - geom.signed_angle_2vec3D(s1*X1, s2*X2, self.fnormals[F])
            self._transport[(e2,e1)] = - self._transport[(e1,e2)]

            self._transport[(e1,e3)] = - geom.signed_angle_2vec3D(s1*X1, s3*X3, self.fnormals[F])
            self._transport[(e3,e1)] = - self._transport[(e1,e3)]
            
            self._transport[(e2,e3)] = - geom.signed_angle_2vec3D(s2*X2, s3*X3, self.fnormals[F])
            self._transport[(e3,e2)] = - self._transport[(e2,e3)]
