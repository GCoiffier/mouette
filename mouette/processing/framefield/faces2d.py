from .base import FrameField
from ...mesh.datatypes import *
from ...mesh.mesh_attributes import Attribute, ArrayAttribute
from ... import geometry as geom
from ... import operators

from ...utils.maths import *
from ... import optimize

from ...attributes import cotangent, angle_defects, mean_edge_length
from ..features import FeatureEdgeDetector
from .. import trees

import numpy as np
from math import pi, atan2
import cmath
import scipy.sparse as sp
from scipy.sparse import linalg
from osqp import OSQP

class _BaseFrameField2DFaces(FrameField) : 
    """
    Base class for any frame field defined on the vertices of a surface mesh. Is not meant to be instanciated as is.
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, supporting_mesh : SurfaceMesh, order:int = 4, feature_edges : bool = False, verbose:bool=True):
        super().__init__(verbose=verbose)
        self.mesh : SurfaceMesh = supporting_mesh
        self.order : int = order

        self.cot : Attribute = None
        self.defect : Attribute = None
        self.tbaseX : Attribute = None # local basis X vector (on triangles)
        self.tbaseY : Attribute = None # local basis Y vector (on triangles)
        self.tnormals : Attribute = None # local normal vector

        self.features : bool = feature_edges # whether feature edges are enabled or not
        self.feat : FeatureEdgeDetector = None
        
        self.initialized = False

    def _initialize_attributes(self):
        #processing.split_double_boundary_edges_triangles(self.mesh) # A triangle has only one edge on the boundary
        self.cot = cotangent(self.mesh)
        self.defect = angle_defects(self.mesh,persistent=False, dense=True)

    def _initialize_features(self):
        self.feat = FeatureEdgeDetector(only_border = not self.features, verbose=self.verbose)(self.mesh)

    def _initialize_bases(self):
        NF = len(self.mesh.faces)
        self.tbaseX, self.tbaseY = ArrayAttribute(float, NF, 3), ArrayAttribute(float, NF, 3)
        self.tnormals = self.mesh.faces.create_attribute("normals", float, 3, dense=True)
        for id_face, (A,B,C) in enumerate(self.mesh.faces):
            # bnd = [self.mesh.is_edge_on_border(_u,_v) for (_u,_v) in [(A,B),(B,C), (C,A)]]
            # if np.any(bnd):
            #     # face is on the boundary:
            #     A,B,C = utils.offset([A,B,C],np.argmax(bnd))
            #     # assert self.mesh.is_edge_on_border(A,B)
            pA,pB,pC = (self.mesh.vertices[_v] for _v in (A,B,C))
            X,Y,Z = geom.face_basis(pA,pB,pC) # local basis of the triangle
            self.tbaseX[id_face] = X
            self.tbaseY[id_face] = Y
            self.tnormals[id_face] = Z

    def _initialize_variables(self):
        self.var = np.zeros(len(self.mesh.faces), dtype=complex)

        # fix orientation on features
        for e in self.feat.feature_edges:
            e1,e2 = self.mesh.edges[e] # the edge on border
            edge = self.mesh.vertices[e2] - self.mesh.vertices[e1]
            for T in self.mesh.half_edges.edge_to_triangles(e1,e2):
                if T is None: continue # edge may be on boundary
                X,Y = self.tbaseX[T], self.tbaseY[T]
                c = complex(edge.dot(X), edge.dot(Y)) # compute edge in local basis coordinates (edge.dot(Z) = 0 -> complex number for 2D vector)
                self.var[T] = (c/abs(c))**4 # c^4 is the same for all four directions of the cross

    def flag_singularities(self, singul_attr_name:str = "singuls"):
        """Compute singularity data.

        Creates 2 attributes:
            - an attribute "singuls" on vertices storing the value (+- 1) of singularities (eventually 0 for non singular vertices)
            - an attribute "angles" on edges storing the angle of the edge, given as the difference between the two frames
        """
        self._check_init()
        ZERO_THRESHOLD = 1e-3

        if self.mesh.edges.has_attribute("angles"):
            edge_rot = self.mesh.edges.get_attribute("angles")
            edge_rot.clear()
        else:
            edge_rot = self.mesh.edges.create_attribute("angles", float, 1, dense=True)
            # the rotation induced by the frame field on every edge
            # if edge is uv, positive orientation is from T(uv) to T(vu)

        for ie,(A,B) in enumerate(self.mesh.edges):
            T1,T2 = self.mesh.half_edges.edge_to_triangles(A,B)
            if T1 is None or T2 is None: continue
            f1,f2 = self.var[T1], self.var[T2] # representation complex for T1 and T2
            
            # parallel transport
            E = self.mesh.vertices[B] - self.mesh.vertices[A]
            a1 = atan2(self.tbaseY[T1].dot(E), self.tbaseX[T1].dot(E))
            a2 = atan2(self.tbaseY[T2].dot(E), self.tbaseX[T2].dot(E))

            # matching
            u2 = roots(f2, self.order)[0]
            angles = [angle_diff( cmath.phase(u2) - a2, cmath.phase(u1) - a1) for u1 in roots(f1, self.order)]
            abs_angles = [abs(_a) for _a in angles]
            i_angle = np.argmin(abs_angles)
            edge_rot[ie] = angles[i_angle]

        if self.mesh.vertices.has_attribute(singul_attr_name):
            singuls = self.mesh.vertices.get_attribute(singul_attr_name)
            singuls.clear()
        else:
            singuls = self.mesh.vertices.create_attribute(singul_attr_name, float)
        
        for v in self.mesh.id_vertices:
            angle = self.defect[v]
            for e in self.mesh.connectivity.vertex_to_edge(v):
                u = self.mesh.connectivity.other_edge_end(e,v)
                angle += edge_rot[e] if u<v else -edge_rot[e]
            if abs(angle)>ZERO_THRESHOLD:
                singuls[v] = angle*2/pi

    def export_as_mesh(self) -> PolyLine:
        FFMesh = PolyLine()
        L = mean_edge_length(self.mesh)/3
        for id_face, face in enumerate(self.mesh.faces):
            basis, normal = self.tbaseX[id_face], self.tnormals[id_face]
            pA,pB,pC = (self.mesh.vertices[_v] for _v in face)
            angle = cmath.phase(self.var[id_face])/4
            bary = (pA+pB+pC)/3 # reference point for display
            r1,r2,r3,r4 = (geom.rotate_around_axis(basis, normal, angle + k*pi/2) for k in range(4))
            p1,p2,p3,p4 = (bary + abs(self.var[id_face])*L*r for r in (r1,r2,r3,r4))
            FFMesh.vertices += [bary, p1, p2, p3, p4]
            FFMesh.edges += [(5*id_face, 5*id_face+k) for k in range(1,5)]            
        return FFMesh

class FrameField2DFaces(_BaseFrameField2DFaces) : 

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, supporting_mesh : SurfaceMesh, order:int = 4,  feature_edges : bool = False, verbose:bool=True):
        super().__init__(supporting_mesh, order,feature_edges, verbose)
           
    def initialize(self):
        self._initialize_attributes() # /!\ may change mesh combinatorics near boundary
        self._initialize_features()
        self._initialize_bases()
        self._initialize_variables()
        self.initialized = True

    def optimize(self, n_renorm=10):
        self._check_init()
    
        self.log("Build laplacian operator")
        lap = operators.laplacian_triangles(self.mesh, cotan=False, order=self.order).tocsc()
        A = operators.area_weight_matrix_faces(self.mesh).tocsc().astype(complex)

        ###### Border ######
        if len(self.feat.feature_vertices)>0: # We have a border / feature elements -> linear solve
            # Build fixed and variable indexes
            self.log("Feature element detected (border and/or feature edges)")

            fixed = self.mesh.faces.create_attribute("fixed", bool)
            for ie in self.feat.feature_edges:
                u,v = self.mesh.edges[ie]
                T1,T2 = self.mesh.half_edges.edge_to_triangles(u,v)
                if T1 is not None: fixed[T1] = True
                if T2 is not None: fixed[T2] = True
            freeInds,fixedInds = [],[]
            for T in self.mesh.id_faces:
                if fixed[T] : fixedInds.append(T)
                else: freeInds.append(T)
              
            if len(freeInds)==0 : 
                self.log("Everything is on boundary : no optimization required")
                return

            lapI = lap[freeInds,:][:,freeInds]
            lapB = lap[freeInds,:][:,fixedInds]
            # for lapI and lapB, only lines of freeInds are relevant : lines of fixedInds link fixed variables -> useless constraints
            AI = A[freeInds, :][:, freeInds]
            valB = lapB.dot(self.var[fixedInds]) # right hand side
            
            self.log("Initial solve of linear system")
            res = linalg.spsolve(lapI, -valB) # first system solved without diffusion
            self.var[freeInds] = res

            if n_renorm>0:
                self.log(f"Solve linear system {n_renorm} times with diffusion")
                alpha = 5e-3 * mean_edge_length(self.mesh)
                self.log("Attach weight: {}".format(alpha))
                mat = lapI - alpha * AI
                for _ in range(n_renorm):
                    self.normalize()
                    valI2 = alpha * AI.dot(self.var[freeInds])
                    res = linalg.spsolve(mat, -valB - valI2)
                    self.var[freeInds] = res
            self.normalize()

        ###### No border ######
        else:
            self.log("No border detected")
            self.log("Initial solve of linear system using an eigensolver")
            self.var = optimize.inverse_power_method(lap)
            if n_renorm>0:
                alpha = 0.01
                self.log(f"Solve linear system {n_renorm} times with diffusion")
                mat = lap - alpha * A
                solve = sp.linalg.factorized(mat)
                for _ in range(n_renorm):
                    self.normalize()
                    valI2 = alpha * A.dot(self.var)
                    self.var = solve(-valI2)
            self.normalize()

class TrivialConnectionFaces(_BaseFrameField2DFaces):
    """
    Implementation of 'Trivial Connections on Discrete Surfaces' by Keenan Crane and Mathieu Desbrun and Peter Schröder, 2010
    
    A frame field on faces that computes the smoothest possible frame field with prescribed singularity cones at some vertices.
    Does not constraint non-contractible cycles
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, supporting_mesh : SurfaceMesh, singus_indices:Attribute, order:int = 4, verbose:bool=True):
        super().__init__(supporting_mesh, order, feature_edges=False, verbose=verbose)
        self.singus = singus_indices
        self.rotations : np.ndarray = None

    def initialize(self):
        self._initialize_attributes() # /!\ may change mesh combinatorics near boundary
        self._initialize_bases()
        self.var = np.zeros(len(self.mesh.faces), dtype=complex)
        self.initialized = True

    def optimize(self):

        ### Optimize for rotations between frames
        n_cstr = len(self.mesh.interior_vertices)
        # if not self.free_bnd:
        #     n_cstr += len(self.feat.feature_edges)
        n_rot = len(self.mesh.edges)
        CstMat = sp.lil_matrix((n_cstr,n_rot))
        CstX = np.zeros(n_cstr)
        for i,v in enumerate(self.mesh.interior_vertices):
            for e in self.mesh.connectivity.vertex_to_edge(v):
                v2 = self.mesh.connectivity.other_edge_end(e,v)
                CstMat[i,e] = 1 if v<v2 else -1
            CstX[i] = self.defect[v] - self.singus[v] * 2 * pi / self.order
        instance = OSQP()
        instance.setup(P = sp.eye(n_rot,format="csc"), q=None, A=CstMat.tocsc(), l=CstX, u=CstX)
        res = instance.solve()
        self.rotations = res.x

        ### Now rebuild frame field along a tree
        tree = trees.FaceSpanningTree(self.mesh)()
        for face,parent in tree.traverse():
            if parent is None: # root
                self.var[face] = complex(1., 0.)
                continue
            zp = self.var[parent]
            ea,eb = self.mesh.half_edges.common_edge(parent,face)
            ea,eb = min(ea,eb),max(ea,eb)
            e = self.mesh.connectivity.edge_id(ea,eb)
            X1,Y1 = self.tbaseX[parent], self.tbaseY[parent]
            X2,Y2 = self.tbaseX[face], self.tbaseY[face]
            # T1 -> T2 : angle of e in basis of T2 - angle of e in basis of T1
            E = self.mesh.vertices[eb] - self.mesh.vertices[ea]
            angle1 = atan2( geom.dot(E,Y1), geom.dot(E,X1))
            angle2 = atan2( geom.dot(E,Y2), geom.dot(E,X2))
            pt = principal_angle(angle2 - angle1)
            w = self.rotations[e] if self.mesh.half_edges.adj(ea,eb)[0]==parent else -self.rotations[e]
            zf = zp * cmath.rect(1, 4*(w + pt))
            self.var[face] = zf