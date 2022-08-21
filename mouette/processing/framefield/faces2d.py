from .base import FrameField
from ...mesh.datatypes import *
from ...mesh.mesh_attributes import Attribute, ArrayAttribute
from ... import geometry as geom
from ...geometry import Vec
from ... import operators
from ...utils.maths import roots, angle_diff
from ...attributes import cotangent, angle_defects, mean_edge_length, curvature_matrices
from ..features import FeatureEdgeDetector

import numpy as np
from math import pi, atan2
import cmath
import scipy.sparse as sp
from scipy.sparse import linalg

class _BaseFrameField2DFaces(FrameField) : 

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, supporting_mesh : SurfaceMesh, order:int = 4, feature_edges : bool = False, verbose:bool=True):
        super().__init__(verbose=verbose)
        self.mesh : SurfaceMesh = supporting_mesh
        self.order : int = order

        self.cot : Attribute = None
        self.defect : Attribute = None
        self.tbaseX : Attribute = None # local basis X vector (on triangles)
        self.tbaseY : Attribute = None # local basis Y vector (on triangles)
        self.tnormals : Attribute = None # local basis Z vector (= mesh normal at triangles)

        self.features : bool = feature_edges # whether feature edges are enabled or not
        self.feat : FeatureEdgeDetector = None
        
        self.initialized = False

    def _initialize_attributes(self):
        #processing.split_double_boundary_edges_triangles(self.mesh) # A triangle has only one edge on the boundary
        self.cot = cotangent(self.mesh)
        self.defect = angle_defects(self.mesh,persistent=False)

    def _initialize_features(self):
        self.feat = FeatureEdgeDetector(only_border = not self.features, verbose=self.verbose)(self.mesh)

    def _initialize_bases(self):
        NF = len(self.mesh.faces)
        self.tbaseX, self.tbaseY = ArrayAttribute(float, NF, 3), ArrayAttribute(float, NF, 3)
        self.tnormals = self.mesh.faces.create_attribute("normals", float, 3, dense=True)
        for id_face, (A,B,C) in enumerate(self.mesh.faces):
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

    def flag_singularities(self):
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
            E = geom.Vec.normalized(self.mesh.vertices[B] - self.mesh.vertices[A])
            a1 = atan2(self.tbaseY[T1].dot(E), self.tbaseX[T1].dot(E))
            a2 = atan2(self.tbaseY[T2].dot(E), self.tbaseX[T2].dot(E))

            # matching
            u2 = roots(f2, self.order)[0]
            angles = [angle_diff( cmath.phase(u2) - a2, cmath.phase(u1) - a1) for u1 in roots(f1, self.order)]
            abs_angles = [abs(_a) for _a in angles]
            i_angle = np.argmin(abs_angles)
            edge_rot[ie] = angles[i_angle]

        if self.mesh.vertices.has_attribute("singuls"):
            singuls = self.mesh.vertices.get_attribute("singuls")
            singuls.clear()
        else:
            singuls = self.mesh.vertices.create_attribute("singuls", float)
        
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
            pA,pB,pC = (self.mesh.vertices[u] for u in face)
            basis,_,normal = geom.face_basis(pA,pB,pC)

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
                alpha = 1e-3 * mean_edge_length(self.mesh)
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
            try:
                egv, U = sp.linalg.eigsh(lap, k=1, M=A, which="SM", tol=1e-4,maxiter=1000)
            except linalg.ArpackNoConvergence as e:
                self.log("Initial eigensolve failed :", e)
                self.log("Retry using connectivity laplacian")
                lap = operators.laplacian_triangles(self.mesh, cotan=False).tocsc()
                egv, U = sp.linalg.eigsh(lap, k=2, M=A, which="SM", tol=1e-4)

            alpha = abs(egv[0])
            self.log("Eigenvalues:", egv)
            self.var = U[:,0]

            if n_renorm>0:
                self.log(f"Solve linear system {n_renorm} times with diffusion")
                mat = lap - alpha * A
                for _ in range(n_renorm):
                    self.normalize()
                    valI2 = alpha * A.dot(self.var)
                    self.var = linalg.spsolve(mat, -valI2)
            self.normalize()

class CurvatureFaces(_BaseFrameField2DFaces):

    def __init__(self, supporting_mesh : SurfaceMesh, verbose :bool = False):
        super().__init__(supporting_mesh, 4, False, verbose)
        self.curv_mat_edges : np.ndarray = None # shape (|E|,3,3)
        self.curv_mat_faces : np.ndarray = None # shape (|T|,3,3)

    def _initialize_curvature(self):
        self.curv_mat_edges = curvature_matrices(self.mesh)
        adj_e = dict([(e,set()) for e in self.mesh.id_edges])
        for T in self.mesh.id_faces:
            for T2 in self.mesh.connectivity.face_to_face(T):
                for e in self.mesh.connectivity.face_to_edge(T2):
                    adj_e[e].add(T)

        self.curv_mat_faces = np.zeros((len(self.mesh.faces),3,3))
        for e in self.mesh.id_edges:
            for T in adj_e[e]:
                self.curv_mat_faces[T] += self.curv_mat_edges[e]

    def run(self, n_smooth=0):
        self.initialize()
        self.log("Optimize")
        self.optimize(n_smooth)
        self.log("Done.")
        return self

    def initialize(self):
        self.log("Initialize attributes")
        self._initialize_attributes() # /!\ may change mesh combinatorics near boundary
        self._initialize_features()
        self.log("Initialize bases")
        self._initialize_bases()
        self.log("Compute curvature matrices")
        self._initialize_curvature()
        self.log("Initialize variables")
        self._initialize_variables()
        self.initialized = True

    def optimize(self, n_smooth):
        for T in self.mesh.id_faces:
            if abs(self.var[T])!=0: continue # already initialized => T is on the boundary
            u,s,_ = np.linalg.svd(self.curv_mat_faces[T,:,:], hermitian=True)
            X,Y = self.tbaseX[T], self.tbaseY[T]
            # three vectors -> normal and two principal components
            # eigenvalue of normal is 0
            # PC are orthogonal (eigenvects of symmetric matrix) -> we rely on the eigenvect of greatest eigenvalue and take orthog direction
            if s[0]<1e-10 : # zero matrix -> no curvature information -> we take the first basis vector and hope to smooth correctly
                self.var[T] = 0 + 0j
            else:
                # representation vector
                v = u[:,0]
                c = complex(X.dot(v), Y.dot(v))
                self.var[T] = (c/abs(c))**4

        if n_smooth > 0:
            # diffuse the curvature results to get a smoother results (especially where curvature was not defined)
            lap = operators.laplacian_triangles(self.mesh, order=self.order)
            A = operators.area_weight_matrix_faces(self.mesh).astype(complex)
            alpha = 1e-3

            if len(self.feat.feature_vertices)>0: # we have a boundary
                pass
            
            else:
                mat = lap - alpha * A
                for _ in range(n_smooth):
                        valI2 = alpha * A.dot(self.var)
                        self.var = linalg.spsolve(mat, - valI2)
                        self.normalize()