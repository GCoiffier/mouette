from .base import FrameField
from ...mesh.datatypes import *
from ...mesh.mesh_attributes import Attribute, ArrayAttribute
from ... import geometry as geom
from ... import operators

from ... import utils
from ... import optimize

from ...attributes import cotangent, angle_defects, mean_edge_length
from ..features import FeatureEdgeDetector
from ..connection import SurfaceConnectionFaces
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
    def __init__(self, 
        supporting_mesh : SurfaceMesh, 
        order:int = 4, 
        feature_edges : bool = False, 
        verbose:bool=True,
        **kwargs
    ):
        super().__init__("faces", verbose=verbose)
        self.mesh : SurfaceMesh = supporting_mesh
        self.order : int = order
        self.features : bool = feature_edges # whether feature edges are enabled or not
        self.use_cotan = kwargs.get("use_cotan", True)
        self.n_smooth = kwargs.get("n_smooth", 10)
        self.smooth_attach_weight = kwargs.get("smooth_attach_weight", None)

        self.cot : Attribute = None
        self.defect : Attribute = None
        self.conn : SurfaceConnectionFaces = kwargs.get("custom_connection", None)
        self.feat : FeatureEdgeDetector = kwargs.get("custom_features", None)
        
        self.initialized = False

    def _initialize_attributes(self):
        #processing.split_double_boundary_edges_triangles(self.mesh) # A triangle has only one edge on the boundary
        self.cot = cotangent(self.mesh, persistent=False)
        self.defect = angle_defects(self.mesh,persistent=False, dense=True)
        if self.feat is None:
            self.feat = FeatureEdgeDetector(only_border = not self.features, verbose=self.verbose)(self.mesh)
        if self.conn is None:
            self.conn = SurfaceConnectionFaces(self.mesh, self.feat)

    def _initialize_variables(self):
        self.var = np.zeros(len(self.mesh.faces), dtype=complex)

        # fix orientation on features
        for e in self.feat.feature_edges:
            e1,e2 = self.mesh.edges[e] # the edge on border
            edge = self.mesh.vertices[e2] - self.mesh.vertices[e1]
            for T in self.mesh.connectivity.edge_to_faces(e1,e2):
                if T is None: continue # edge may be on boundary
                X,Y = self.conn.base(T)
                c = complex(edge.dot(X), edge.dot(Y)) # compute edge in local basis coordinates (edge.dot(Z) = 0 -> complex number for 2D vector)
                self.var[T] = (c/abs(c))**4 # c^4 is the same for all four directions of the cross

    def _compute_attach_weight(self, A, fail_value=1e-3):
        # A is area weight matrix
        lap_no_pt = operators.laplacian_triangles(self.mesh, cotan=self.use_cotan)
        try:
            eigs = sp.linalg.eigsh(lap_no_pt, k=2, M=A, which="SM", tol=1e-3, maxiter=100, return_eigenvectors=False)
        except Exception as e:
            try:
                self.log("First estimation of alpha failed: {}".format(e))
                lap_no_pt = operators.laplacian_triangles(self.mesh, cotan=False)
                eigs = sp.linalg.eigsh(lap_no_pt+1e-3*sp.identity(lap_no_pt.shape[0]), M=A, k=2, which="SM", tol=1e-3, maxiter=100, return_eigenvectors=False)
            except:
                self.log("Second estimation of alpha failed: taking alpha = ", fail_value)
                return fail_value
        eigs_non_zero = [e for e in eigs if abs(e)>1e-6]
        if len(eigs_non_zero)==0:
            return fail_value
        return abs(min(eigs_non_zero))

    def flag_singularities(self, singul_attr_name:str = "singuls"):        
        """
        Detects singularities of the frame field

        Creates 2 attributes:
            - An attribute "singuls" on vertices storing the value (+- 1) of singularities (eventually 0 for non singular vertices)
            - An attribute "angles" on edges storing the angle of the edge, given as the difference between the two frames (associated 1-form of the frame field)
        
        Args:
            singul_attr_name (str, optional): Name of the singularity attribute created. Defaults to "singuls".
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
            T1,T2 = self.mesh.connectivity.edge_to_faces(A,B)
            if T1 is None or T2 is None: continue
            f1,f2 = self.var[T1], self.var[T2] # representation complex for T1 and T2
            
            # parallel transport
            E = self.mesh.vertices[B] - self.mesh.vertices[A]
            X1,Y1 = self.conn.base(T1)
            X2,Y2 = self.conn.base(T2)
            a1 = atan2(Y1.dot(E), X1.dot(E))
            a2 = atan2(Y2.dot(E), X2.dot(E))

            # matching
            u2 = utils.maths.roots(f2, self.order)[0]
            angles = [utils.maths.angle_diff( cmath.phase(u2) - a2, cmath.phase(u1) - a1) for u1 in utils.maths.roots(f1, self.order)]
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
            for e in self.mesh.connectivity.vertex_to_edges(v):
                u = self.mesh.connectivity.other_edge_end(e,v)
                angle += edge_rot[e] if u<v else -edge_rot[e]
            if abs(angle)>ZERO_THRESHOLD:
                singuls[v] = angle*2/pi

    def export_as_mesh(self) -> PolyLine:
        """
        Exports the frame field as a set of crosses on each faces, for visualization purposes

        Returns:
            PolyLine: representation of the frame field
        """
        FFMesh = PolyLine()
        L = mean_edge_length(self.mesh)/3
        n = self.order+1
        for id_face, face in enumerate(self.mesh.faces):
            basis,Y = self.conn.base(id_face)
            normal = geom.cross(basis,Y)
            pA,pB,pC = (self.mesh.vertices[_v] for _v in face)
            angle = cmath.phase(self.var[id_face])/self.order
            bary = (pA+pB+pC)/3 # reference point for display
            cmplx = [geom.rotate_around_axis(basis, normal, angle + 2*k*pi/self.order) for k in range(self.order)]
            pts = [bary + abs(self.var[id_face])*L*r for r in cmplx]
            FFMesh.vertices.append(bary)
            FFMesh.vertices += pts
            FFMesh.edges += [(n*id_face, n*id_face+k) for k in range(1,n)]
        return FFMesh

class FrameField2DFaces(_BaseFrameField2DFaces) : 

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
        supporting_mesh : SurfaceMesh, 
        order:int = 4,  
        feature_edges : bool = False, 
        verbose:bool=True,
        **kwargs):
        super().__init__(supporting_mesh, order, feature_edges, verbose, **kwargs)
           
    def initialize(self):
        self._initialize_attributes()
        self._initialize_variables()
        self.initialized = True

    def optimize(self):
        self._check_init()
    
        self.log("Build laplacian operator")
        lap = operators.laplacian_triangles(
            self.mesh, 
            cotan=self.use_cotan, 
            connection=self.conn, 
            order=self.order).tocsc()
        A = operators.area_weight_matrix_faces(self.mesh).tocsc()

        ###### Border ######
        if len(self.feat.feature_vertices)>0: # We have a border / feature elements -> linear solve
            # Build fixed and variable indexes
            self.log("Feature element detected (border and/or feature edges)")

            fixed = self.mesh.faces.create_attribute("fixed", bool)
            for ie in self.feat.feature_edges:
                u,v = self.mesh.edges[ie]
                T1,T2 = self.mesh.connectivity.edge_to_faces(u,v)
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
            AI = A[freeInds, :][:, freeInds].astype(complex)
            valB = lapB.dot(self.var[fixedInds]) # right hand side
            
            self.log("Initial solve of linear system")
            res = linalg.spsolve(lapI, -valB) # first system solved without diffusion
            self.var[freeInds] = res

            if self.n_smooth>0:
                self.log(f"Solve linear system {self.n_smooth} times with diffusion")
                alpha = self.smooth_attach_weight or self._compute_attach_weight(A)
                self.log("Attach weight: {}".format(alpha))
                mat = lapI - alpha * AI
                for _ in range(self.n_smooth):
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
            if self.n_smooth>0:
                self.log(f"Solve linear system {self.n_smooth} times with diffusion")
                alpha = self.smooth_attach_weight or self._compute_attach_weight(A)
                self.log("Attach weight: {}".format(alpha))
                mat = lap - alpha * A
                solve = sp.linalg.factorized(mat)
                for _ in range(self.n_smooth):
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
    def __init__(self, 
        supporting_mesh : SurfaceMesh, 
        singus_indices : Attribute, 
        order : int = 4, 
        verbose : bool=True,
        **kwargs):
        super().__init__(supporting_mesh, order, feature_edges=False, verbose=verbose, **kwargs)
        self.singus = singus_indices
        self.rotations : np.ndarray = None

    def initialize(self):
        self._initialize_attributes() # /!\ may change mesh combinatorics near boundary
        self.var = np.zeros(len(self.mesh.faces), dtype=complex)
        self.initialized = True

    def optimize(self):
        self._check_init()
        ### Optimize for rotations between frames
        n_cstr = len(self.mesh.vertices)
        n_rot = len(self.mesh.edges)
        CstMat = sp.lil_matrix((n_cstr,n_rot))
        CstX = np.zeros(n_cstr)
        r = 0
        for v in self.mesh.interior_vertices:
            for e in self.mesh.connectivity.vertex_to_edges(v):
                v2 = self.mesh.connectivity.other_edge_end(e,v)
                CstMat[r,e] = 1 if v<v2 else -1
            CstX[r] = self.defect[v] - self.singus[v] * 2 * pi / self.order
            r += 1

        for v in self.mesh.boundary_vertices:
            for v2 in self.mesh.connectivity.vertex_to_vertices(v):
                if self.mesh.is_edge_on_border(v,v2): continue
                e = self.mesh.connectivity.edge_id(v,v2)
                CstMat[r,e] = 1 if v<v2 else -1
            CstX[r] = self.defect[v] - (2 - self.feat.corners[v]) * 2 * pi / self.order
            r += 1
        
        instance = OSQP()
        instance.setup(P = sp.eye(n_rot,format="csc"), q=None, A=CstMat.tocsc(), l=CstX, u=CstX, verbose=self.verbose)
        res = instance.solve()
        if res.info.status != "solved":
            raise Exception(f"Solver exited with status '{res.info.status}'. Check validity of singularity indices (Poincarré-Hopf condition)")
        self.rotations = res.x

        ### Now rebuild frame field along a tree
        for _e0 in self.feat.feature_edges : break # get a feature edge
        root = [T for T in self.mesh.connectivity.edge_to_faces(*self.mesh.edges[_e0]) if T is not None][0]
        tree = trees.FaceSpanningTree(self.mesh, root)()
        for face,parent in tree.traverse():
            if parent is None: # root
                self.var[face] = complex(1., 0.)
                continue
            zp = self.var[parent]
            ea,eb = self.mesh.connectivity.common_edge(parent,face)
            ea,eb = min(ea,eb),max(ea,eb)
            e = self.mesh.connectivity.edge_id(ea,eb)
            X1,Y1 = self.conn.base(parent)
            X2,Y2 = self.conn.base(face)
            # T1 -> T2 : angle of e in basis of T2 - angle of e in basis of T1
            E = self.mesh.vertices[eb] - self.mesh.vertices[ea]
            angle1 = atan2( geom.dot(E,Y1), geom.dot(E,X1))
            angle2 = atan2( geom.dot(E,Y2), geom.dot(E,X2))
            pt = utils.maths.principal_angle(angle2 - angle1)
            w = self.rotations[e] if self.mesh.connectivity.direct_face(ea,eb)==parent else -self.rotations[e]
            zf = zp * cmath.rect(1, 4*(w + pt))
            self.var[face] = zf
        self.smoothed = True