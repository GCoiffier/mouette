from .base import FrameField
from ...mesh.mesh_data import RawMeshData
from ...mesh.datatypes import *
from ...mesh.mesh_attributes import Attribute, ArrayAttribute
from ... import geometry as geom
from ... import operators
from ... import attributes
from ...utils import maths

from ...optimize import inverse_power_method

from ...attributes import cotangent, mean_edge_length
from ..features import FeatureEdgeDetector
from ..connection import SurfaceConnectionEdges

import numpy as np
from math import pi, atan2
import cmath
import scipy.sparse as sp
from scipy.sparse import linalg


class FrameField2DEdges(FrameField) : 
    """
    n-RoSy frame field defined on the edges of a surface mesh
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
        self.conn : SurfaceConnectionEdges = kwargs.get("custom_connection", None)
        self.feat : FeatureEdgeDetector = kwargs.get("custom_features", None)
        self.initialized = False


    def initialize(self):
        self._initialize_attributes()
        self._initialize_variables()
        self.initialized = True


    def _initialize_attributes(self):
        self.cot = cotangent(self.mesh, persistent=False)
        if self.feat is None:
            self.feat = FeatureEdgeDetector(only_border = not self.features, verbose=self.verbose)(self.mesh)
        if self.conn is None:
            self.conn = SurfaceConnectionEdges(self.mesh)

    def _initialize_variables(self):
        self.var = np.zeros(len(self.mesh.edges), dtype=complex)
        for e in self.feat.feature_edges:
            # fix orientation on features
            self.var[e] = 1.

    def _compute_attach_weight(self, A, fail_value=1e-3):
        # A is area weight matrix
        lap_no_pt = operators.laplacian_edges(self.mesh, cotan=self.use_cotan)
        try:
            eigs = sp.linalg.eigsh(lap_no_pt, k=2, M=A, which="SM", tol=1e-3, maxiter=100, return_eigenvectors=False)
        except Exception as e:
            try:
                self.log("First estimation of alpha failed: {}".format(e))
                lap_no_pt = operators.laplacian_edges(self.mesh, cotan=False)
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
        Detects singularities of the frame field. Singularities of an edge-based frame field can appear both at vertices and inside faces.

        Creates 3 attributes:
            - An attribute "<singul_attr_name>" on *vertices* storing the value (+- 1) of singularities (eventually 0 for non singular vertices)
            - An attribute "<singul_attr_name>" on *faces* storing the values (+- 1) of singularities 
            - An attribute "ff_angles" on faces corners storing the angle between two edge frames
        
        Args:
            singul_attr_name (str, optional): Name of the singularity attribute created. Defaults to "singuls".
        """
        self._check_init()
        ZERO_THRESHOLD = 1e-3

        cnct = self.mesh.connectivity
        
        # compute all corner rotations
        angles = self.mesh.face_corners.create_attribute("ff_angles", float)
        for c in self.mesh.id_corners:
            e1 = cnct.edge_id(*cnct.corner_to_half_edge(cnct.previous_corner(c)))
            e2 = cnct.edge_id(*cnct.corner_to_half_edge(c))
        
            frame1, frame2 = self.var[e1], self.var[e2]
            pt = self.conn.transport(e1,e2)

            u1 = maths.roots(frame1, self.order)[0]
            phase1 = cmath.phase(u1)
            abs_angles12 = [abs(maths.angle_diff(phase1, cmath.phase(u2) - pt)) for u2 in maths.roots(frame2, self.order)]
            angles12 = [maths.angle_diff(phase1, cmath.phase(u2) - pt) for u2 in maths.roots(frame2, self.order)]
            i_angle = np.argmin(abs_angles12)
            angles[c] = angles12[i_angle]

        # compute singularities 
        singulsV = self.mesh.vertices.create_attribute(singul_attr_name, float)
        defectsV = attributes.average_corners_to_vertices(self.mesh, angles, ArrayAttribute(float, len(self.mesh.vertices)), weight="sum")
        angle_defect = attributes.angle_defects(self.mesh, persistent=False)
       
        for v in self.mesh.id_vertices:
            if defectsV[v]+angle_defect[v] > ZERO_THRESHOLD:
                singulsV[v] = 1
            elif defectsV[v]+angle_defect[v] < -ZERO_THRESHOLD:
                singulsV[v] = -1
        
        defectsF = attributes.average_corners_to_faces(self.mesh, angles, ArrayAttribute(float, len(self.mesh.faces)) , weight="sum")
        singulsF = self.mesh.faces.create_attribute(singul_attr_name, float)
        for F in self.mesh.id_faces:
            if defectsF[F]>ZERO_THRESHOLD:
                singulsF[F] = -1 # sign is reversed from vertices
            elif defectsF[F]<-ZERO_THRESHOLD:
                singulsF[F] = 1 # sign is reversed from vertices

    def export_as_mesh(self) -> PolyLine:
        """
        Exports the frame field as a set of crosses on each faces, for visualization purposes

        Returns:
            PolyLine: representation of the frame field
        """
        FFMesh = RawMeshData()
        L = mean_edge_length(self.mesh)/5
        n = self.order+1
        for id_edge, (A,B) in enumerate(self.mesh.edges):
            pA,pB = self.mesh.vertices[A], self.mesh.vertices[B]
            bary = (pA+pB)/2
            basis,Y = self.conn.base(id_edge)
            angle = cmath.phase(self.var[id_edge])/self.order
            cmplx = [geom.rotate_around_axis(basis, self.conn.enormals[id_edge], angle + 2*k*pi/self.order) for k in range(self.order)]
            pts = [bary + abs(self.var[id_edge])*L*r for r in cmplx]
            FFMesh.vertices.append(bary)
            FFMesh.vertices += pts
            FFMesh.edges += [(n*id_edge, n*id_edge+k) for k in range(1,n)]
        return PolyLine(FFMesh)


    def optimize(self):
        self._check_init()
    
        self.log("Build laplacian operator")
        lap = operators.laplacian_edges(
            self.mesh,
            self.use_cotan,
            self.conn,
            self.order
        )
        A = operators.area_weight_matrix_edges(self.mesh).tocsc()

        ###### Border ######
        if len(self.feat.feature_edges)>0: # We have a border / feature elements -> linear solve
            # Build fixed and variable indexes
            self.log("Feature element detected (border and/or feature edges)")
            freeInds,fixedInds = [],[]
            for e in self.mesh.id_edges:
                if e in self.feat.feature_edges: # feature_edges is a set
                    fixedInds.append(e)
                else:
                    freeInds.append(e)
              
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
            self.var = inverse_power_method(lap)
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
