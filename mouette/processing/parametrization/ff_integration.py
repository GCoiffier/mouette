from .base import BaseParametrization

from ...mesh.datatypes import SurfaceMesh,PolyLine
from ...mesh.mesh_attributes import Attribute
from ...mesh.mesh import copy

from ..cutting import SingularityCutter
from ..connection import SurfaceConnectionFaces
from ..features import FeatureEdgeDetector
from ..framefield import FrameField
from ..trees import FaceSpanningTree
from ...import attributes

from ... import geometry as geom
from ...geometry import Vec
from ... import utils

import numpy as np
import cmath
import scipy.sparse as sp
from osqp import OSQP

class FrameFieldIntegration(BaseParametrization):
    """
    Integration of a smooth vector field into a seamless parametrization

    References:
        - [1] _Mixed-integer quadrangulation_, Bommes D, Zimmer H. and Kobbelt L., ACM Transaction on Graphics, 2009
        
        - [2] _QuadCover - Surface Parameterization using Branched Coverings_, Kälberer F., Nieser M. and Polthier K., Computer Graphics Forum, 2007
    """

    def __init__(self, framefield: FrameField, verbose:bool=True, **kwargs):
        """
        Initializer.

        Args:
            framefield (FrameField): the frame field object to integrate.
            singularities (Attribute): float Attribute on vertices. Gives the target angle defects of vertices
            use_cotan (bool, optional): whether to use adjacency or cotangents as weights in the laplacian matrix. Defaults to True. 
            verbose (bool, optional): verbose mode. Defaults to True.
            debug (bool, optional) : debug mode. Generates additionnal output. Defaults to False.
        """
        super().__init__("Frame Field Integration", framefield.mesh, verbose, **kwargs)
        self._use_cotan = kwargs.get("use_cotan", True)
        self._debug = kwargs.get("debug", False)
        self.save_on_corners = True

        self.frame_field = framefield
        assert self.frame_field.element == "faces"
        assert self.frame_field.order == 4

        self._cutter : SingularityCutter
        self._matching : np.ndarray = None # matching attribute on faces
        self._jumps : Attribute = None # jump attribute on edges
        self._singus : Attribute = None # position and indices of cones (on vertices)
        self._tree : FaceSpanningTree = None # traversal tree

    @property
    def _conn(self) -> SurfaceConnectionFaces:
        # Shortcut getter
        return self.frame_field.conn

    @property
    def _feat(self) -> FeatureEdgeDetector:
        # Shortcut getter
        return self.frame_field.feat

    def get_ff_dir(self, T:int) -> complex:
        """
        First frame field direction after the matching was computed

        Args:
            T (int): face index

        Returns:
            complex : the matched frame branch  
        """
        if self._matching is None: return None
        return utils.maths.roots(self.frame_field[T],4)[self._matching[T]]

    def _compute_matching(self):
        self._tree = FaceSpanningTree(self.mesh, forbidden_edges=self._cutter.cut_edges)()
        self._matching = np.zeros(len(self.mesh.faces),dtype=np.int32)
        for face, parent in self._tree.traverse():
            if parent is None: continue # brushing of root is 0 by convention
            rp = self.get_ff_dir(parent)
            angles = [utils.maths.angle_diff( cmath.phase(rp), cmath.phase(rf) - self._conn.transport(face,parent)) for rf in utils.maths.roots(self.frame_field[face], 4)]
            abs_angles = [abs(_a) for _a in angles]
            self._matching[face] = np.argmin(abs_angles)

    def _compute_jumps(self):
        self.jumps = self.mesh.edges.create_attribute("jump", int)
        for e in self._cutter.cut_edges:
            T1,T2 = self.mesh.connectivity.edge_to_faces(*self.mesh.edges[e])
            if T1 is None or T2 is None : continue # boundary edge -> no jump
            ff1, ff2 = self.get_ff_dir(T1), self.get_ff_dir(T2)
            jump = utils.maths.angle_diff( cmath.phase(ff1), cmath.phase(ff2) - self._conn.transport(T2,T1))
            self.jumps[e] = round(2*jump/np.pi)%4 # integer multiple of pi/2

    def _integrate(self):
        nvar = 4*len(self.mesh.faces) # one 2x2 Jacobian per face
        
        ## Compute objective
        q = np.zeros(nvar, dtype=float)
        for T in self.mesh.id_faces:
            # We want the jacobian ( a  b ) to match the rotation matrix of the frame ( Re(zT)  Im(zT) )
            #                      ( c  d )                                           ( -Im(zT) Re(zT) )
            zT = self.get_ff_dir(T)
            q[4*T:4*T+4] = [zT.real, zT.imag, -zT.imag, zT.real]
        
        ## Compute constraints
        rows,cols,vals = [], [], []
        irow = 0
        
        ## Build jump constraint matrix
        for e in self.mesh.id_edges:
            je = self.jumps[e]
            A,B = self.mesh.edges[e]
            T1,T2 = self.mesh.connectivity.edge_to_faces(A,B)
            if T1 is None or T2 is None : continue # boundary edge
            E = self.mesh.vertices[B] - self.mesh.vertices[A]
            E1 = self._conn.project(E,T1)
            E2 = self._conn.project(E,T2)
            
            # J1 E1 = R^{je} J2 E2
            rows += [irow, irow,   irow+1, irow+1]
            cols += [4*T1, 4*T1+1, 4*T1+2, 4*T1+3]
            vals += [E1.x, E1.y, E1.x, E1.y]

            rows += [irow, irow,   irow+1, irow+1]
            if je==0:
                cols += [4*T2, 4*T2+1, 4*T2+2, 4*T2+3]
                vals += [-E2.x, -E2.y, -E2.x, -E2.y]
            elif je==1:
                cols += [4*T2+2, 4*T2+3, 4*T2, 4*T2+1]
                vals += [-E2.x, -E2.y, E2.x, E2.y]
            elif je==2:
                cols += [4*T2, 4*T2+1, 4*T2+2, 4*T2+3]
                vals += [E2.x, E2.y, E2.x, E2.y]
            elif je==3:
                cols += [4*T2+2, 4*T2+3, 4*T2, 4*T2+1]
                vals += [E2.x, E2.y, -E2.x, -E2.y]
            irow += 2
        
        ## Feature edge alignment
        for e in self._feat.feature_edges:
            A,B = self.mesh.edges[e]
            E = self.mesh.vertices[B] - self.mesh.vertices[A] 
            for T in self.mesh.connectivity.edge_to_faces(A,B):
                if T is None: continue
                ET = Vec.normalized(self._conn.project(E,T))
                zT = self.get_ff_dir(T)
                zT = Vec(zT.real, zT.imag)
                # determine the orientation of the feature (horizontal or vertical) in parameter space depending on the matching
                dtp = round(abs(ET.dot(zT)))
                rows += [irow, irow]
                irow += 1
                if dtp==0: # zero => horizontal
                    cols += [4*T,4*T+1]
                    vals += [ET.x, ET.y]
                elif dtp==1:
                    cols += [4*T+2,4*T+3]
                    vals += [ET.x, ET.y]

        CstMat = sp.csc_matrix((vals, (rows,cols)), shape=(irow,nvar))
        CstRHS = np.zeros(irow)

        ## Solve system
        instance = OSQP()
        instance.setup(sp.eye(nvar,format="csc"), -q, CstMat, CstRHS, CstRHS, verbose=self.verbose)
        result = instance.solve()
        return result.x

    def _export_jacobian_dets(self,Jac):
        det_attr = self.mesh.faces.create_attribute("det", float)
        for T in self.mesh.id_faces:
            J = Jac[4*T:4*T+4].reshape((2,2))
            detJ = np.linalg.det(J)
            det_attr[T] = detJ

    def _build_UVs(self,Jac):
        self.uvs = self.mesh.face_corners.create_attribute("uv_coords", float, 2, dense=True)

        def build_triangle(iT):
            pA,pB,pC = (self.mesh.vertices[_u] for _u in self.mesh.faces[iT])
            X,Y = self._conn.base(iT)
            qA = Vec.zeros(2)
            qB = Vec(X.dot(pB-pA), Y.dot(pB-pA))
            qC = Vec(X.dot(pC-pA), Y.dot(pC-pA))
            J = Jac[4*iT:4*iT+4].reshape((2,2))
            qB = Vec(J.dot(qB))
            qC = Vec(J.dot(qC))
            return qA,qB,qC
        
        def align_triangles(pA,pB, qA, qB, qC):
            target = Vec(pB - pA)

            # 1) translation to align point A
            translation = pA - qA
            qA += translation
            qB += translation
            qC += translation

            # 2) rotation around point A to align the point B
            q = Vec(qB - qA)
            rot = np.arctan2(target.y, target.x) - np.arctan2(q.y,q.x)
            rc, rs = np.cos(rot), np.sin(rot)

            q = Vec(qB - qA)
            qB.x, qB.y = qA.x + rc*q.x - rs*q.y , qA.y + rs*q.x + rc*q.y
            q = Vec(qC-qA)
            qC.x, qC.y = qA.x + rc*q.x - rs*q.y , qA.y + rs*q.x + rc*q.y

            return qA,qB,qC

        for T,parent in self._tree.traverse():
            if parent is None:
                # build the root
                pA,pB,pC = build_triangle(T)
                self.uvs[3*T+0] = pA
                self.uvs[3*T+1] = pB
                self.uvs[3*T+2] = pC
                continue

            A,B = self.mesh.connectivity.common_edge(T,parent) # common vertices
            iA,iB = (self.mesh.connectivity.in_face_index(parent,_u) for _u in (A,B))
            jA,jB = (self.mesh.connectivity.in_face_index(T,_u) for _u in (A,B))
            jC = 3-jA-jB
            
            pA,pB = self.uvs[3*parent+iA], self.uvs[3*parent+iB]
            q = build_triangle(T)
            qA,qB,qC = q[jA], q[jB], q[jC]
            qA,qB,qC = align_triangles(pA,pB,qA,qB,qC)

            for (j,q) in [(jA,qA), (jB,qB), (jC,qC)]:
                self.uvs[3*T+j] = q

    def run(self) :
        self.log("Smooth frame field if needed")
        self.frame_field.run()
        self.frame_field.flag_singularities() # computes singularities
        self._singus = self.mesh.vertices.get_attribute("singuls")

        self.log("Perform cuts between cones")
        ## Define a cut mesh along the shortest path towards the boundary
        self._cutter = SingularityCutter(self.mesh, self._singus, features=self._feat)
        self._cutter.run()

        self.log("Compute frame field matching")
        self._compute_matching()
        self._compute_jumps()
       
        self.log("Solve gradient least-square system")
        Jac = self._integrate()
        self._export_jacobian_dets(Jac)

        self.log("Build UVs")
        self._build_UVs(Jac)

        self.log("Rescale UVs and output result")
        ## Rescale UV coordinates to fit in [0,1]²
        xmin,xmax,ymin,ymax = float("inf"), -float("inf"), float("inf"), -float("inf")
        for c in self.mesh.id_corners:
            cx,cy = self.uvs[c]
            xmin = min(xmin, cx)
            xmax = max(xmax, cx)
            ymin = min(ymin, cy)
            ymax = max(ymax, cy)
        scale_x = xmax-xmin
        scale_y = ymax-ymin
        scale = min(scale_x, scale_y)

        ## Write final result in attribute
        for c in self.mesh.id_corners:
            self.uvs[c] = self.uvs[c]/scale

    def export_frame_field_as_mesh(self) -> PolyLine:
        if self._matching is None:
            return self.frame_field.export_as_mesh()

        FFmesh = PolyLine()
        L = attributes.mean_edge_length(self.mesh)/3
        match_attr = FFmesh.edges.create_attribute("matching", int, dense=True)
        for id_face, face in enumerate(self.mesh.faces):
            basis,Y = self._conn.base(id_face)
            normal = geom.cross(basis,Y)
            pA,pB,pC = (self.mesh.vertices[_v] for _v in face)
            angle = cmath.phase(self.get_ff_dir(id_face))
            bary = (pA+pB+pC)/3 # reference point for display
            r1,r2 = (geom.rotate_around_axis(basis, normal, angle + k*np.pi/2) for k in range(2))
            p1,p2= (bary + L*r for r in (r1,r2))
            FFmesh.vertices += [bary, p1, p2]
            FFmesh.edges += [(3*id_face, 3*id_face+k) for k in range(1,3)]
            match_attr[2*id_face]   = 0
            match_attr[2*id_face+1] = 1
        return FFmesh  

    @property
    def cut_mesh(self) -> SurfaceMesh:
        return self._cutter.output_mesh

    @property
    def cut_graph(self) -> PolyLine:
        return self._cutter.cut_graph

    @property
    def flat_mesh(self) -> SurfaceMesh:
        """
        A flat representation of the mesh where uv-coordinates are copied to xy.

        Returns:
            SurfaceMesh: the flat mesh
        """
        if self.uvs is None : return None
        if self._flat_mesh is None:
            # build the flat mesh : vertex coordinates are uv of original mesh
            self._flat_mesh = copy(self.cut_mesh)
            for T in self.cut_mesh.id_faces:
                for i,v in enumerate(self.cut_mesh.faces[T]):
                    self._flat_mesh.vertices[v] = Vec(self.uvs[3*T+i][0], self.uvs[3*T+i][1], 0.)
        return self._flat_mesh