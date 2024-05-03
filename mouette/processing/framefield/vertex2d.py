from .base import FrameField
from ...mesh.mesh_attributes import ArrayAttribute, Attribute, Attribute
from ...mesh.datatypes import *
from ... import operators, utils, attributes, processing
from...utils import maths
from ..features import FeatureEdgeDetector
from ..connection import SurfaceConnectionVertices, FlatConnectionVertices
from ... import geometry as geom
from ...optimize import inverse_power_method

from math import pi
import numpy as np
import cmath
import scipy.sparse as sp
from scipy.sparse import linalg
from osqp import OSQP

class _BaseFrameField2DVertices(FrameField):
    """
    Base class for any frame field defined on the vertices of a surface mesh. Is not meant to be instanciated as is.
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
        supporting_mesh : SurfaceMesh, 
        order : int,
        features : bool = True,
        verbose : bool = True,
        **kwargs
    ):
        """
        Parameters:
            supporting_mesh (SurfaceMesh): the mesh (surface) on which to calculate the frame field
            order (int, optional): Order of the frame field (number of branches). Defaults to 4.
            feature_edges (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to True.
        """
        super().__init__("vertices", verbose=verbose)
        self.mesh : SurfaceMesh = supporting_mesh
        self.order : int = order
        
        self.use_cotan : bool = kwargs.get("use_cotan",True)
        self.cad_correction : bool = kwargs.get("cad_correction", False)
        self.features : bool = features or self.cad_correction # whether feature edges are enabled or not
        self.n_smooth : int = kwargs.get("n_smooth", 10)
        self.smooth_attach_weight = kwargs.get("smooth_attach_weight", None) # either None or the provided value
        self.conn = kwargs.get("custom_connection", None) # (A,B) -> direction (angle) of edge (A,B) in local basis of A
        self.feat : FeatureEdgeDetector = kwargs.get("custom_features", None) # either a FeatureEdgeDetector or None

        self.smooth_normals : bool = kwargs.get("smooth_normals", True) # controls which direction to lock on features and boundary (average of features or first one)

        self.vnormals : ArrayAttribute = None # local basis Z vector (normal)

        self.angles : Attribute = None # angles of every triangle corner
        self.defect : Attribute = None # sum of angles around a vertex (/!\ not 'real' defect which is 2*pi - this)
        self.cot : Attribute = None

        self.var = np.zeros(len(self.mesh.vertices), dtype=complex)

    def _initialize_attributes(self):
        self.angles = attributes.corner_angles(self.mesh)
        self.cot = attributes.cotangent(self.mesh) # re-uses angle attribute
        self.vnormals = attributes.vertex_normals(self.mesh, interpolation="angle") # re-uses angle attribute
        # build defects from angles: 
        self.defect = ArrayAttribute(float, len(self.mesh.vertices))
        for iC, C in enumerate(self.mesh.face_corners):
            self.defect[C] = self.defect[C] + self.angles[iC]
        if self.feat is None:
            self.feat = FeatureEdgeDetector(only_border = not self.features, corner_order=self.order, verbose=self.verbose)
            self.feat.run(self.mesh)
        self.conn = self.conn or SurfaceConnectionVertices(self.mesh, self.feat, vnormal=self.vnormals, angles=self.angles)
    
    def _initialize_variables(self):
        """ Init self.var for feature vertices """
        if self.smooth_normals and self.order%2 != 1:
            for e in self.feat.feature_edges:
                A,B = self.mesh.edges[e]
                edge = self.mesh.vertices[B] - self.mesh.vertices[A]
                vx,vy = self.conn.project(edge,B)
                v = complex(vx, vy)
                vpow = (v/abs(v)) ** self.order
                if abs(self.var[B] + vpow)>1e-10:
                    self.var[B] += vpow

                vx,vy = self.conn.project(edge,A)
                v = complex(vx, vy)
                vpow = (v/abs(v)) ** self.order
                if abs(self.var[A] + vpow)>1e-10:
                    self.var[A] += vpow
        else:
            for e in self.feat.feature_edges:
                A,B = self.mesh.edges[e]
                self.var[A] += cmath.rect(1,self.conn.transport(A,B))**self.order
                self.var[B] += cmath.rect(1,self.conn.transport(B,A))**self.order

        for A in self.feat.feature_vertices:
            if abs(self.var[A])>1e-8:
                self.var[A] /= abs(self.var[A])

    def _compute_attach_weight(self, A, fail_value=1e-3):
        # A is area weight matrix
        lap_no_pt = operators.laplacian(self.mesh)
        try:
            eigs = sp.linalg.eigsh(lap_no_pt, k=2, M=A, which="SM", tol=1e-3, maxiter=1000, return_eigenvectors=False)
        except Exception as e:
            try:
                self.log("First estimation of alpha failed: {}".format(e))
                eigs = sp.linalg.eigsh(lap_no_pt+0.1*sp.identity(lap_no_pt.shape[0]), M=A, k=2, which="SM", tol=1e-3, maxiter=1000, return_eigenvectors=False)
            except:
                self.log("Second estimation of alpha failed: taking alpha = ", fail_value)
                return fail_value
        eigs_non_zero = [e for e in eigs if abs(e)>1e-6]
        if len(eigs_non_zero)==0:
            return fail_value
        return abs(min(eigs_non_zero))


    def flag_singularities(self, singul_attr_name:str = "singuls"):
        """Compute singularity data.

        Creates 3 attributes:
            - an attribute "curvature" on faces storing the gaussian curvature (used as a corrective term for computing the values of the singularities)
            - an attribute "<singul_attr_name>" on faces storing the value (+- 1) of singularities (eventually 0 for non singular triangles)
            - an attribute "angles" on edges storing the angle of the edge, given as the difference between the two frames
        """
        self._check_init()
        curvature = attributes.parallel_transport_curvature(self.mesh, self.conn, persistent=False)
        ZERO_THRESHOLD = 1e-2

        edge_rot = dict() # the rotation induced by the frame field on every edge
        if self.mesh.edges.has_attribute("angles"):
            edge_rot_attr = self.mesh.edges.get_attribute("angles")
            edge_rot_attr.clear()
        else:
            edge_rot_attr = self.mesh.edges.create_attribute("angles", float, 1)
        for ie,(A,B) in enumerate(self.mesh.edges):
            fA,fB = self.var[A], self.var[B] # representation complex for A and B
            aA,aB = self.conn.transport(A,B), self.conn.transport(B,A) # local basis orientation for A and B
            uB = maths.roots(fB, self.order)[0]
            abs_angles = [abs(maths.angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA)) for uA in maths.roots(fA, self.order)]
            angles = [maths.angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA) for uA in maths.roots(fA, self.order)]
            i_angle = np.argmin(abs_angles)
            edge_rot[(A,B)] = angles[i_angle]
            edge_rot[(B,A)] = -angles[i_angle]
            edge_rot_attr[ie] = -angles[i_angle]

        if self.mesh.faces.has_attribute(singul_attr_name):
            singuls = self.mesh.faces.get_attribute(singul_attr_name)
        else:
            singuls = self.mesh.faces.create_attribute(singul_attr_name, int)
        for id_face,(A,B,C) in enumerate(self.mesh.faces):
            angle = 0
            for u,v in [(A,B), (B,C), (C,A)]:
                angle += edge_rot[(u,v)]
            angle += curvature[id_face]
            if angle>ZERO_THRESHOLD:
                singuls[id_face] = 1
            elif angle<-ZERO_THRESHOLD:
                singuls[id_face] = -1

    def export_as_mesh(self, repr_vector=False) -> PolyLine:
        """
        Exports the frame field as a mesh for visualization.

        Returns:
            PolyLine: the frame field as a mesh object
        """
        self._check_init()
        FFMesh = PolyLine()
        L = attributes.mean_edge_length(self.mesh)/3
        n = self.order + 1
        for id_vertex, P in enumerate(self.mesh.vertices):
            E,_ = self.conn.base(id_vertex)
            N = self.vnormals[id_vertex]
            if repr_vector:
                # Representation vector only: 
                angle = cmath.phase(self.var[id_vertex])
                r = geom.rotate_around_axis(E, N, angle)
                p = P + L*r
                FFMesh.vertices += [P, p]
                FFMesh.edges.append((2*id_vertex, 2*id_vertex+1))
            else:
                # Representation of the whole frame
                angle = cmath.phase(self.var[id_vertex])/self.order
                cmplx = [geom.rotate_around_axis(E, N, angle + 2*k*pi/self.order) for k in range(self.order)]
                pts = [P + abs(self.var[id_vertex])*L*r for r in cmplx]
                FFMesh.vertices.append(P)
                FFMesh.vertices += pts
                FFMesh.edges += [(n*id_vertex, n*id_vertex+k) for k in range(1,n)]
        return FFMesh

class FrameField2DVertices(_BaseFrameField2DVertices):
    """
    n-RoSy frame field defined on the vertices of a surface mesh
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
        supporting_mesh : SurfaceMesh, 
        order : int = 4, 
        feature_edges : bool = False, 
        verbose=True,
        **kwargs
    ):
        """
        Parameters:
            supporting_mesh (SurfaceMesh): the mesh (surface) on which to calculate the frame field
            order (int, optional): Order of the frame field (number of branches). Defaults to 4.
            feature_edges (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to True.
        """
        super().__init__(supporting_mesh, order, feature_edges, verbose, **kwargs)
                
    def initialize(self):
        self._initialize_attributes()
        self._initialize_variables()
        if self.cad_correction:
            self._modify_parallel_transport()
        self.initialized = True

    def _modify_parallel_transport(self):
        """
        Implementation of 'Frame Fields for CAD models', Desobry et al, 2021.

        This frame field on vertices has a modified parallel transport so avoid placing singularities in very sharp corners of the mesh.
        """
        curvature = attributes.parallel_transport_curvature(self.mesh, self.conn, persistent=False)
        target_w = dict()

        for e in self.feat.feature_edges:
            A,B = self.mesh.edges[e]
            aA,aB = self.conn.transport(A,B), self.conn.transport(B,A) # local basis orientation for A and B
            fA = self.var[A] # representation complex for frame field at A
            fB = self.var[B] # representation complex for frame field at B
            uB = maths.roots(fB, self.order)[0]
            abs_angles = [abs(maths.angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA)) for uA in maths.roots(fA, self.order)]
            angles = [maths.angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA) for uA in maths.roots(fA, self.order)]
            i_angle = np.argmin(abs_angles)
            target_w[e] = angles[i_angle]

        ## build start w for rotation penalty energy
        cstrfaces = attributes.faces_near_border(self.mesh, 5)
        # 1) constraints
        ncstr = len(self.feat.feature_edges)
        nvar = len(self.mesh.edges)
        Cstr = sp.lil_matrix((ncstr, nvar))
        Rhs = np.zeros(ncstr)
        for i,e in enumerate(self.feat.feature_edges):
            Cstr[i,e] = 1
            Rhs[i] = target_w[e]

        # objective
        D1 = sp.lil_matrix((len(self.mesh.faces), nvar))
        b1 = np.zeros(len(self.mesh.faces))
        for iT,T in enumerate(self.mesh.faces):
            A,B,C = T
            for u,v in [(A,B), (B,C), (C,A)]:
                e = self.mesh.connectivity.edge_id(u,v)
                D1[iT, e]= 1 if u<v else -1
            b1[iT] = -curvature[iT]
        
        D2 = sp.lil_matrix((len(cstrfaces), nvar))
        b2 = np.zeros(len(cstrfaces))
        for i,T in enumerate(cstrfaces):
            A,B,C = self.mesh.faces[T]
            for u,v in [(A,B), (B,C), (C,A)]:
                e = self.mesh.connectivity.edge_id(u,v)
                D2[i,e] = 1e3 if (u<v) else -1e3
            b2[i] = -1e3*curvature[T]

        I = sp.identity(len(self.mesh.edges), format="csc")
        b3 = np.zeros(len(self.mesh.edges))

        P = sp.vstack((D1,D2,I))
        b = np.concatenate([b1,b2,b3])
        b = P.transpose().dot(b)
        P = P.transpose().dot(P)
        osqp_instance = OSQP()
        osqp_instance.setup(P, b, A=Cstr.tocsc(), l=Rhs, u=Rhs, verbose=False)
        res = osqp_instance.solve().x
        for e,(A,B) in enumerate(self.mesh.edges):
            self.conn._transport[(A,B)] +=  res[e]/2
            self.conn._transport[(B,A)] -=  res[e]/2

    def optimize(self):
        self._check_init()
        self.log("Build laplacian operator")
        lap = operators.laplacian(self.mesh, cotan=self.use_cotan, connection=self.conn, order=self.order)
        A = operators.area_weight_matrix(self.mesh).tocsc()

        if len(self.feat.feature_vertices)>0: # We have a border / feature elements -> linear solve
            self.log("Feature element detected (border and/or feature edges)")

            # Build fixed/free vertex partition
            fixedInds,freeInds = [],[]
            for v in self.mesh.id_vertices:
                if v in self.feat.feature_vertices:
                    fixedInds.append(v)
                else:
                    freeInds.append(v)
            
            lapI = lap[freeInds,:][:,freeInds]
            lapB = lap[freeInds,:][:,fixedInds]
            AI = A[freeInds, :][:,freeInds]
            # for lapI and lapB, only lines of freeInds are relevant : lines of fixedInds link fixed variables -> useless constraints
            valB = lapB.dot(self.var[fixedInds]) # right hand side

            self.log("Initial solve of linear system")
            res = linalg.spsolve(lapI, -valB) # first system solved without diffusion
            self.var[freeInds] = res

            if self.n_smooth>0:
                self.log(f"Solve linear system {self.n_smooth} times with diffusion")
                alpha = self.smooth_attach_weight or self._compute_attach_weight(A) # Compute attach weight as smallest eigenvalue of the laplacian if not provided as argument
                self.log("Attach weight: {}".format(alpha))
                mat = lapI - alpha * AI.astype(complex)
                for _ in range(self.n_smooth):
                    self.normalize()
                    valI2 = alpha * AI.dot(self.var[freeInds])
                    res = linalg.spsolve(mat, - valB - valI2)
                    self.var[freeInds] = res
            self.normalize()

        else: # No border -> eigensolve
            self.log("No border detected")
            self.log("Initial solve of linear system using an eigensolver")
            self.var = inverse_power_method(lap,A)
            if self.n_smooth>0:
                self.log(f"Solve linear system {self.n_smooth} times with diffusion")
                alpha = self.smooth_attach_weight or self._compute_attach_weight(A) # Compute attach weight as smallest eigenvalue of the laplacian
                self.log("Attach weight: {}".format(alpha))
                mat = lap  - alpha * A.astype(complex)
                for _ in range(self.n_smooth):
                    self.normalize()
                    valI2 = alpha * A.dot(self.var)
                    self.var = linalg.spsolve(mat, - valI2)
            self.normalize()
        self.smoothed = True

class TrivialConnectionVertices(_BaseFrameField2DVertices):
    """
    Implementation of 'Trivial Connections on Discrete Surfaces' by Keenan Crane and Mathieu Desbrun and Peter Schröder, 2010
    
    A frame field on vertices that computes the smoothest possible frame field with prescribed singularity cones at some vertices.
    Does not constraint non-contractible cycles
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
        supporting_mesh : SurfaceMesh, 
        singus_indices:Attribute, 
        order:int = 4, 
        verbose:bool=True,
        **kwargs):
        super().__init__(supporting_mesh, order, feature_edges=False, verbose=verbose, **kwargs)
        self.singus = singus_indices
        self.rotations : np.ndarray = None

    def initialize(self):
        self._initialize_attributes()
        self.var = np.zeros(len(self.mesh.vertices), dtype=complex)
        self.initialized = True
        
    def optimize(self):
        nvar = len(self.mesh.edges)
        ncstr = len(self.mesh.faces)
        curvature = attributes.parallel_transport_curvature(self.mesh, self.conn, persistent=False)
        CstM = sp.lil_matrix((ncstr,nvar))
        CstX = np.zeros(ncstr)
        for F,face in enumerate(self.mesh.faces):
            for u,v in utils.cyclic_pairs(face):
                e = self.mesh.connectivity.edge_id(u,v)
                CstM[F,e] = 1 if (u<v) else -1
            CstX[F] = curvature[F] - 2* pi * self.singus[F] / self.order
        CstM = CstM.tocsc()
        A = sp.eye(nvar, format="csc")
        instance = OSQP()
        instance.setup(P=A, q=None, A=CstM, u=CstX, l=CstX)
        res = instance.solve()
        self.rotations = res.x

        tree = processing.trees.EdgeSpanningTree(self.mesh)()
        for vertex,parent in tree.traverse():
            if parent is None:
                self.var[vertex] = complex(1.,0.)
                continue
            zf = self.var[parent]
            e = self.mesh.connectivity.edge_id(parent,vertex)
            pt = self.conn.transport(vertex,parent) - self.conn.transport(parent,vertex) + pi
            w = self.rotations[e] if vertex<parent else -self.rotations[e]
            zv = zf * cmath.rect(1, 4*(w + pt))
            self.var[vertex] = zv
        self.smoothed = True