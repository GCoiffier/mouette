from .base import FrameField
from ...mesh.mesh_attributes import ArrayAttribute, Attribute, Attribute
from ...mesh.datatypes import *
from ... import operators, utils, attributes, processing
from...utils import maths
from ..features import FeatureEdgeDetector
from ... import geometry as geom
from ...geometry import Vec
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
    def __init__(self, supporting_mesh : SurfaceMesh, order:int= 4, feature_edges:bool=False, verbose=True):
        """
        Parameters:
            supporting_mesh (SurfaceMesh): the mesh (surface) on which to calculate the frame field
            order (int, optional): Order of the frame field (number of branches). Defaults to 4.
            feature_edges (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to True.
        """
        super().__init__(verbose=verbose)
        self.mesh : SurfaceMesh = supporting_mesh
        self.order : int = order

        self.vbaseX : ArrayAttribute = None # local basis X vector (tangent)
        self.vbaseY : ArrayAttribute = None # local basis Y vector (tangent)
        self.vnormals : ArrayAttribute = None # local basis Z vector (normal)

        self.angles : Attribute = None # angles of every triangle corner
        self.defect : Attribute = None # sum of angles around a vertex (/!\ not 'real' defect which is 2*pi - this)
        self.cot : Attribute = None
        self.parallel_transport = dict() # (A,B) -> direction (angle) of edge (A,B) in local basis of A
        self._curvature : Attribute = None # gaussian curvature of each triangle (computed from the parallel transport)

        self.var = np.zeros(len(self.mesh.vertices), dtype=complex)

        self.features : bool = feature_edges # whether feature edges are enabled or not
        self.feat : FeatureEdgeDetector = None

    def _initialize_attributes(self):
        self.angles = attributes.corner_angles(self.mesh)
        self.cot = attributes.cotangent(self.mesh) # re-uses angle attribute
        self.vnormals = attributes.vertex_normals(self.mesh, interpolation="angle") # re-uses angle attribute
        # build defects from angles: 
        self.defect = ArrayAttribute(float, len(self.mesh.vertices))
        for iC, C in enumerate(self.mesh.face_corners):
            self.defect[C] = self.defect[C] + self.angles[iC]

    def _initialize_features(self):
        self.feat = FeatureEdgeDetector(only_border = not self.features, corner_order=self.order, verbose=self.verbose)(self.mesh)

    def _initialize_basis(self):
        """Init self.vbaseX, self.vbaseY and self.edge_angles"""
        self.log("Initialize parallel transport")
        n_vert = len(self.mesh.vertices)
        self.vbaseX = ArrayAttribute(float, n_vert, 3)
        self.vbaseY = ArrayAttribute(float, n_vert, 3)

        for u in range(n_vert):
            # find basis vector -> first edge
            P = Vec(self.mesh.vertices[u])
            N = self.vnormals[u]
            # extract basis edge
            v = self.mesh.connectivity.vertex_to_vertex(u)[0]
            E = self.mesh.vertices[v] - P
            X = Vec.normalized(E - np.dot(E,N)*N) # project on tangent plane
            self.vbaseX[u] = X
            self.vbaseY[u] = geom.cross(N,X)

            # initialize angles of every edge in this basis
            ang = 0.
            vert_u = self.mesh.connectivity.vertex_to_vertex(u)
            if self.mesh.is_vertex_on_border(u):
                # for v in self.mesh.connectivity.vertex_to_vertex(u):
                #     T,iu,_ = self.mesh.half_edges.adj(u,v)
                #     self.parallel_transport[(u,v)] = ang
                #     ang += self.angles[(T,iu)]

                fst, lst = vert_u[0], vert_u[-1]
                pfst, plst = (self.mesh.vertices[x] for x in (fst, lst))
                comp_angle = geom.signed_angle_3pts(plst,P,pfst, N) # complementary angle, ie "exterior" angle between two edges on the boundary
                comp_angle = 2*pi + comp_angle if comp_angle<0 else comp_angle
                
                for v in vert_u:
                    T = self.mesh.half_edges.adj(u,v)[0]
                    self.parallel_transport[(u,v)] = ang * 2 * pi / (self.defect[u] + comp_angle)
                    # self.parallel_transport[(u,v)] = ang * pi / self.defect[u]
                    if T is None : continue
                    c = self.mesh.connectivity.vertex_to_corner_in_face(u,T)
                    ang += self.angles[c]
            else:
                for v in vert_u:
                    T = self.mesh.half_edges.adj(u,v)[0]
                    c = self.mesh.connectivity.vertex_to_corner_in_face(u,T)
                    self.parallel_transport[(u,v)] = ang * 2 * pi / self.defect[u]
                    ang += self.angles[c]

    def _initialize_variables(self, mean_normals=True):
        """Init self.var for feature vertices
        
        mean_normals : whether to initialize the frame field as a mean of adjacent feature edges (True), or following one of the edges (False)
        """
        if mean_normals and self.order%2 != 1:
            for e in self.feat.feature_edges:
                A,B = self.mesh.edges[e]
                edge = self.mesh.vertices[B] - self.mesh.vertices[A]
                vx,vy = np.dot(edge, self.vbaseX[B]), np.dot(edge, self.vbaseY[B])
                v = complex(vx, vy)
                vpow = (v/abs(v)) ** self.order
                if abs(self.var[B] + vpow)>1e-10:
                    self.var[B] += vpow

                vx,vy = np.dot(edge, self.vbaseX[A]), np.dot(edge, self.vbaseY[A])
                v = complex(vx, vy)
                vpow = (v/abs(v)) ** self.order
                if abs(self.var[A] + vpow)>1e-10:
                    self.var[A] += vpow
        else:
            for e in self.feat.feature_edges:
                A,B = self.mesh.edges[e]
                self.var[A] += cmath.rect(1,self.parallel_transport[(A,B)])**self.order
                self.var[B] += cmath.rect(1,self.parallel_transport[(B,A)])**self.order

        for A in self.feat.feature_vertices:
            if abs(self.var[A])>1e-8:
                self.var[A] /= abs(self.var[A])

    def _compute_attach_weight(self, A, fail_value=1e-3):
        # A is area weight matrix
        lap_no_pt = operators.laplacian(self.mesh, parallel_transport=None)
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

    @property
    def curvature(self):
        """Angle defect of each triangles of the mesh, computed from the parallel transport

        Returns:
            Attribute: an attribute on faces
        """
        if self._curvature is None:
            # compute curvature on the mesh
            if self.mesh.faces.has_attribute("curvature"):
                self._curvature = self.mesh.faces.get_attribute("curvature")
            else:
                self._curvature = self.mesh.faces.create_attribute("curvature", float)
                for iF, (A,B,C) in enumerate(self.mesh.faces):
                    v = 1+0j
                    for a,b in [(A,B) ,(B,C), (C,A)]:
                        v *= cmath.rect(1., self.parallel_transport[(b,a)] - self.parallel_transport[(a,b)] - pi)
                    self._curvature[iF] = cmath.phase(v)
        return self._curvature

    def flag_singularities(self, singul_attr_name:str = "singuls"):
        """Compute singularity data.

        Creates 3 attributes:
            - an attribute "curvature" on faces storing the gaussian curvature (used as a corrective term for computing the values of the singularities)
            - an attribute "<singul_attr_name>" on faces storing the value (+- 1) of singularities (eventually 0 for non singular triangles)
            - an attribute "angles" on edges storing the angle of the edge, given as the difference between the two frames
        """
        self._check_init()
        ZERO_THRESHOLD = 1e-2

        edge_rot = dict() # the rotation induced by the frame field on every edge
        if self.mesh.edges.has_attribute("angles"):
            edge_rot_attr = self.mesh.edges.get_attribute("angles")
            edge_rot_attr.clear()
        else:
            edge_rot_attr = self.mesh.edges.create_attribute("angles", float, 1)
        for ie,(A,B) in enumerate(self.mesh.edges):
            fA,fB = self.var[A], self.var[B] # representation complex for A and B
            aA,aB = self.parallel_transport[(A,B)], self.parallel_transport[(B,A)] # local basis orientation for A and B
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
            angle += self.curvature[id_face]
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
            E,N = self.vbaseX[id_vertex], self.vnormals[id_vertex]
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
    def __init__(self, supporting_mesh : SurfaceMesh, order : int = 4, feature_edges : bool = False, verbose=True):
        """
        Parameters:
            supporting_mesh (SurfaceMesh): the mesh (surface) on which to calculate the frame field
            order (int, optional): Order of the frame field (number of branches). Defaults to 4.
            feature_edges (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to True.
        """
        super().__init__(supporting_mesh, order, feature_edges, verbose)
                
    def initialize(self):
        self._initialize_attributes()
        self._initialize_features() # /!\ before initialize basis
        self._initialize_basis()
        self._initialize_variables()
        self.initialized = True

    def optimize(self, n_renorm=10):
        self._check_init()
        self.log("Build laplacian operator")
        lap = operators.laplacian(self.mesh, parallel_transport=self.parallel_transport, order=self.order)
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

            if n_renorm>0:
                self.log(f"Solve linear system {n_renorm} times with diffusion")
                alpha = self._compute_attach_weight(A) # Compute attach weight as smallest eigenvalue of the laplacian
                self.log("Attach weight: {}".format(alpha))
                mat = lapI - alpha * AI.astype(complex)
                for _ in range(n_renorm):
                    self.normalize()
                    valI2 = alpha * AI.dot(self.var[freeInds])
                    res = linalg.spsolve(mat, - valB - valI2)
                    self.var[freeInds] = res
            self.normalize()

        else: # No border -> eigensolve
            self.log("No border detected")
            self.log("Initial solve of linear system using an eigensolver")
            A = A.astype(complex)
            self.var = inverse_power_method(lap,A)
            if n_renorm>0:
                self.log(f"Solve linear system {n_renorm} times with diffusion")
                alpha = self._compute_attach_weight(A) # Compute attach weight as smallest eigenvalue of the laplacian
                self.log("Attach weight: {}".format(alpha))
                mat = lap  - alpha * A
                for _ in range(n_renorm):
                    self.normalize()
                    valI2 = alpha * A.dot(self.var)
                    self.var = linalg.spsolve(mat, - valI2)
            self.normalize()

class CadFF2DVertices(FrameField2DVertices):
    """
    Implementation of 'Frame Fields for CAD models', Desobry et al, 2021.

    This frame field on vertices has a modified parallel transport so avoid placing singularities in very sharp corners of the mesh.
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, supporting_mesh : SurfaceMesh, order:int = 4, verbose=True):
        """
        Parameters:
            supporting_mesh (SurfaceMesh): the mesh (surface) on which to calculate the frame field
            order (int, optional): Order of the frame field (number of branches). Defaults to 4.
            verbose (bool, optional): _description_. Defaults to True.

        Note:
            Feature edges are automatically set to True and cannot be disabled with CadFF
        """
        super().__init__(supporting_mesh, order, feature_edges=True, verbose=verbose)
        self.target_w : dict = None # the modified parallel transport

    def _initialize_target_w(self):
        self.target_w = dict() 
        for e in self.feat.feature_edges:
            A,B = self.mesh.edges[e]
            aA,aB = self.parallel_transport[(A,B)], self.parallel_transport[(B,A)] # local basis orientation for A and B
            fA = self.var[A] # representation complex for frame field at A
            fB = self.var[B] # representation complex for frame field at B
            uB = maths.roots(fB, self.order)[0]
            abs_angles = [abs(maths.angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA)) for uA in maths.roots(fA, self.order)]
            angles = [maths.angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA) for uA in maths.roots(fA, self.order)]
            i_angle = np.argmin(abs_angles)
            self.target_w[e] = angles[i_angle]

        ## build start w for rotation penalty energy
        cstrfaces = attributes.faces_near_border(self.mesh, 4)
        # 1) constraints
        ncstr = len(self.feat.feature_edges)
        nvar = len(self.mesh.edges)
        Cstr = sp.lil_matrix((ncstr, nvar))
        Rhs = np.zeros(ncstr)
        for i,e in enumerate(self.feat.feature_edges):
            Cstr[i,e] = 1
            Rhs[i] = self.target_w[e]

        # objective
        D1 = sp.lil_matrix((len(self.mesh.faces), nvar))
        b1 = np.zeros(len(self.mesh.faces))
        for iT,T in enumerate(self.mesh.faces):
            A,B,C = T
            for u,v in [(A,B), (B,C), (C,A)]:
                e = self.mesh.connectivity.edge_id(u,v)
                D1[iT, e]= 1 if u<v else -1
            b1[iT] = -self.curvature[iT]
        
        D2 = sp.lil_matrix((len(cstrfaces), nvar))
        b2 = np.zeros(len(cstrfaces))
        for i,f in enumerate(cstrfaces):
            A,B,C = self.mesh.faces[f]
            for u,v in [(A,B), (B,C), (C,A)]:
                e = self.mesh.connectivity.edge_id(u,v)
                D2[i,e] = 1e3 if (u<v) else -1e3
            b2[i] = -1e3*self.curvature[f]

        I = sp.identity(len(self.mesh.edges), format="csc")
        b3 = np.zeros(len(self.mesh.edges))

        P = sp.vstack((D1,D2,I))
        b = np.concatenate([b1,b2,b3])
        b = P.transpose().dot(b)
        P = P.transpose().dot(P)
        osqp_instance = OSQP()
        osqp_instance.setup(P, b, A=Cstr.tocsc(), l=Rhs, u=Rhs, verbose=False)
        res = osqp_instance.solve().x
        for e in self.mesh.id_edges:
            self.target_w[e] = res[e]

    def initialize(self):
        self._initialize_attributes()
        self._initialize_features() # /!\ before initialize basis
        self._initialize_basis()
        self._initialize_variables()
        self._initialize_target_w()
        self.initialized = True

    def _cotan_laplacian_parallel_transport(self)-> sp.lil_matrix :
        """For CadFF, the cotan laplacian operator is modified by the target angles computed at the initialization

        Returns:
            scipy.sparse.lil_matrix: cotan laplacian as sparse matrix
        """
        n = len(self.mesh.vertices)
        mat = sp.lil_matrix((n,n), dtype=complex)
        for it, (p,q,r) in enumerate(self.mesh.faces):
            a,b,c = (self.cot[(it, k)]/2 for k in range(3))
            for (i, j, v) in [(p, q, c), (q, r, a), (r, p, b)]:
                ai, aj = self.parallel_transport[(i,j)], self.parallel_transport[(j,i)]
                e = self.mesh.connectivity.edge_id(i,j)
                w = self.target_w[e] if i<j else -self.target_w[e]
                mat[i,i] -= v
                mat[j,j] -= v
                mat[i,j] += v * cmath.rect(1., self.order*(ai - aj + pi + w))
                mat[j,i] += v * cmath.rect(1., self.order*(aj - ai + pi - w))
        return mat

    def _adj_laplacian_parallel_transport(self) -> sp.lil_matrix :
        """
        Returns:
            scipy.sparse.lil_matrix: laplacian with classic +- 1 weights 
        """
        n = len(self.mesh.vertices)
        mat = sp.lil_matrix((n,n), dtype=complex)
        for (p,q,r) in self.mesh.faces:
            for (i, j) in [(p, q), (q, r), (r, p)]:
                ai, aj = self.parallel_transport[(i,j)], self.parallel_transport[(j,i)]
                e = self.mesh.connectivity.edge_id(i,j)
                w = self.target_w[e] if i<j else -self.target_w[e]
                mat[i,i] -= 1
                mat[j,j] -= 1
                mat[i,j] += cmath.rect(1., self.order*(ai - aj + pi + w))
                mat[j,i] += cmath.rect(1., self.order*(aj - ai + pi - w))
        return mat

class TrivialConnectionVertices(_BaseFrameField2DVertices):
    """
    Implementation of 'Trivial Connections on Discrete Surfaces' by Keenan Crane and Mathieu Desbrun and Peter Schröder, 2010
    
    A frame field on vertices that computes the smoothest possible frame field with prescribed singularity cones at some vertices.
    Does not constraint non-contractible cycles
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, supporting_mesh : SurfaceMesh, singus_indices:Attribute, order:int = 4, verbose:bool=True):
        super().__init__(supporting_mesh, order, feature_edges=False, verbose=verbose)
        self.singus = singus_indices
        self.rotations : np.ndarray = None

    def initialize(self):
        self._initialize_attributes()
        self._initialize_basis()
        self.var = np.zeros(len(self.mesh.vertices), dtype=complex)
        self.initialized = True
        
    def optimize(self):
        nvar = len(self.mesh.edges)
        ncstr = len(self.mesh.faces)
        CstM = sp.lil_matrix((ncstr,nvar))
        CstX = np.zeros(ncstr)
        for F,face in enumerate(self.mesh.faces):
            for u,v in utils.cyclic_pairs(face):
                e = self.mesh.connectivity.edge_id(u,v)
                CstM[F,e] = 1 if (u<v) else -1
            CstX[F] = self.curvature[F] - 2* pi * self.singus[F] / self.order
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
            pt = self.parallel_transport[(vertex,parent)] - self.parallel_transport[(parent,vertex)] + pi
            w = self.rotations[e] if vertex<parent else -self.rotations[e]
            zv = zf * cmath.rect(1, 4*(w + pt))
            self.var[vertex] = zv