from .base import FrameField
from ...mesh.mesh_attributes import ArrayAttribute, Attribute, Attribute
from ...mesh.datatypes import *
from ...import attributes
from ..features import FeatureEdgeDetector
from ... import geometry as geom
from ...geometry import Vec
from ...utils.maths import *
from ... import operators

from math import pi
import numpy as np
import cmath
import scipy.sparse as sp
from scipy.sparse import linalg
from osqp import OSQP

class _BaseFrameField2DVertices(FrameField): 

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, supporting_mesh : SurfaceMesh, order:int= 4, feature_edges:bool=False, verbose=True):
        super().__init__(verbose=verbose)
        self.mesh : SurfaceMesh = supporting_mesh
        self.order : int = order

        self.vbaseX : Attribute = None # local basis X vector (tangent)
        self.vbaseY : Attribute = None # local basis Y vector (tangent)
        self.vnormals : Attribute = None # local basis Z vector (normal)

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
        self.feat = FeatureEdgeDetector(only_border = not self.features, verbose=self.verbose)(self.mesh)

    def _initialize_basis(self):
        """Init self.vbaseX, self.vbaseY and self.edge_angles"""
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
            if self.mesh.is_vertex_on_border(u):
                # for v in self.mesh.connectivity.vertex_to_vertex(u):
                #     T,iu,_ = self.mesh.half_edges.adj(u,v)
                #     self.parallel_transport[(u,v)] = ang
                #     ang += self.angles[(T,iu)]

                fst, lst = self.mesh.connectivity.vertex_to_vertex(u)[0], self.mesh.connectivity.vertex_to_vertex(u)[-1]
                pfst, plst = (self.mesh.vertices[x] for x in (fst, lst))
                comp_angle = geom.signed_angle_3pts(plst,P,pfst, N) # complementary angle, ie "exterior" angle between two edges on the boundary
                comp_angle = 2*pi + comp_angle if comp_angle<0 else comp_angle
                
                for v in self.mesh.connectivity.vertex_to_vertex(u):
                    T = self.mesh.half_edges.adj(u,v)[0]
                    self.parallel_transport[(u,v)] = ang * 2 * pi / (self.defect[u] + comp_angle)
                    #self.parallel_transport[(u,v)] = ang * pi / self.defect[u]
                    if T is None : continue
                    c = self.mesh.connectivity.vertex_to_corner_in_face(u,T)
                    ang += self.angles[c]
            else:
                for v in self.mesh.connectivity.vertex_to_vertex(u):
                    T = self.mesh.half_edges.adj(u,v)[0]
                    c = self.mesh.connectivity.vertex_to_corner_in_face(u,T)
                    self.parallel_transport[(u,v)] = ang * 2 * pi / self.defect[u]
                    ang += self.angles[c]

    def _initialize_variables(self, mean_normals=True):
        """Init self.var for feature vertices
        
        mean_normals : whether to initialize the frame field as a mean of adjacent feature edges (True), or following one of the edges (False)
        """
        if mean_normals:
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
            self.var[A] /= abs(self.var[A])

    def _compute_attach_weight(self, A, fail_value=1e-3):
        return fail_value
        # A is area weight matrix
        lap_nopt = operators.laplacian(self.mesh, parallel_transport=None)
        try:
            eigs = sp.linalg.eigsh(lap_nopt, k=2, M=A, which="SM", tol=1e-3, maxiter=500, return_eigenvectors=False)
        except Exception as e:
            try:
                self.log("Estimation of alpha failed: {}".format(e))
                eigs = sp.linalg.eigsh(lap_nopt-0.01*sp.identity(lap_nopt.shape[0]), k=2, which="SM", tol=1e-3, maxiter=500, return_eigenvectors=False)
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

    def flag_singularities(self):
        """Compute singularity data.

        Creates 3 attributes:
            - an attribute "curvature" on faces storing the gaussian curvature (used as a corrective term for computing the values of the singularities)
            - an attribute "singuls" on faces storing the value (+- 1) of singularities (eventually 0 for non singular triangles)
            - an attribute "angles" on edges storing the angle of the edge, given as the difference between the two frames
        """
        self._check_init()
        ZERO_THRESHOLD = 1e-4

        edge_rot = dict() # the rotation induced by the frame field on every edge
        if self.mesh.edges.has_attribute("angles"):
            edge_rot_attr = self.mesh.edges.get_attribute("angles")
            edge_rot_attr.clear()
        else:
            edge_rot_attr = self.mesh.edges.create_attribute("angles", float, 1)
        for ie,(A,B) in enumerate(self.mesh.edges):
            fA,fB = self.var[A], self.var[B] # representation complex for A and B
            aA,aB = self.parallel_transport[(A,B)], self.parallel_transport[(B,A)] # local basis orientation for A and B
            uB = roots(fB, self.order)[0]
            abs_angles = [abs( angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA)) for uA in roots(fA, self.order)]
            angles = [angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA) for uA in roots(fA, self.order)]
            i_angle = np.argmin(abs_angles)
            edge_rot[(A,B)] = angles[i_angle]
            edge_rot[(B,A)] = -angles[i_angle]
            edge_rot_attr[ie] = -angles[i_angle]

        if self.mesh.faces.has_attribute("singuls"):
            singuls = self.mesh.faces.get_attribute("singuls")
        else:
            singuls = self.mesh.faces.create_attribute("singuls", int)
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
    
    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, supporting_mesh : SurfaceMesh, order : int = 4, feature_edges : bool = False, verbose=True):
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
            # Compute attach weight as smallest eigenvalue of the laplacian
            self.log("Compute attach weight")
            alpha = self._compute_attach_weight(A)
            self.log("Attach weight: {}".format(alpha))

            A = A.astype(complex)

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
                mat = lapI - alpha * AI
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
            try:
                egv, U = sp.linalg.eigsh(lap, k=2, M=A, which="SM", tol=1e-3, maxiter=1000)
            except linalg.ArpackNoConvergence as e:
                self.log("Initial eigensolve failed :", e)
                self.log("Retry using connectivity laplacian")
                lap = operators.laplacian(self.mesh, cotan=False, parallel_transport=self.parallel_transport, order=self.order)
                egv, U = sp.linalg.eigsh(lap-0.1*sp.identity(lap.shape[0]), k=2, M=A, which="SM", tol=1e-3)
            iu = np.argmin(abs(egv))
            alpha = abs(egv[iu])
            self.log("Eigenvalues:", egv)
            self.var = U[:,iu]

            if n_renorm>0:
                self.log(f"Solve linear system {n_renorm} times with diffusion")
                mat = lap  - alpha * A
                for _ in range(n_renorm):
                    self.normalize()
                    valI2 = alpha * A.dot(self.var)
                    self.var = linalg.spsolve(mat, - valI2)
            self.normalize()

class CadFF2DVertices(FrameField2DVertices):

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, supporting_mesh : SurfaceMesh, order:int = 4, verbose=True):
        super().__init__(supporting_mesh, order, feature_edges=True, verbose=verbose)
        self.target_w : dict = None

    def _initialize_target_w(self):
        self.target_w = dict()
        for e in self.feat.feature_edges:
            A,B = self.mesh.edges[e]
            aA,aB = self.parallel_transport[(A,B)], self.parallel_transport[(B,A)] # local basis orientation for A and B
            fA = self.var[A] # representation complex for frame field at A
            fB = self.var[B] # representation complex for frame field at B
            uB = roots(fB, self.order)[0]
            abs_angles = [abs(angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA)) for uA in roots(fA, self.order)]
            angles = [angle_diff( cmath.phase(uB) - aB - pi, cmath.phase(uA)-aA) for uA in roots(fA, self.order)]
            i_angle = np.argmin(abs_angles)
            self.target_w[e] = angles[i_angle]

        ## build start w for rotation penalty energy
        cstrfaces = attributes.faces_near_border(self.mesh, 3)
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

    def _cotan_laplacian_parallel_transport(self):
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
                mat[i,j] += v * cmath.rect(1., self.order*(ai - aj + w))
                mat[j,i] += v * cmath.rect(1., self.order*(aj - ai - w))
        return mat

    def _adj_laplacian_parallel_transport(self):
        """Laplacian with classic +- 1 weights 
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
                mat[i,j] += cmath.rect(1., self.order*(ai - aj + w))
                mat[j,i] += cmath.rect(1., self.order*(aj - ai - w))
        return mat

class CurvatureVertices(_BaseFrameField2DVertices):

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, feature_edges:bool = False, verbose=True):
        super().__init__(mesh, 
            4, # order is always 4 
            feature_edges,
            verbose
        )
        self.face_areas : Attribute = None
        self.curv_mat_vert : np.ndarray = None # shape (|V|,3,3)

    def run(self, n_smooth=0):
        self.log("Compute curvature matrices")
        self.initialize()
        self.log("Optimize")
        self.optimize(n_smooth)
        self.log("Done.")
        return self

    def _initialize_attributes(self):
        super()._initialize_attributes()
        self.face_areas = attributes.face_area(self.mesh)

    def _initialize_curv_matrices(self):
        curv_mat_edges = attributes.curvature_matrices(self.mesh)
        self.curv_mat_vert = np.zeros((len(self.mesh.vertices),3,3))
        for v in self.mesh.id_vertices:
            #total_area = 0
            for e in self.mesh.connectivity.vertex_to_edge(v):
                v2 = self.mesh.connectivity.other_edge_end(e,v)
                area_ab = sum([self.face_areas[_T] for _T in self.mesh.half_edges.edge_to_triangles(v,v2) if _T is not None])
                #total_area += area_ab
                self.curv_mat_vert[v,:,:] += area_ab * curv_mat_edges[e,:,:]
            #self.curv_mat_vert[v,:,:] /= total_area

    def initialize(self):
        self._initialize_attributes()
        self._initialize_features() # /!\ before initialize basis
        self._initialize_basis()
        self._initialize_curv_matrices()
        self._initialize_variables()
        self.initialized = True

    def optimize(self, n_smooth: int = 1, normalize:bool = True):
        for v in self.mesh.id_vertices:
            if v in self.feat.feature_vertices: continue # do not override frames on boundary
            
            U,S,V = np.linalg.svd(self.curv_mat_vert[v,:,:], hermitian=True)
            # three vectors -> normal and two principal components
            # eigenvalue of normal is 0
            # PC are orthogonal (eigenvects of symmetric matrix) -> we rely on the eigenvect of greatest eigenvalue and take orthog direction

            X,Y = self.vbaseX[v], self.vbaseY[v]
            if S[0]<1e-8 : # zero matrix -> no curvature information
                self.var[v] = 0 + 0j
            else:
                # representation vector
                eig = V[0,:]
                c = complex(X.dot(eig), Y.dot(eig))
                self.var[v] = (c/abs(c))**4

        if n_smooth > 0:
            # diffuse the curvature results to get a smoother results (especially where curvature was not defined)
            lap = operators.laplacian(self.mesh, parallel_transport=self.parallel_transport, order=4)
            A = operators.area_weight_matrix(self.mesh)
            alpha = 0.1*attributes.mean_edge_length(self.mesh)
            self.log("Attach weight:", alpha)

            if len(self.feat.feature_vertices)>0:
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
                mat = lapI - alpha * AI
                for _ in range(n_smooth):
                    valI2 = alpha * AI.dot(self.var[freeInds])
                    res = linalg.spsolve(mat, - valB - valI2)
                    self.var[freeInds] = res
                    if normalize: self.normalize()
            else:
                mat = lap  - alpha * A
                for _ in range(n_smooth):
                    valI2 = alpha * A.dot(self.var)
                    self.var = linalg.spsolve(mat, - valI2)
                    if normalize: self.normalize()
