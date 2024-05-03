from .base import FrameField

from ...mesh.datatypes import *
from ...mesh.mesh_data import RawMeshData
from ...mesh.mesh_attributes import Attribute

from ... import attributes
from ..features import FeatureEdgeDetector

from ... import geometry as geom
from ...geometry import Vec, SphericalHarmonics
from ...geometry.rotations import match_rotation
from ...geometry import transform

from ...operators.laplacian_op import *
from ...procedural import axis_aligned_cube

import numpy as np
import scipy
import scipy.sparse as sp
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from math import atan2
from osqp import OSQP
from ...utils.osqp_lin_solve import get_osqp_lin_solver

class FrameField3DVertices(FrameField): 

    @allowed_mesh_types(VolumeMesh)
    def __init__(self, 
        supporting_mesh : VolumeMesh, 
        feature_edges : bool = True, 
        verbose=True,
        **kwargs):
        super().__init__("vertices", "FrameField3D", verbose)
        self.mesh : VolumeMesh = supporting_mesh
        supporting_mesh.enable_boundary_connectivity()

        self.boundary_mesh : SurfaceMesh = None
        self.vertex_normals : Attribute = None
        self.n_smooth = kwargs.get("n_smooth", 3)
        self.smooth_attach_weight = kwargs.get("smooth_attach_weight", None)

        self.frames : list = None # raw representation of frames as scipy.Rotation objects
        self.var = np.concatenate([SphericalHarmonics.EYE]*len(self.mesh.vertices)) # representation vectors in spherical harmonics basis

        self._omegas : dict = None # rotation on edges
        self._singuls_tri : Attribute = None # value of defect on every triangle

        self.features : bool = feature_edges # whether feature edges are enabled or not
        self.feat : FeatureEdgeDetector = kwargs.get("custom_boundary_features", None)

    def initialize(self):
        self.log(" | Compute boundary manifold")
        self.boundary_mesh = self.mesh.boundary_connectivity.mesh
        if self.features and self.feat is None :
            self.feat = FeatureEdgeDetector(flag_corners=False, verbose=self.verbose)(self.boundary_mesh)
        self.vertex_normals = attributes.vertex_normals(self.boundary_mesh, persistent=False, interpolation="angle")
        
        self.log(" | Compute spherical harmonics coordinates on the boundary")
        face_bases_sh = Attribute(float, 9)
        for iF,F in enumerate(self.boundary_mesh.faces):
            pA,pB,pC = (self.boundary_mesh.vertices[v] for v in F)
            _,_,NF = geom.face_basis(pA,pB,pC)
            axis = geom.cross(Vec(0.,0.,1.), NF)
            if abs(NF.z)<0.99:
                # if abs(axis.norm()) > 1:
                #     axis = .99 * Vec.normalized(axis)
                axis = Vec.normalized(axis) * atan2(axis.norm(), NF.z)
                face_bases_sh[iF] = SphericalHarmonics.from_vec3(axis)
            else:
                face_bases_sh[iF] =  Vec(0., 0., 0., 0., 1., 0., 0., 0., 0.)
        vertex_bases_sh = Attribute(float, 9)
        vertex_bases_sh = attributes.interpolate_faces_to_vertices(self.boundary_mesh, face_bases_sh, vertex_bases_sh, weight="angle")

        self.frames = [Rotation.identity() for _ in self.mesh.id_vertices]
        for vb in self.boundary_mesh.id_vertices:
            v = self.mesh.boundary_connectivity.b2m_vertex[vb]
            f,a = SphericalHarmonics.project_to_frame(vertex_bases_sh[vb])
            self.var[9*v:9*(v+1)] = a
            self.frames[v] = f
        self.initialized = True

    def _compute_frame_from_sh(self):
        iterator = self.mesh.id_vertices
        if self.verbose: iterator = tqdm(iterator, desc="sh_to_frames")
        for v in iterator:
            shv = Vec(self.var[9*v:9*(v+1)])
            f,a = SphericalHarmonics.project_to_frame(shv)
            self.var[9*v:9*(v+1)] = a
            self.frames[v] = f

    def _compute_sh_from_frames(self):
        for i, frame in enumerate(self.frames):
            self.var[9*i:9*(i+1)] = SphericalHarmonics.from_frame(frame)

    def _graph_laplacian(self):
        """Tensor product between graph laplacian matrix and Id(9x9)"""
        lap = graph_laplacian(self.mesh).tolil()
        return scipy.sparse.kron(lap, scipy.sparse.eye(9), format="csc")

    def _volume_laplacian(self):
        lap = volume_laplacian(self.mesh).tolil()
        return scipy.sparse.kron(lap, scipy.sparse.eye(9), format="csc")

    def _volume_weight_matrix(self):
        A = volume_weight_matrix(self.mesh).tolil()
        return scipy.sparse.kron(A, scipy.sparse.eye(9), format="csc")

    def compute_constraints(self):
        self.log(" | Compute constraints matrix")
        nvar = len(self.var)
        ncstr_normals = len(self.mesh.boundary_vertices) - (0 if self.feat is None else len(self.feat.feature_vertices))
        ncstr_features = 0 if self.feat is None else 9*len(self.feat.feature_vertices)
        ncstr = ncstr_normals + ncstr_features
        cstrMat = scipy.sparse.lil_matrix((ncstr, nvar))
        cstrRHS = np.zeros(ncstr)
        row = 0
        for iv in self.boundary_mesh.id_vertices:
            if self.feat is not None and iv in self.feat.feature_vertices: continue
            axis = rotations.axis_rot_from_z(self.vertex_normals[iv])
            nrml_frame = SphericalHarmonics.from_vec3(axis)
            v = self.mesh.boundary_connectivity.b2m_vertex[iv]
            for _c in range(9):
                cstrMat[row,9*v+_c] = nrml_frame[_c]
            cstrRHS[row] = nrml_frame.dot(self.var[9*v:9*(v+1)])
            row += 1

        if self.feat is not None:
            for iv in self.feat.feature_vertices:
                v = self.mesh.boundary_connectivity.b2m_vertex[iv]
                # frame at iv is locked
                for _c in range(9):
                    cstrMat[row+_c, 9*v+_c] = 1
                    cstrRHS[row+_c] = self.var[9*v+_c]
                row += 9
        cstrMat = cstrMat.tocsc()
        return cstrMat, cstrRHS

    def normalize(self):
        """Calls projection for every vertex and also updates representations variables. 
        The projection garantees that boundary frames stay aligned with their normal.
        """
        if self.feat is None:
            iterator = self.mesh.id_vertices
        else:
            iterator = (v for v in self.mesh.id_vertices if not self.mesh.is_vertex_on_border(v) or self.mesh.boundary_connectivity.m2b_vertex[v] not in self.feat.feature_vertices)
        if self.verbose: iterator = tqdm(iterator, desc="normalize")
        for i in iterator:
            nrml=self.vertex_normals[i] if self.mesh.is_vertex_on_border(i) else None
            frame, a = SphericalHarmonics.project_to_frame(self.var[9*i:9*(i+1)], stop_threshold=1e-5, nrml_cstr=nrml)
            self.var[9*i:9*(i+1)] = a
            self.frames[i] = frame

    def optimize(self):
        self._check_init()
        # Create Laplacian
        lap = self._volume_laplacian()
        # lap = self._graph_laplacian()
        A = self._volume_weight_matrix()

        # Create Constraints Matrix
        nvar = len(self.var)
        cstrMat, cstrRHS = self.compute_constraints()
       
        self.log(" | Initial solve of linear system")
        instance = OSQP()
        instance.setup(lap, None, A=cstrMat, l=cstrRHS, u=cstrRHS, verbose=self.verbose, linsys_solver=get_osqp_lin_solver())
        res = instance.solve()
        self.var = res.x

        if self.smooth_attach_weight is not None:
            alpha = self.smooth_attach_weight
        else:
            try:
                alpha = abs(sp.linalg.eigsh(lap, k=1, which="SM", M=A, tol=1e-4, maxiter=1000, return_eigenvectors=False)[0])
            except Exception as e:
                self.log("Estimation of alpha failed: {}".format(e))
                alpha = abs(sp.linalg.eigsh(lap-0.01*sp.identity(lap.shape[0]), k=1, M=A, which="SM", tol=1e-4, maxiter=1000, return_eigenvectors=False)[0])            
        self.log("Attach weight: {}".format(alpha))

        if self.n_smooth>0:
            self.log("Solve linear system {} times with diffusion".format(self.n_smooth))
            Id = scipy.sparse.eye(nvar, format="csc")
            mat = lap + alpha * Id
            instance = OSQP()
            instance.setup(mat,self.var, A=cstrMat, l=cstrRHS, u=cstrRHS)
            for _ in range(self.n_smooth):
                self.normalize()
                instance.update(q = -alpha * self.var)
                res = instance.solve()
                self.var = res.x
        self.normalize()
        self.smoothed = True

    def flag_singularities(self):
        self._check_init()
        ZERO_THRESHOLD = 1e-2
        self.log("Flag Singularities")
        if self.mesh.faces.has_attribute("singuls"):
            self._singuls_tri = self.mesh.faces.attribute("singuls")
        else:
            self._singuls_tri = self.mesh.faces.create_attribute("singuls", float)
        self._omegas = dict()
        # first compute every matching for every edge
        for e,(A,B) in enumerate(self.mesh.edges):
            rA,rB = self.frames[A], self.frames[B]
            self._omegas[e] = match_rotation(rA,rB)

        for T in self.mesh.id_faces:
            A,B,C = self.mesh.faces[T]
            eAB,eBC,eCA = self.mesh.connectivity.edge_id(A,B), self.mesh.connectivity.edge_id(B,C), self.mesh.connectivity.edge_id(C,A)

            rAB = self.omegas[eAB] if A<B else self.omegas[eAB].inv()
            rBC = self.omegas[eBC] if B<C else self.omegas[eBC].inv()
            rCA = self.omegas[eCA] if C<A else self.omegas[eCA].inv()
            rot =  rCA * rBC * rAB
            dfct = rot.magnitude()
            if dfct> ZERO_THRESHOLD:
                self._singuls_tri[T] = dfct #round(dfct*2/pi)

    @property
    def omegas(self):
        if self._omegas is None:
            self.flag_singularities()
        return self._omegas

    @property
    def singuls(self):
        if self._singuls_tri is None:
            self.flag_singularities()
        return self._singuls_tri

    @property
    def singularity_graph(self):
        singu_graph = PolyLine()
        singuls_on_graph = singu_graph.edges.create_attribute("singus", float)
        bary_tet = attributes.cell_barycenter(self.mesh, persistent=False)
        bary_tri = attributes.face_barycenter(self.mesh, persistent=False)

        n = 0
        for Tri in self.mesh.id_faces:
            if abs(self.singuls[Tri])>1e-2:
                # triangle is singular
                if self.mesh.is_face_on_border(Tri):
                    Tet = self.mesh.connectivity.face_to_cells(Tri)[0]
                    singu_graph.vertices += [bary_tri[Tri], bary_tet[Tet]]
                else:
                    T1, T2 = self.mesh.connectivity.face_to_cells(Tri)
                    singu_graph.vertices += [bary_tet[T1], bary_tet[T2]]
                singu_graph.edges.append((2*n, 2*n+1))
                singuls_on_graph[n] = self.singuls[Tri]
                n += 1
        return singu_graph

    def export_as_mesh(self) -> SurfaceMesh:
        """
        Exports the frame field as a mesh for visualization.

        Returns:
            SurfaceMesh: the frame field as a mesh object, where small cubes represent frames
        """
        self._check_init()
        L = attributes.mean_edge_length(self.mesh,50)
        FFMesh = RawMeshData()
        col = FFMesh.faces.create_attribute("color", float, 3)
        RED,GREEN,BLUE = Vec(1.,0.,0), Vec(0.,1.,0.), Vec(0.,0.1,1.)
        
        for i,pV in enumerate(self.mesh.vertices):
            r : Rotation = self.frames[i]
            cube_v = axis_aligned_cube()
            cube_v = transform.scale(cube_v, 0.5*L, orig=Vec(0.,0.,0.))
            cube_v = transform.rotate(cube_v, r, orig=Vec(0.,0.,0.))
            cube_v = transform.translate(cube_v, pV)
            FFMesh.vertices += cube_v.vertices
            FFMesh.faces += [tuple((8*i+x for x in F)) for F in cube_v.faces]

            col[12*i] = RED
            col[12*i+1] = RED
            col[12*i+10] = RED
            col[12*i+11] = RED

            col[12*i+2] = GREEN
            col[12*i+3] = GREEN
            col[12*i+6] = GREEN
            col[12*i+7] = GREEN
            
            col[12*i+4] = BLUE
            col[12*i+5] = BLUE
            col[12*i+8] = BLUE
            col[12*i+9] = BLUE
        return SurfaceMesh(FFMesh)
