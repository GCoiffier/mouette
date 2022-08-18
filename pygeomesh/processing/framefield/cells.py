from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation
from osqp import OSQP
from itertools import chain

from .base import FrameField

from ...mesh.datatypes import *
from ...mesh.mesh_data import RawMeshData
from ...mesh.mesh_attributes import Attribute, ArrayAttribute

from ... import attributes
# from ..features import FeatureEdgeDetector

from ... import geometry as geom
from ... import utils
from ...geometry import Vec, SphericalHarmonics
from ...geometry.rotations import match_rotation
from ...geometry import transform

from ...operators.laplacian import *
from ...procedural import axis_aligned_cube

class FrameField3DCells(FrameField):
    
    def __init__(self, supporting_mesh : VolumeMesh, verbose=True):
        super().__init__(verbose=verbose)
        self.mesh : VolumeMesh = supporting_mesh
        self.mesh.enable_boundary_connectivity()
        self.boundary_mesh : SurfaceMesh = None

        self.fnormals : ArrayAttribute = None
        self.cell_on_bnd : Attribute = None

        self.frames : list = None # raw representation of frames as scipy.Rotation objects
        self.var = np.zeros(9*len(self.mesh.cells))

        self._singul_vertices : Attribute = None
        self._singul_edges : Attribute = None

    def initialize(self):
        self.frames = [Rotation.identity() for _ in self.mesh.id_cells]
        
        self.log(" | Compute boundary manifold")
        self.boundary_mesh = self.mesh.boundary_connectivity.mesh
        self.fnormals = attributes.face_normals(self.boundary_mesh)
        self.cell_on_bnd = attributes.cell_faces_on_boundary(self.mesh)

        self.log(" | Compute spherical harmonics coordinates on the boundary")
        face_bases_sh = Attribute(float, 9)
        for iF,F in enumerate(self.boundary_mesh.faces):
            NF = self.fnormals[iF]
            axis = geom.cross(Vec(0.,0.,1.), NF)
            if abs(NF.z)<0.99:
                axis = Vec.normalized(axis) * atan2(axis.norm(), NF.z)
                face_bases_sh[iF] = SphericalHarmonics.from_vec3(axis)
            else:
                face_bases_sh[iF] = Vec(0., 0., 0., 0., 1., 0., 0., 0., 0.)
        for iFb in self.boundary_mesh.id_faces:
            iF = self.mesh.boundary_connectivity.b2m_face[iFb]
            iC = self.mesh.connectivity.face_to_cell(iF)[0]
            f,a = SphericalHarmonics.project_to_frame(face_bases_sh[iFb])
            self.var[9*iC:9*(iC+1)] += a
            self.frames[iC] *= f
        
        # normalization
        for iC in self.cell_on_bnd:
            nrm = geom.norm(self.var[9*iC:9*(iC+1)])
            if nrm > 1e-8 : self.var[9*iC:9*(iC+1)] /= nrm
        self.initialized = True

    def compute_frame_from_sh(self):
        iterator = self.mesh.id_cells
        if self.verbose: iterator = tqdm(iterator, desc="sh_to_frames")
        for iC in iterator:
            shv = Vec(self.var[9*iC:9*(iC+1)])
            f,a = SphericalHarmonics.project_to_frame(shv)
            self.var[9*iC:9*(iC+1)] = a
            self.frames[iC] = f

    def compute_sh_from_frames(self):
        for i, frame in enumerate(self.frames):
            self.var[9*i:9*(i+1)] = SphericalHarmonics.from_frame(frame)

    def normalize(self):
        """Calls projection for every vertex and also updates representations variables. 
        The projection guarantees that boundary frames stay aligned with their normal.
        """

        for iC in self.mesh.id_cells:
            nrm = geom.norm(self.var[9*iC:9*(iC+1)])
            if nrm>1e-8: 
                self.var[9*iC:9*(iC+1)] /= nrm

        for iC in tqdm(self.mesh.id_cells,desc="normalize1"):
            if self.cell_on_bnd[iC]==0:
                frame, a = SphericalHarmonics.project_to_frame(self.var[9*iC:9*(iC+1)], stop_threshold=1e-3)
                self.var[9*iC:9*(iC+1)] = a
                self.frames[iC] = frame

        for iFb in tqdm(self.boundary_mesh.id_faces, desc="normalize2"):
            iF = self.mesh.boundary_connectivity.b2m_face[iFb]
            iC = self.mesh.connectivity.face_to_cell(iF)[0]
            # if self.cell_on_bnd[iC]!=1 : continue
            nrml = self.fnormals[iFb]
            frame, a = SphericalHarmonics.project_to_frame(self.var[9*iC:9*(iC+1)], stop_threshold=1e-3, nrml_cstr=nrml)
            self.var[9*iC:9*(iC+1)] = a
            self.frames[iC] = frame
            
    def _laplacian(self):
        lap = laplacian_tetrahedra(self.mesh)
        return sp.kron(lap, sp.eye(9), format="csc")

    def _compute_constraints(self):
        self.log(" | Compute constraints matrix")
        nvar = len(self.var)
        ncstr = 0
        rows, cols, coeffs = [], [], []
        cstrRHS = []

        Z = Vec(0.,0.,1.)
        for iC in self.cell_on_bnd:
            if self.cell_on_bnd[iC]>1 :
                # completely lock the frame
                rows += [ncstr + _r for _r in range(9)]
                cols += [9*iC + _c for _c in range(9)]
                coeffs += [1]*9
                cstrRHS += [self.var[9*iC+_c] for _c in range(9)]
                ncstr += 9

        for iFb in self.boundary_mesh.id_faces:
            iF = self.mesh.boundary_connectivity.b2m_face[iFb]
            iC = self.mesh.connectivity.face_to_cell(iF)[0]
            if self.cell_on_bnd[iC] != 1 : continue
            axis = geom.cross(Z, self.fnormals[iFb])
            angle = geom.angle_2vec3D(Z, self.fnormals[iFb])
            if axis.norm()>1e-8: axis = Vec.normalized(axis) * angle
            nrml_frame = SphericalHarmonics.from_vec3(axis)
            rows += [ncstr]*9
            cols += [9*iC + _c for _c in range(9)]
            coeffs += [nrml_frame[_c] for _c in range(9)]
            cstrRHS.append(nrml_frame.dot(self.var[9*iC:9*(iC+1)]))
            ncstr += 1

        cstrMat = sp.csc_matrix((coeffs, (rows, cols)), shape=(ncstr, nvar))
        cstrRHS = np.array(cstrRHS)
        # print(geom.norm(cstrMat.dot(self.var) - cstrRHS))
        return cstrMat, cstrRHS

    def optimize(self, n_renorm : int = 0):
        self._check_init()
        self.log(" | Compute Laplacian")
        lap = self._laplacian()
        Q = lap #lap.transpose() @ lap
        cstrMat, cstrRHS = self._compute_constraints()
        self.log(" | Initial solve of linear system")
        instance = OSQP()
        instance.setup(Q, None, A=cstrMat, l=cstrRHS, u=cstrRHS, verbose=self.verbose, polish=True, check_termination=10, 
                        adaptive_rho=True, linsys_solver='mkl pardiso')
        res = instance.solve()
        self.var = res.x

        if n_renorm>0:
            alpha = 1e-3
            self.log(" | Solve linear system {} times with diffusion".format(n_renorm))
            Id = sp.eye(self.var.size, format="csc")
            mat = Q + alpha * Id
            instance = OSQP()
            instance.setup(mat,self.var, A=cstrMat, l=cstrRHS, u=cstrRHS)
            for _ in range(n_renorm):
                self.normalize()
                instance.update(q = -alpha * self.var)
                res = instance.solve()
                self.var = res.x

        self.log(" | Final Normalize")
        self.normalize()

    @property
    def singular_vertices(self):
        if self._singul_vertices is None :
            self.flag_singularities()
        return self._singul_vertices

    @property
    def singular_edges(self):
        if self._singul_edges is None:
            self.flag_singularities()
        return self._singul_edges

    @property
    def singularity_graph(self):
        singu_graph = RawMeshData()
        indir = dict()
        for i,v in enumerate(self.singular_vertices):
            indir[v] = i
            singu_graph.vertices.append(self.mesh.vertices[v])
        for e in self.singular_edges:
            A,B = self.mesh.edges[e]
            singu_graph.edges.append((indir[A], indir[B]))
        return PolyLine(singu_graph)

    def flag_singularities(self):
        self._check_init()
        ZERO_THRESHOLD = 0.1
        self.log("Flag Singularities")
        self._singul_vertices = self.mesh.vertices.create_attribute("singuls", bool)
        self._singul_edges = self.mesh.edges.create_attribute("singuls", float)
        for e in self.mesh.id_edges:
            r = Rotation.identity()
            for c1, c2 in utils.cyclic_pairs(self.mesh.connectivity.edge_to_cell(e)):
                r = match_rotation(self.frames[c1], self.frames[c2]) * r
            if r.magnitude()>ZERO_THRESHOLD:
                self._singul_edges[e] = r.magnitude()
        
        for e in self._singul_edges:
            A,B = self.mesh.edges[e]
            self._singul_vertices[A] = True
            self._singul_vertices[B] = True


    def export_as_mesh(self) -> Mesh:
        """
        Exports the frame field as a mesh for visualization.

        Returns:
            Mesh: the frame field as a mesh object, either SurfaceMesh for cube mode, or Polyline for frame mode
        """
        self._check_init()
        L = attributes.mean_edge_length(self.mesh,100)
        FFMesh = RawMeshData()
        col = FFMesh.faces.create_attribute("color", float, 3)
        RED,GREEN,BLUE = Vec(1.,0.,0), Vec(0.,1.,0.), Vec(0.,0.,1.)
        barycenters = attributes.cell_barycenter(self.mesh, persistent=False)

        for iC in self.mesh.id_cells:
            r : Rotation = self.frames[iC]
            cube_v = axis_aligned_cube()
            cube_v = transform.scale(cube_v, 0.3*L, orig=Vec(0.,0.,0.))
            cube_v = transform.rotate(cube_v, r, orig=Vec(0.,0.,0.))
            cube_v = transform.translate(cube_v, barycenters[iC])
            FFMesh.vertices += cube_v.vertices
            FFMesh.faces += [tuple((8*iC+x for x in F)) for F in cube_v.faces]

            col[12*iC] = RED
            col[12*iC+1] = RED
            col[12*iC+10] = RED
            col[12*iC+11] = RED

            col[12*iC+2] = GREEN
            col[12*iC+3] = GREEN
            col[12*iC+6] = GREEN
            col[12*iC+7] = GREEN
            
            col[12*iC+4] = BLUE
            col[12*iC+5] = BLUE
            col[12*iC+8] = BLUE
            col[12*iC+9] = BLUE
        return SurfaceMesh(FFMesh)
