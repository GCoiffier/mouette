from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation
from osqp import OSQP
import math

from .base import FrameField

from ...mesh.datatypes import PolyLine,SurfaceMesh,VolumeMesh
from ...mesh.mesh_data import RawMeshData
from ...mesh.mesh_attributes import Attribute, ArrayAttribute

from ... import attributes

from ... import geometry as geom
from ... import utils
from ...geometry import Vec, SphericalHarmonics
from ...geometry.rotations import match_rotation
from ...geometry import transform

from ...operators import laplacian_tetrahedra
from ...procedural import axis_aligned_cube


class FrameField3DCells(FrameField):

    class FileParsingError(Exception):
        def __init__(self, line, message):
            message = f"Reading error at line {line}: {message}"
            super().__init__(message)

    def __init__(self, 
        supporting_mesh : VolumeMesh, 
        verbose=True,
        **kwargs):
        """
        Parameters:
            supporting_mesh (VolumeMesh): the input mesh
            verbose (bool, optional): verbose mode. Defaults to True.
        """
        super().__init__("cells", "FrameField3D", verbose=verbose)
        self.mesh : VolumeMesh = supporting_mesh
        self.mesh.enable_boundary_connectivity()
        self.n_smooth = kwargs.get("n_smooth", 3)
        self.smooth_attach_weight = kwargs.get("smooth_attach_weight", None)
        self._boundary_mesh : SurfaceMesh = None

        self._fnormals : ArrayAttribute = None
        self._cell_on_bnd : Attribute = None

        self.frames : list = None # raw representation of frames as scipy.Rotation objects
        self.var = np.zeros(9*len(self.mesh.cells))

        self._singul_vertices : Attribute = None
        self._singul_edges : Attribute = None

    def read_from_file(self, file_path :str):
        """Reads the values of the frames from a .frame file

        file is supposed to have the following syntax:

            FRAME
            number_of_frames
            a1x a1y a1z b1x b1y b1z c1x c1y c1z
            a2x a2y a2z b2x b2y b2z c2x c2y c2z
            ...
            anx any anz bnx bny bnz cnx cny cnz
            END

        Parameters:
            file_path (str): path to the file

        Raises:
            FrameField3DCells.FileParsingError: if the file is not of the correct format
        """
        if self.initialized:
            self.log("Warning: reading frame field from file has erased current frames.")
            self.var = np.zeros(9*len(self.mesh.cells))
        self.frames = []
        with open(file_path, "r") as f:
            data = [l.strip() for l in f.readlines()]
        
        if data[0] != "FRAME":
            # Line 1 should only contain the word "FRAME"
            raise FrameField3DCells.FileParsingError(1,"Invalid frame field file.")
        
        try:
            # Line 2 should only contain the total number of frames
            n_frames = int(data[1])
        except Exception:
            raise FrameField3DCells.FileParsingError(2, f"Invalid number of frames '{data[1]}'")
        
        if n_frames != len(self.mesh.cells):
            raise FrameField3DCells.FileParsingError(2, f"Read number of frames ({n_frames}) does not match number of cells in the mesh ({len(self.mesh.cells)})")
        
        for line in range(n_frames):
            try:
                mat = np.array([float(x) for x in data[line+2].split()]).reshape((3,3)).T
                frame = Rotation.from_matrix(mat)
                self.frames.append(frame)
                self.var[9*line:9*(line+1)] = SphericalHarmonics.from_frame(frame)
            except Exception:
                raise FrameField3DCells.FileParsingError(line+3, f"Invalid frame {data[line+2]}")
        self.initialized = True

    def initialize(self):
        self.frames = [Rotation.identity() for _ in self.mesh.id_cells]
        
        self.log(" | Compute boundary manifold")
        self._boundary_mesh = self.mesh.boundary_connectivity.mesh
        self._fnormals = attributes.face_normals(self._boundary_mesh)
        self._cell_on_bnd = attributes.cell_faces_on_boundary(self.mesh)

        self.log(" | Compute spherical harmonics coordinates on the boundary")
        face_bases_sh = Attribute(float, 9)
        for iF,F in enumerate(self._boundary_mesh.faces):
            NF = self._fnormals[iF]
            axis = geom.cross(Vec(0.,0.,1.), NF)
            if abs(NF.z)<0.99:
                axis = Vec.normalized(axis) * math.atan2(axis.norm(), NF.z)
                face_bases_sh[iF] = SphericalHarmonics.from_vec3(axis)
            else:
                face_bases_sh[iF] = Vec(0., 0., 0., 0., 1., 0., 0., 0., 0.)
        for iFb in self._boundary_mesh.id_faces:
            iF = self.mesh.boundary_connectivity.b2m_face[iFb]
            iC = self.mesh.connectivity.face_to_cells(iF)[0]
            f,a = SphericalHarmonics.project_to_frame(face_bases_sh[iFb])
            self.var[9*iC:9*(iC+1)] += a
            self.frames[iC] *= f
        
        # normalization
        for iC in self._cell_on_bnd:
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
            if self._cell_on_bnd[iC]==0:
                frame, a = SphericalHarmonics.project_to_frame(self.var[9*iC:9*(iC+1)], stop_threshold=1e-3)
                self.var[9*iC:9*(iC+1)] = a
                self.frames[iC] = frame

        for iFb in tqdm(self._boundary_mesh.id_faces, desc="normalize2"):
            iF = self.mesh.boundary_connectivity.b2m_face[iFb]
            iC = self.mesh.connectivity.face_to_cells(iF)[0]
            # if self.cell_on_bnd[iC]!=1 : continue
            nrml = self._fnormals[iFb]
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
        for iC in self._cell_on_bnd:
            if self._cell_on_bnd[iC]>1 :
                # completely lock the frame
                rows += [ncstr + _r for _r in range(9)]
                cols += [9*iC + _c for _c in range(9)]
                coeffs += [1]*9
                cstrRHS += [self.var[9*iC+_c] for _c in range(9)]
                ncstr += 9

        for iFb in self._boundary_mesh.id_faces:
            iF = self.mesh.boundary_connectivity.b2m_face[iFb]
            iC = self.mesh.connectivity.face_to_cells(iF)[0]
            if self._cell_on_bnd[iC] != 1 : 
                continue
            axis = geom.cross(Z, self._fnormals[iFb])
            angle = geom.angle_2vec3D(Z, self._fnormals[iFb])
            if axis.norm()>1e-8: 
                axis = Vec.normalized(axis) * angle
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

    def optimize(self):
        self._check_init()
        self.log(" | Compute Laplacian")
        lap = self._laplacian()
        Q = lap #lap.transpose() @ lap
        cstrMat, cstrRHS = self._compute_constraints()
        self.log(" | Initial solve of linear system")
        instance = OSQP()
        instance.setup(Q, None, A=cstrMat, l=cstrRHS, u=cstrRHS, verbose=self.verbose, polish=True, check_termination=10, 
                        adaptive_rho=True,  linsys_solver=utils.get_osqp_lin_solver())
        res = instance.solve()
        self.var = res.x

        if self.n_smooth>0:
            alpha = self.smooth_attach_weight or 1e-3
            self.log(" | Solve linear system {} times with diffusion".format(self.n_smooth))
            Id = sp.eye(self.var.size, format="csc")
            mat = Q + alpha * Id
            instance = OSQP()
            instance.setup(mat,self.var, A=cstrMat, l=cstrRHS, u=cstrRHS)
            for _ in range(self.n_smooth):
                self.normalize()
                instance.update(q = -alpha * self.var)
                res = instance.solve()
                self.var = res.x

        self.log(" | Final Normalize")
        self.normalize()
        self.smoothed = True

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

    def export_frames_to_file(self, file_path :str):
        """
        Exports the frames as 3x3 matrices in a .frame file, with syntax:

            FRAME
            number_of_frames
            a1x a1y a1z b1x b1y b1z c1x c1y c1z
            a2x a2y a2z b2x b2y b2z c2x c2y c2z
            ...
            anx any anz bnx bny bnz cnx cny cnz
            END

        Parameters:
            file_path (str): path to the file
        """
        self.compute_frame_from_sh()
        with open(file_path, 'w') as f:
            f.write("FRAME\n")
            f.write(f"{len(self.mesh.cells)}\n")
            for ic in self.mesh.id_cells:
                mat = self.frames[ic].as_matrix()
                f.write(f"{mat[0][0]} {mat[1][0]} {mat[2][0]} {mat[0][1]} {mat[1][1]} {mat[2][1]} {mat[0][2]} {mat[1][2]} {mat[2][2]}\n")
            f.write("END\n")

    def export_as_mesh(self) -> SurfaceMesh:
        """
        Exports the frame field as a mesh for visualization.
        
        Returns:
            SurfaceMesh: The frame field as a surface mesh object, where small cubes represent frames
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
