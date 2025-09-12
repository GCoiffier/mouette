from .base import BaseParametrization

from ...mesh.datatypes import SurfaceMesh,PolyLine
from ...mesh.mesh_attributes import Attribute, ArrayAttribute
from ...mesh.mesh import copy, save

from ..cutting import SingularityCutter
from ..connection import SurfaceConnectionFaces
from ..features import FeatureEdgeDetector
from ..framefield import FrameField
from ..trees import FaceSpanningTree
from ... import attributes

from ... import geometry as geom
from ...geometry import Vec
from ... import utils

import numpy as np
import cmath
import scipy.sparse as sp
# from scipy.optimize import milp
from osqp import OSQP

class FrameFieldIntegration(BaseParametrization):
    """
    Integration of a smooth vector field into a seamless parametrization

    References:
        - [1] _Mixed-integer quadrangulation_, Bommes D, Zimmer H. and Kobbelt L., ACM Transaction on Graphics, 2009
        
        - [2] _QuadCover - Surface Parameterization using Branched Coverings_, Kälberer F., Nieser M. and Polthier K., Computer Graphics Forum, 2007
    """

    def __init__(self, framefield: FrameField, scaling: float = 1., verbose:bool=False, **kwargs):
        """
        Args:
            framefield (FrameField): the frame field object to integrate.
            singularities (Attribute): float Attribute on vertices. Gives the target angle defects of vertices
            use_cotan (bool, optional): if True, uses cotangents as weights in the laplacian matrix. Defaults to True. 
            verbose (bool, optional): verbose mode. Defaults to False.

        Keyword Args:    
            debug (bool, optional) : debug mode. Generates additionnal output. Defaults to False.

        Note:
            It is not necessary to provide the input mesh, as it is accessed via the `framefield` argument

        Example:
            [https://github.com/GCoiffier/mouette/blob/main/examples/parametrization/ff_integrate.py](https://github.com/GCoiffier/mouette/blob/main/examples/parametrization/ff_integrate.py)
        
        Raises:
            AssertionError : the frame field should be defined on faces and be of order 4
        """

        super().__init__("Frame Field Integration", framefield.mesh, verbose, **kwargs)
        self._use_cotan = kwargs.get("use_cotan", True)
        self._debug = kwargs.get("debug", False)
        self.save_on_corners : bool = True
        self.scaling : float = scaling

        self.frame_field = framefield
        if self.frame_field.element != "faces":
            raise Exception("The frame field to integrate is not defined on faces. Aborting")
        if self.frame_field.order != 4:
            raise Exception("The frame field to integrate is not of order 4. Aborting")

        ###
        self._cutter : SingularityCutter
        self._matching : np.ndarray = None # matching attribute on faces
        self._jumps : Attribute = None # jump attribute on edges
        self._singus : Attribute = None # position and indices of cones (on vertices)
        self._feature_dir : Attribute = None # whether a feature edge is horizontal (constant v) or vertical (constant u)
        self._tree : FaceSpanningTree = None # traversal tree
        
        self._var_indirection : np.ndarray = None # corner_id -> variable id in system
        self._nvar : int = None
        
        ### System matrices
        # Solve min ||Qx - q||² s.t. Ax = b with OSQP
        self._Q = None
        self._q = None
        self._A = None
        self._b = None

    @property
    def _conn(self) -> SurfaceConnectionFaces:
        # Shortcut getter
        return self.frame_field.conn

    @property
    def _feat(self) -> FeatureEdgeDetector:
        # Shortcut getter
        return self.frame_field.feat
    
    @property
    def seam_graph(self) -> PolyLine:
        """
        Returns:
            PolyLine: Polyline of seam edges
        """
        return self._cutter.cut_graph

    def run(self):
        """Computes the integration"""

        self.log("Smooth frame field if needed")
        self.frame_field.run()
        self.frame_field.flag_singularities() # computes singularities
        self._singus = self.mesh.vertices.get_attribute("singuls")

        self.log("Perform cuts between cones")
        ## Define a cut mesh along the shortest path towards the boundary
    
        # Determine cutting strategy
        cutting_strategy = "shortest"
        if self._feat is not None and not self._feat.only_border: 
            cutting_strategy = "feature"
        if len(self._singus)>200: 
            cutting_strategy = "simple"
        self._cutter = SingularityCutter(self.mesh, self._singus, strategy=cutting_strategy, features=self._feat, verbose=self.verbose)
        self._cutter.run()

        self.log("Compute frame field matching")
        self._compute_matching()
        self._compute_jumps()

        self.log("Solve gradient least-square system")
        self._gather_variables()
        self._integrate()

        # if self._debug : self._export_jacobian_dets(Jac)
        # self._rescale_UVs()

    def _get_ff_dir(self, T:int) -> complex:
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
            rp = self._get_ff_dir(parent)
            angles = [utils.maths.angle_diff( cmath.phase(rp), cmath.phase(rf) - self._conn.transport(face,parent)) for rf in utils.maths.roots(self.frame_field[face], 4)]
            abs_angles = [abs(_a) for _a in angles]
            self._matching[face] = np.argmin(abs_angles)

    def _compute_jumps(self):
        self._jumps = self.mesh.edges.create_attribute("jump", int)
        for e in self._cutter.cut_edges:
            T1,T2 = self.mesh.connectivity.edge_to_faces(*self.mesh.edges[e])
            if T1 is None or T2 is None : continue # boundary edge -> no jump
            ff1, ff2 = self._get_ff_dir(T1), self._get_ff_dir(T2)
            jump = utils.maths.angle_diff( cmath.phase(ff1), cmath.phase(ff2) - self._conn.transport(T2,T1))
            self._jumps[e] = round(2*jump/np.pi)%4 # integer multiple of pi/2

    def _gather_variables(self):
        uf = utils.UnionFind(self.mesh.id_corners)
        for e,(A,B) in enumerate(self.mesh.edges):
            if e in self._cutter.cut_edges : continue
            T1, T2 = self.mesh.connectivity.edge_to_faces(A,B)
            if T1 is None or T2 is None: continue
            cA1,cB1 = self.mesh.connectivity.vertex_to_corner_in_face(A,T1), self.mesh.connectivity.vertex_to_corner_in_face(B,T1)
            cA2,cB2 = self.mesh.connectivity.vertex_to_corner_in_face(A,T2), self.mesh.connectivity.vertex_to_corner_in_face(B,T2)
            uf.union(cA1, cA2)
            uf.union(cB1, cB2)
        self.nvar = 2*uf.n_comps
        component_id = dict()
        for i,r in enumerate(uf.roots()):
            component_id[r] = i
        self._var_indirection = np.zeros(len(self.mesh.face_corners), dtype=int)
        for c in self.mesh.id_corners:
            self._var_indirection[c] = component_id[uf.find(c)]


    def _integrate(self):
        id = lambda c : self._var_indirection[c] # shortcut
        area = attributes.face_area(self.mesh)        

        #### Compute objective
        # We want grad u and grad v to align with directions of the frame field in each face
        vals, cols, rows = [], [], []
        n_lines = 4*len(self.mesh.faces)
        n_cols = self.nvar #uf.n_comps
        rhs = np.zeros(n_lines, dtype=float) # right hand side of the least square problem

        for iT,(A,B,C) in enumerate(self.mesh.faces):
            aT = 2*area[iT]
            xA,yA = self._conn.project(self.mesh.vertices[A], iT)
            xB,yB = self._conn.project(self.mesh.vertices[B], iT)
            xC,yC = self._conn.project(self.mesh.vertices[C], iT)
            iA,iB,iC = (id(_c) for _c in self.mesh.connectivity.face_to_corners(iT))
            # assert iA<uf.n_comps and iB<uf.n_comps and iC<uf.n_comps

            # system for u
            rows += [4*iT, 4*iT+1, 4*iT, 4*iT+1, 4*iT, 4*iT+1]
            cols += [2*iA, 2*iA, 2*iB, 2*iB, 2*iC, 2*iC]
            vals += [(yB-yC)/aT, (xC-xB)/aT, (yC-yA)/aT, (xA-xC)/aT, (yA-yB)/aT, (xB-xA)/aT] 
        
            # system for v
            rows += [4*iT+2, 4*iT+3, 4*iT+2, 4*iT+3, 4*iT+2, 4*iT+3]
            cols += [2*iA+1, 2*iA+1, 2*iB+1, 2*iB+1, 2*iC+1, 2*iC+1]
            vals += [(yB-yC)/aT, (xC-xB)/aT, (yC-yA)/aT, (xA-xC)/aT, (yA-yB)/aT, (xB-xA)/aT] 

            zT = self._get_ff_dir(iT)
            rhs[4*iT:4*iT+4] = [zT.real, zT.imag, -zT.imag, zT.real]
        self._Q = sp.coo_matrix((vals, (rows, cols)), shape=(n_lines, n_cols)).tocsc()
        self._q = rhs * self.scaling
        # problem to solve is min_x ||Qx - q||² 
        
        #### Compute constraints
        self._A = None
        self._b = None

        # Seam constraints
        rows, cols, vals = [], [], []
        irow = 0
        for e in self._cutter.cut_edges:
            A,B = self.mesh.edges[e]
            T1, T2 = self.mesh.connectivity.edge_to_faces(A,B)
            if T1 is None or T2 is None:
                raise Exception("Cut edge with only one adjacent triangle. Should not happen")
            iA1 = id(self.mesh.connectivity.vertex_to_corner_in_face(A,T1))
            iA2 = id(self.mesh.connectivity.vertex_to_corner_in_face(A,T2))
            iB1 = id(self.mesh.connectivity.vertex_to_corner_in_face(B,T1))
            iB2 = id(self.mesh.connectivity.vertex_to_corner_in_face(B,T2))

            je = self._jumps[e]
            # B1-A1 = R(je*pi/2)(B2- A2)
            if je==0:
                rows += [irow,  irow,  irow,  irow,  irow+1,  irow+1,  irow+1,  irow+1 ]
                cols += [2*iB1, 2*iA1, 2*iB2, 2*iA2, 2*iB1+1, 2*iA1+1, 2*iB2+1, 2*iA2+1]
                vals += [1,       -1,   -1,    1,      1,      -1,       -1,     1    ]
            
            elif je==1:
                rows += [irow,  irow,  irow,     irow,     irow+1,  irow+1,  irow+1,  irow+1]
                cols += [2*iB1, 2*iA1, 2*iB2+1,  2*iA2+1,  2*iB1+1, 2*iA1+1, 2*iB2,   2*iA2]
                vals += [1,       -1,    -1,      1,         1,      -1,       1,       -1 ]

            elif je==2:
                rows += [irow,  irow,  irow,  irow,  irow+1,  irow+1,  irow+1,  irow+1 ]
                cols += [2*iB1, 2*iA1, 2*iB2, 2*iA2, 2*iB1+1, 2*iA1+1, 2*iB2+1, 2*iA2+1]
                vals += [  1,    -1,    1,    -1,     1,      -1,       1,       -1    ]
                
            elif je==3:
                rows += [irow,  irow,  irow,  irow,  irow+1,  irow+1,  irow+1,  irow+1 ]
                cols += [2*iB1, 2*iA1, 2*iB2+1,  2*iA2+1,  2*iB1+1, 2*iA1+1, 2*iB2,   2*iA2]
                vals += [1,       -1,     1,       -1,        1,      -1,     -1,        1 ]
            irow+=2

        self._feature_dir = self.mesh.edges.create_attribute("feat_dir", int)
        for e in self._feat.feature_edges:
            # a triangle is adjacent to only one feature edge
            A,B = self.mesh.edges[e]
            for T in self.mesh.connectivity.edge_to_faces(A,B):
                if T is None: continue
                zT = Vec.from_complex(self._get_ff_dir(T))
                iA = id(self.mesh.connectivity.vertex_to_corner_in_face(A,T))
                iB = id(self.mesh.connectivity.vertex_to_corner_in_face(B,T))
                ET = self._conn.project(self.mesh.vertices[B] - self.mesh.vertices[A], T)
                self._feature_dir[e] = round(2/np.pi * geom.angle_2vec2D(zT,ET))%2
                rows += [irow, irow]
                vals += [1, -1]
                if self._feature_dir[e] == 0: # feature edge is horizontal : vB = vA
                    cols += [2*iB+1, 2*iA+1]
                else: # feature edge is vertical : uB = uA
                    cols += [2*iB, 2*iA]
                irow += 1

        self._A = sp.coo_matrix((vals, (rows, cols)), shape=(irow, n_cols)).tocsc()
        self._b = np.zeros(irow, dtype=float)

        #### Solve system with OSQP
        Qt = self._Q.transpose()
        instance = OSQP()
        instance.setup(Qt @ self._Q, -Qt @ self._q, self._A, self._b, self._b, verbose=self.verbose,
                    eps_abs=1e-6, eps_rel=1e-6, max_iter=1000)
        result = instance.solve()
        self._wm = result
        self.uvs = self.mesh.face_corners.create_attribute("uv_coords", float, 2, dense=True)
        for cnr in self.mesh.id_corners:
            idc = id(cnr)
            self.uvs[cnr] = result.x[2*idc:2*idc+2]
    
    def export_frame_field_as_mesh(self) -> PolyLine:
        """
        Exports the frame field as a PolyLine for visualization
        """
        if self._matching is None:
            return self.frame_field.export_as_mesh()

        FFmesh = PolyLine()
        L = attributes.mean_edge_length(self.mesh)/3
        match_attr = FFmesh.edges.create_attribute("matching", int, dense=True)
        for id_face, face in enumerate(self.mesh.faces):
            basis,Y = self._conn.base(id_face)
            normal = geom.cross(basis,Y)
            pA,pB,pC = (self.mesh.vertices[_v] for _v in face)
            angle = cmath.phase(self._get_ff_dir(id_face))
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
        """Disk-topology mesh where cuts have been performed on seams"""
        return self._cutter.cut_mesh

    @property
    def cut_graph(self) -> PolyLine:
        """Seam edges as a PolyLine"""
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