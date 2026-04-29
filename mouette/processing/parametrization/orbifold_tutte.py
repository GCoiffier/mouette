from .base import BaseParametrization

from ...mesh import copy, RawMeshData
from ...mesh.datatypes import *

from ...geometry import Vec
from ...mesh.mesh_attributes import Attribute
from ...attributes.glob import euler_characteristic
from ...attributes import cotan_weights
from ..paths import shortest_path
from .. import SurfaceMeshCutter
from ... import utils

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
from enum import Enum

class OrbifoldTutteEmbedding(BaseParametrization):
    """
    Orbifold Tutte's embedding generalizes Tutte's embedding to spherical topologies. Using periodic conditions on a virtually defined boundary, it creates a parametrization that is a tiling of the plane, leading to a seamless parametrization of meshes with sphere topologies

    References:
        [1] Orbifold Tutte Embeddings, Noam Aigerman and Yaron Lipman, ACM Transaction on Graphics, 2015
    """

    class OrbifoldType(Enum):
        SQUARE        = 1 # { pi/2,  pi,  pi/2 }
        DIAMOND       = 2 # { 2pi/3, 2pi/3, 2pi/3 }
        TRIANGLE      = 3 # { pi, 2pi/3, 2pi/6 }
        PARALLELOGRAM = 4 # { pi, pi, pi, pi }

        @classmethod
        def from_string(cls, txt:str):
            txt = txt.lower()
            if "square" in txt:
                return OrbifoldTutteEmbedding.OrbifoldType.SQUARE
            if "diamond" in txt:
                return OrbifoldTutteEmbedding.OrbifoldType.DIAMOND
            if "triangle" in txt:
                return OrbifoldTutteEmbedding.OrbifoldType.TRIANGLE
            if "parallelogram" in txt:
                return OrbifoldTutteEmbedding.OrbifoldType.PARALLELOGRAM


    class InvalidConesException(Exception):
        def __init__(self):
            message = f"Cones should be 3 indices (for SQUARE, DIAMOND and TRIANGLE orbifolds) or 4 indices (for PARALLELOGRAM orbifold). Aborting."
            super().__init__(message)

    def __init__(
            self, 
            mesh: SurfaceMesh, 
            orbifold_type: str, 
            cones: list, 
            use_cotan:bool=True, 
            verbose:bool=False, 
            **kwargs
        ):
        """
        Args:
            mesh (SurfaceMesh): the mesh to embed. Should be a surface with disk topology.
            orbifold_type (str): which orbifold to project to. Choices are ["square", "diamond", "triangle", "parallelogram"].
            cones (iterable): Cone points that define the virtual boundary (three or four points, depending on the orbifold type). Need to be valid vertex indices. If not specified, will be generated automatically. Defaults to None.
            use_cotan (bool, optional): whether to use Tutte's original barycentric embedding [1], or use cotangents as weights in the laplacian matrix ([2]). Defaults to True.
            verbose (bool, optional): verbose mode. Defaults to True.
        """
        kwargs["save_on_corners"] = True # method always saves on face corners 
        super().__init__("OrbifoldTutte", mesh, verbose, **kwargs)
        self._use_cotan : bool = use_cotan

        self.orbifold_type = OrbifoldTutteEmbedding.OrbifoldType.from_string(orbifold_type)
        self.log("Orbifold type:", self.orbifold_type)
        self.cones = self._check_cone_validity(cones)

        self.vertex_type : Attribute = None
        self._cutter : SurfaceMeshCutter = None

    def _check_cone_validity(self, cones):
        try:
            if self.orbifold_type == OrbifoldTutteEmbedding.OrbifoldType.PARALLELOGRAM:
                p1,p2,p3,p4 = cones
                assert all((isinstance(p,int) for p in (p1,p2,p3,p4)))
                # assert all([0 <= p < len(self.mesh.vertices) for p in (p1,p2,p3,p4)])
                return [p1,p2,p3,p4]
            else:
                p1,p2,p3 = cones
                assert all((isinstance(p,int) for p in (p1,p2,p3)))
                # assert all([0 <= p < len(self.mesh.vertices) for p in (p1,p2,p3)])
                return [p1,p2,p3]
        except Exception as e:
            raise OrbifoldTutteEmbedding.InvalidConesException()

    @property
    def cones_as_pointcloud(self) -> PointCloud:
        """Returns the cone distribution as a point cloud
        """
        pc = RawMeshData()
        for i in self.cones:
            pc.vertices.append(self.mesh.vertices[i])
        return PointCloud(pc)

    def run(self) :
        if not (len(self.mesh.boundary_vertices)==0 and euler_characteristic(self.mesh)==2):
            raise Exception("Mesh is not a topological sphere. Cannot run parametrization.")

        UV = None
        if self.orbifold_type == OrbifoldTutteEmbedding.OrbifoldType.SQUARE:
            UV = self._run_square()
        elif self.orbifold_type == OrbifoldTutteEmbedding.OrbifoldType.DIAMOND:
            raise NotImplementedError
        elif self.orbifold_type ==OrbifoldTutteEmbedding.OrbifoldType.TRIANGLE:
            raise NotImplementedError
        elif self.orbifold_type == OrbifoldTutteEmbedding.OrbifoldType.PARALLELOGRAM:
            UV = self._run_parallelogram()
        else:
            raise Exception("Invalid Orbifold type. should not happen.")
        
        if UV is None: return
        self.uvs = self.mesh.face_corners.create_attribute("uv_coords", float, 2, dense=True)
        for c,v in enumerate(self.cut_mesh.face_corners):
            self.uvs[c] = UV[2*v:2*v+2]
        
    def _run_square(self):
        ## Define a cut mesh between boundaries
        paths = shortest_path(self.mesh, self.cones[1], (self.cones[0],self.cones[2]))
        
        # compute set of cut edges
        edges_to_cut = set()
        for path in paths.values():
            for (a,b) in utils.consecutive_pairs(path):
                edges_to_cut.add(self.mesh.connectivity.edge_id(a,b))

        # cut mesh
        self._cutter = SurfaceMeshCutter(self.mesh, verbose=self.verbose)
        self._cutter.cut(edges_to_cut)

        # compute cut infos necessary to setup constraints 
        bnd_vertex_pairs = set()
        for singu in (self.cones[0], self.cones[2]):
            for v,v_next in utils.consecutive_pairs(paths[singu]):
                if v in self.cones : continue
                T_left, T_right = self.mesh.connectivity.edge_to_faces(v,v_next)
                # assert T_left is not None and T_right is not None # should not happen since input mesh has sphere topology
                c_left, c_right = self.mesh.connectivity.vertex_to_corner_in_face(v, T_left), self.mesh.connectivity.vertex_to_corner_in_face(v, T_right)
                a = self.cut_mesh.face_corners[c_left]
                b = self.cut_mesh.face_corners[c_right]
                bnd_vertex_pairs.add((a,b,singu))

        weights = cotan_weights(self.mesh) if self._use_cotan else Attribute(float, default_value=1.)
        get_weight = lambda a,b : weights[self.mesh.connectivity.edge_id(self._cutter.ref_vertex(a), self._cutter.ref_vertex(b))]
        
        ### Build system
        s0 = self.cones[0]
        s3 = self.cones[2]
        
        # get s1 and s2 from mesh connectivity to ensure they are not permuted (=> folded parametrization)
        v,v_next = paths[self.cones[0]][:2]
        T_left, T_right = self.mesh.connectivity.edge_to_faces(v,v_next)
        s1 = self.cut_mesh.face_corners[self.mesh.connectivity.vertex_to_corner_in_face(v, T_left)]
        s2 = self.cut_mesh.face_corners[self.mesh.connectivity.vertex_to_corner_in_face(v, T_right)]

        rows, cols, vals, rhs = [],[],[],[]
        irow = 0

        # 1] Interior vertices should be convex combination of neighbors
        for v in self.cut_mesh.interior_vertices:
            sum_w = 0.
            for nv in self.cut_mesh.connectivity.vertex_to_vertices(v):
                rows += [irow, irow+1]
                cols += [2*nv, 2*nv+1]
                w = get_weight(v,nv)
                vals += [w, w]
                sum_w += w
            rows += [irow, irow+1]
            cols += [2*v, 2*v+1]
            vals += [-sum_w, -sum_w]
            rhs += [0., 0.]
            irow += 2

        # 2] - Boundary vertices should be a convex combination up to the cut jump
        for a,b,sng in bnd_vertex_pairs:
            sum_wa, sum_wb = 0., 0.
            for nv in self.cut_mesh.connectivity.vertex_to_vertices(a):
                rows += [irow, irow+1]
                cols += [2*nv, 2*nv+1]
                w = get_weight(a,nv)
                vals += [w, w]
                sum_wa += w
            rows += [irow, irow+1]
            cols += [2*a, 2*a+1]
            vals += [-sum_wa, -sum_wa]
            
            for nv in self.cut_mesh.connectivity.vertex_to_vertices(b):
                rows += [irow, irow+1]
                cols += [2*nv+1, 2*nv]
                w = get_weight(b,nv)
                vals += [-w, w]
                sum_wb += w
            rows += [irow, irow+1]
            cols += [2*b+1, 2*b]
            vals += [sum_wb, -sum_wb]
            rhs += [0., 0.]
            irow += 2

            # - edges along the cut should match w.r.t the nearest singularity
            rows += [irow, irow,  irow,    irow,    irow+1, irow+1,  irow+1, irow+1]
            cols += [2*a,  2*sng, 2*sng+1, 2*b+1,   2*a+1,  2*sng+1, 2*sng,  2*b   ]
            vals += [1,    -1,    -1,      1,       1,      -1,      1,      -1    ]
            rhs  += [0., 0.]
            irow += 2
            
        # 3] Singularity vertices are fixed at the corners of the unit square
        rows += [irow+i for i in range(8)]
        cols += [2*s0, 2*s0+1, 2*s1, 2*s1+1, 2*s2, 2*s2+1, 2*s3, 2*s3+1]
        vals += [1.,   1.,     1.,   1.,     1.,   1.,     1.,   1.]
        rhs  += [0.,   0.,     0.,   1.,     1.,   0.,     1.,   1.]
        irow += 8

        L = sp.coo_matrix((vals, (rows,cols))).tocsc()
        rhs = np.asarray(rhs, dtype=float)
        UV = linalg.spsolve(L, rhs)
        return UV

    def _run_parallelogram(self):
        ## Define a cut mesh between boundaries
        path1 = shortest_path(self.mesh, self.cones[0], self.cones[1])[self.cones[1]]
        path2 = shortest_path(self.mesh, self.cones[1], self.cones[2])[self.cones[2]]
        path3 = shortest_path(self.mesh, self.cones[2], self.cones[3])[self.cones[3]]

        # compute set of cut edges
        edges_to_cut = set()
        for path in (path1,path2,path3):
            for (a,b) in utils.consecutive_pairs(path):
                edges_to_cut.add(self.mesh.connectivity.edge_id(a,b))
        self._cutter = SurfaceMeshCutter(self.mesh, verbose=self.verbose)
        self._cutter.cut(edges_to_cut)
    

        weights = cotan_weights(self.mesh) if self._use_cotan else Attribute(float, default_value=1.)
        get_weight = lambda a,b : weights[self.mesh.connectivity.edge_id(self._cutter.ref_vertex(a), self._cutter.ref_vertex(b))]
        
        ### Build system
        s0 = self.cones[0]
        s3 = self.cones[3]
        
        # s1 and s2 are duplicated
        v,v_next = path2[:2]
        assert self._cutter.ref_vertex(v) == self.cones[1] 
        T_left, T_right = self.mesh.connectivity.edge_to_faces(v,v_next)
        s1a = self.cut_mesh.face_corners[self.mesh.connectivity.vertex_to_corner_in_face(v, T_left)]
        s1b = self.cut_mesh.face_corners[self.mesh.connectivity.vertex_to_corner_in_face(v, T_right)]

        v,v_next = path3[:2]
        assert self._cutter.ref_vertex(v) == self.cones[2] 
        T_left, T_right = self.mesh.connectivity.edge_to_faces(v,v_next)
        s2a = self.cut_mesh.face_corners[self.mesh.connectivity.vertex_to_corner_in_face(v, T_left)]
        s2b = self.cut_mesh.face_corners[self.mesh.connectivity.vertex_to_corner_in_face(v, T_right)]

        rows, cols, vals, rhs = [],[],[],[]
        irow = 0

        # 1] Interior vertices should be convex combination of neighbors
        for v in self.cut_mesh.interior_vertices:
            sum_w = 0.
            for nv in self.cut_mesh.connectivity.vertex_to_vertices(v):
                rows += [irow, irow+1]
                cols += [2*nv, 2*nv+1]
                w = get_weight(v,nv)
                vals += [w, w]
                sum_w += w
            rows += [irow, irow+1]
            cols += [2*v, 2*v+1]
            vals += [-sum_w, -sum_w]
            rhs += [0., 0.]
            irow += 2

        # 2] - Boundary vertices should be a convex combination up to the cut jump
        for path, sa,sb, mult in ((path1, s0,s0, -1), (path2, s1a,s1b, 1), (path3, s3,s3, -1)):
            # mult takes into account whether rotation is pi (mult=-1) or 0 (mult=1) 
            for v,v_next in utils.consecutive_pairs(path):
                if v in self.cones : continue
                T_left, T_right = self.mesh.connectivity.edge_to_faces(v,v_next)
                # assert T_left is not None and T_right is not None # should not happen since input mesh has sphere topology
                c_left, c_right = self.mesh.connectivity.vertex_to_corner_in_face(v, T_left), self.mesh.connectivity.vertex_to_corner_in_face(v, T_right)
                a = self.cut_mesh.face_corners[c_left]
                b = self.cut_mesh.face_corners[c_right]
       
                sum_wa, sum_wb = 0., 0.
                for nv in self.cut_mesh.connectivity.vertex_to_vertices(a):
                    rows += [irow, irow+1]
                    cols += [2*nv, 2*nv+1]
                    w = get_weight(a,nv)
                    vals += [w, w]
                    sum_wa += w
                rows += [irow, irow+1]
                cols += [2*a, 2*a+1]
                vals += [-sum_wa, -sum_wa]
                
                for nv in self.cut_mesh.connectivity.vertex_to_vertices(b):
                    rows += [irow, irow+1]
                    cols += [2*nv, 2*nv+1]
                    w = get_weight(b,nv)
                    vals += [w*mult, w*mult]
                    sum_wb += w
                rows += [irow, irow+1]
                cols += [2*b, 2*b+1]
                vals += [-sum_wb*mult, -sum_wb*mult]
                rhs += [0., 0.]
                irow += 2

                # - edges along the cut should match w.r.t the nearest singularity
                if mult==1:
                    rows += [irow, irow,  irow, irow,  irow+1, irow+1, irow+1, irow+1]
                    cols += [2*a,  2*sa,  2*b,  2*sb,  2*a+1,  2*sa+1, 2*b+1,  2*sb+1]
                    vals += [1.,   -1.,   -1.,  1.,    1,      -1,     -1,      1    ]
                elif mult==-1:
                    rows += [irow, irow,  irow, irow,  irow+1, irow+1,  irow+1, irow+1]
                    cols += [2*a,  2*sa,  2*b,  2*sb,  2*a+1,  2*sa+1,  2*b+1,  2*sb+1]
                    vals += [1,    -1,    1,    -1.,   1,      -1,      1,      -1.    ]
                rhs  += [0., 0.]
                irow += 2
            
        # 3] Singularity vertices are fixed
        rows += [irow+i for i in range(12)]
        vals += [1.]*12
        cols += [2*s0, 2*s0+1, 2*s1a, 2*s1a+1, 2*s1b, 2*s1b+1, 2*s2a, 2*s2a+1, 2*s2b, 2*s2b+1, 2*s3, 2*s3+1]
        rhs  += [0.,   0.,     1.,    0.,      -1.,   0.,      1.,    1.,      -1.,   1.,      0.,   1.    ]
        irow += 12

        L = sp.coo_matrix((vals, (rows,cols))).tocsc()
        rhs = np.asarray(rhs, dtype=float)
        UV = linalg.spsolve(L, rhs)
        return UV

    @property
    def cut_mesh(self) -> SurfaceMesh:
        return self._cutter.cut_mesh
    
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