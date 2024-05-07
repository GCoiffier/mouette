from .base import BaseParametrization

from ...mesh.datatypes import SurfaceMesh,PolyLine
from ...mesh.mesh_attributes import Attribute
from ...mesh.mesh import copy

from ... import operators
from ... import attributes
from .. import SingularityCutter
from ..trees import FaceSpanningTree
from ..connection  import SurfaceConnectionFaces

from ... import geometry as geom
from ...geometry import Vec

import numpy as np
import cmath
import scipy.sparse as sp

class ConformalConeParametrization(BaseParametrization):
    """
    Conformal cone parametrization. 
    Given a user-defined cone distribution on the surface, this algorithm cuts an input surface mesh into 
    a disk topology and parametrize it using conformal mapping

    References:
        - [1] _Conformal equivalence of triangle meshes_, Springborn B., Schröder P. and Pinkall U., ACM Transaction on Graphics, 2008
        
        - [2] _Boundary first flattening_, Sawhney R. and Crane K., ACM Transaction on Graphics, 2018
    """

    def __init__(self, mesh:SurfaceMesh, singularities: Attribute, verbose:bool=False, **kwargs):
        """
        Args:
            mesh (SurfaceMesh): Input mesh
            singularities (Attribute): float Attribute on vertices. Gives the target angle defects of vertices
            use_cotan (bool, optional): whether to use adjacency weights or cotangents in the laplacian matrix. Defaults to True. 
            verbose (bool, optional): verbose mode. Defaults to False.
            debug (bool, optional) : debug mode. Generates additional output. Defaults to False.
        """
        super().__init__("ConeParam", mesh, verbose, **kwargs)
        self._use_cotan = kwargs.get("use_cotan", True)
        self._debug = kwargs.get("debug", False)
        self.save_on_corners = True
        assert singularities.elemsize == 1
        assert singularities.type == Attribute.Type.Float
        self._singus : Attribute = singularities

        self._scale_factor : np.ndarray = None
        self._frames : np.ndarray = None
        self._conn : SurfaceConnectionFaces = None
        self._cutter : SingularityCutter
    
    def _check_singularity_validity(self):
        ### TODO
        return

    def run(self) :
        self.log("Perform cuts between cones")
        ## Define a cut mesh along the shortest path towards the boundary
        self._cutter = SingularityCutter(self.mesh, self._singus)
        self._cutter.run()

        self.log("Compute scale factor on vertices")
        ## Solve Yamabe problem to retrieve scale factor
        K = attributes.angle_defects(self.mesh, persistent=False).as_array(len(self.mesh.vertices))
        hK = self._singus.as_array(len(self.mesh.vertices))
        lap = operators.laplacian(self.mesh, cotan=self._use_cotan)
        self._scale_factor = sp.linalg.spsolve(lap, hK - K) # one scale factor per vertex
        self._scale_factor -= np.median(self._scale_factor) # log scale factor is defined up to a global constant (<=> the parametrization is defined up to a global scale)

        if self._debug:
            scale_fact_attr = self.mesh.vertices.create_attribute("scale", float, dense=True)
            for v in self.mesh.id_vertices:
                scale_fact_attr[v] = self._scale_factor[v]

        self.log("Integrate scale factor into local frames")
        cot = attributes.cotan_weights(self.mesh, dense=True, persistent=False)
        self._conn = SurfaceConnectionFaces(self.mesh)
        self._frames = np.zeros(len(self.mesh.faces), dtype=complex)

        ## Determine the root of the tree traversal. Should be a boundary face if possible
        tree_root = 0  
        edge_root = None # A boundary edge to be aligned vertically or horizontally
        if len(self.mesh.boundary_edges)>0:
            edge_root = self.mesh.boundary_edges[0]
            tree_root = [_T for _T in self.mesh.connectivity.edge_to_faces(*self.mesh.edges[edge_root]) if _T is not None][0]

        ## Perform tree traversal and integrate the scale factor
        tree = FaceSpanningTree(self.mesh, starting_face=tree_root, forbidden_edges=self._cutter.cut_edges)() # traverse the faces but avoid seams
        for face,parent in tree.traverse():
            if parent is None: # the tree root
                if edge_root is None:
                    self._frames[face] = complex(1.,0.)
                else:
                    A,B = self.mesh.edges[edge_root]
                    E = self.mesh.vertices[B] - self.mesh.vertices[A]
                    X,Y = self._conn.base(face)
                    c = complex(E.dot(X), E.dot(Y)) # edge in local basis coordinates
                    self._frames[face] = c/abs(c) # the edge gives the global frame reference
                continue

            A,B = self.mesh.connectivity.common_edge(face,parent)
            e = self.mesh.connectivity.edge_id(A,B)
            w = cot[e]* (self._scale_factor[B] - self._scale_factor[A])
            if self.mesh.connectivity.direct_face(A,B)==face: 
                w*=-1 # w is an oriented dual 1-form
            pt = self._conn.transport(face,parent)
            self._frames[face] = cmath.rect(1, w+pt) * self._frames[parent]


        self.log("Compute scaled edges in parameter space")
        scale_edges = np.zeros(2*len(self.cut_mesh.edges), dtype=float)
        cot = attributes.cotan_weights(self.cut_mesh, dense=True, persistent=False)
        for e,(A,B) in enumerate(self.cut_mesh.edges):
            rA,rB = self._cutter.ref_vertex[A], self._cutter.ref_vertex[B]
            E = self.mesh.vertices[rB] - self.mesh.vertices[rA]
            sce = np.exp( (self._scale_factor[rA] + self._scale_factor[rB])/2) # discrete scale factor of edge
            if self.cut_mesh.is_edge_on_border(A,B):
                T = [_T for _T in self.cut_mesh.connectivity.edge_to_faces(A,B) if _T is not None][0]
                X,Y = self._conn.base(T)
                ET = complex(X.dot(E), Y.dot(E))
                ET = ET / self._frames[T]
                scale_edges[2*e:2*e+2] = cot[e] * sce * Vec(ET.real, ET.imag)
            else:
                T1,T2 = self.cut_mesh.connectivity.edge_to_faces(A,B)
                X1,Y1 = self._conn.base(T1)
                ET1 = complex(X1.dot(E), Y1.dot(E))
                ET1 = ET1 / self._frames[T1]
                X2,Y2 = self._conn.base(T2)
                ET2 = complex(X2.dot(E), Y2.dot(E))
                ET2 = ET2 / self._frames[T2]
                mean = (ET1+ET2)/2
                scale_edges[2*e:2*e+2] = cot[e] * sce * Vec(mean.real, mean.imag)

        self.log("Solve for closest matching uv-coordinates")
        M = sp.kron(operators.cotan_edge_diagonal(self.cut_mesh, inverse=False) @ operators.vertex_to_edge_operator(self.cut_mesh, oriented=True).transpose(), sp.eye(2))
        UVs = sp.linalg.lsqr(M, scale_edges)[0]

        self.log("Write final result")
        ## Rescale UV coordinates to fit in [0,1]²
        xmin,xmax,ymin,ymax = float("inf"), -float("inf"), float("inf"), -float("inf")
        for v in self.cut_mesh.id_vertices:
            xmin = min(xmin, UVs[2*v])
            xmax = max(xmax, UVs[2*v])
            ymin = min(ymin, UVs[2*v+1])
            ymax = max(ymax, UVs[2*v+1])
        scale_x = xmax-xmin
        scale_y = ymax-ymin
        scale = min(scale_x, scale_y)

        ## Write final result in attribute
        self.uvs = self.mesh.face_corners.create_attribute("uv_coords", float, 2, dense=True)
        for c in self.mesh.id_corners:
            v = self.cut_mesh.face_corners[c]
            self.uvs[c] = Vec(UVs[2*v], UVs[2*v+1])/scale

    @property
    def frame_field(self) -> PolyLine:
        FFmesh = PolyLine()
        L = attributes.mean_edge_length(self.mesh)/3
        for id_face, face in enumerate(self.mesh.faces):
            basis,Y = self._conn.base(id_face)
            normal = geom.cross(basis,Y)
            pA,pB,pC = (self.mesh.vertices[_v] for _v in face)
            angle = cmath.phase(self._frames[id_face])
            bary = (pA+pB+pC)/3 # reference point for display
            r1,r2,r3,r4 = (geom.rotate_around_axis(basis, normal, angle + k*np.pi/2) for k in range(4))
            p1,p2,p3,p4 = (bary + abs(self._frames[id_face])*L*r for r in (r1,r2,r3,r4))
            FFmesh.vertices += [bary, p1, p2, p3, p4]
            FFmesh.edges += [(5*id_face, 5*id_face+k) for k in range(1,5)]
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