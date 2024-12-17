import numpy as np
from ..mesh.datatypes import *
from ..mesh.mesh import from_arrays, merge
from ..mesh.mesh_data import RawMeshData, DataContainer
from ..attributes.misc_vertices import vertex_normals
from ..geometry import Vec
from .. import utils
from .bezier import BezierCurve

from abc import ABC, abstractmethod

class _PnTriangulation(ABC):
    def __init__(self, nodes : np.ndarray, faces : np.ndarray, order: int):
        self.nodes = DataContainer(nodes, id="nodes")
        self.faces = DataContainer(faces, id="faces")
        self._order : int = order

    @classmethod
    def from_P1_mesh(cls, mesh: SurfaceMesh):
        return
    
    @abstractmethod
    def rasterize(self, res: int = 10):
        return
    
    @property
    def id_nodes(self):
        return range(len(self.nodes))
    
    @property
    def id_faces(self):
        return range(len(self.nodes))

    @property
    def dim(self) -> int:
        return self.nodes.shape[1]
    
    @property
    def order(self) -> int:
        return self._order

    @property
    def control_points(self) -> PointCloud:
        pc_nodes = from_arrays(np.array(self.nodes))
        for attr_name in self.nodes.attributes:
            pc_nodes.vertices._attr[attr_name] = self.nodes.get_attribute(attr_name)
        return pc_nodes

def rasterize_pn_as_polylines(nodes, faces, pattern, res: int = 10):
    polylines = []
    visited = set()
    for F in faces:
        for E in pattern:
            EF = [F[e] for e in E]
            if (k := utils.keyify(EF)) in visited: continue
            line = BezierCurve([nodes[_i] for _i in EF]).as_polyline(res)
            polylines.append(line)
            visited.add(k)
    return merge(polylines)

###### ORDER 2 ######

def evaluate_P2(A,B,C,AB,BC,CA, u,v):
    l1,l2,l3 = 1-u-v, u, v
    return l1*l1*A + l2*l2*B + l3*l3*C + 2*l1*l2*AB + 2*l2*l3*BC + 2*l1*l3*CA

class P2Triangulation(_PnTriangulation):

    def __init__(self, nodes: np.ndarray, faces: np.ndarray):
        """
        v
        ^
        |
        2
        |`\ 
        |  `\ 
        5    `4 
        |      `\ 
        |        `\ 
        0-----3----1  -> u


        Args:
            nodes (np.ndarray): array of size V*3 storing nodes' coordinates.
            faces (np.ndarray): array of size F*6 storing face indices of the mesh. Faces are indexes using gmsh convention (see diagram above)
        """
        super().__init__(nodes, faces, 2)
        self.connectivity = P2Triangulation._Connectivity(self)

    @property
    def node_order(self):
        if not self.nodes.has_attribute("order"):
            # Compute node order
            node_order = self.nodes.create_attribute("order",int, dense=True)
            for A0,A1,A2,A3,A4,A5 in self.faces:
                node_order[A0] = 1
                node_order[A1] = 1
                node_order[A2] = 1
                node_order[A3] = 2
                node_order[A4] = 2
                node_order[A5] = 2
        return self.nodes.get_attribute("order")

    @classmethod
    def from_P1_mesh(cls, mesh: SurfaceMesh):
        """Generates an order 2 mesh from a simple (first order) triangular mesh.

        Splits all edges into two and creates flat 2nd order Bezier triangles for each face.

        Args:
            mesh (SurfaceMesh): input triangle mesh

        Returns:
            P2Triangulation: a flat order 2 Bezier mesh.
        """
        nV = len(mesh.vertices)
        V = list(mesh.vertices)
        F = []
        for A,B in mesh.edges:
            pA, pB = V[A], V[B]
            pAB = (pA+pB)/2
            V.append(pAB)
        for A,B,C in mesh.faces:
            eAB = mesh.connectivity.edge_id(A,B)
            eBC = mesh.connectivity.edge_id(B,C)
            eCA = mesh.connectivity.edge_id(C,A)
            F.append((A,B,C, nV+eAB, nV+eBC, nV+eCA))
        return cls(np.array(V),np.array(F))

    def rasterize(self, res:int = 10, only_curves: bool = False):
        """
        Rasterize a P2 mesh as a P1 mesh for lazy rendering 
            
        Args:
            V (np.ndarray): Array of vertex positions (size Nx3)
            F (np.ndarray): Array of face indices (size Mx6). Faces are indexed in order (A,B,C,AB,BC,CA)
            res (int, optional): Resolution of each P2 element. Defaults to 10.
            only_curves (bool, optional): if set to True, will only rasterize edges as Bezier curve and not draw the surface. Defaults to False.

        Returns:
            SurfaceMesh | Polyline: a mesh approximating the P2 mesh
        """
        if only_curves:
            return rasterize_pn_as_polylines(self.nodes, self.faces, [(0,3,1), (1,4,2), (2,5,0)], res)
        else:
            return self._rasterize_as_surface(res)
        
    
    def _rasterize_as_surface(self, res=10):
        out = RawMeshData()

        def ij_to_p(i,j):
            return i*res - i*(i-1)//2 + j

        for triP2 in self.faces:
            pA,pB,pC,pAB,pBC,pCA = (self.nodes[_x] for _x in triP2)
            paramU = np.linspace(0,1,res)
            paramV = np.linspace(0,1,res)
            f_ind = len(out.vertices)

            # evaluate vertices
            for i in range(res):
                for j in range(res-i):
                    pt = Vec(evaluate_P2(pA,pB,pC,pAB,pBC,pCA, paramU[i], paramV[j]))
                    if pt.size==2:
                        pt = Vec(pt.x, pt.y, 0.)
                    out.vertices.append(pt)
                    
            # add faces
            for i in range(res-1):
                for j in range(res-1-i):
                    out.faces.append([f_ind+x for x in (ij_to_p(i,j), ij_to_p(i+1,j), ij_to_p(i,j+1))])
                    if j<res-2-i:
                        out.faces.append([f_ind+x for x in (ij_to_p(i+1,j), ij_to_p(i+1,j+1), ij_to_p(i,j+1))])
        return SurfaceMesh(out)
 
    class _Connectivity:

        def __init__(self, mesh):
            self.master = mesh
            self._connP1 : dict = None
            self._connP2 : dict = None
            self._middle : dict = None
        
        def _build(self):
            self._connP1 = dict()
            self._connP2 = dict()
            self._middle = dict()

            for A0,A1,A2,A3,A4,A5 in self.master.faces:
                self._connP1[A0] = self._connP1.get(A0,set()).union({A1,A2})
                self._connP1[A1] = self._connP1.get(A1,set()).union({A0,A2})
                self._connP1[A2] = self._connP1.get(A2,set()).union({A0,A1})

                self._middle[utils.keyify(A0,A1)] = A3
                self._middle[utils.keyify(A1,A2)] = A4
                self._middle[utils.keyify(A2,A0)] = A5
                
                self._connP2[A0] = self._connP2.get(A0, set()).union({A3,A5})
                self._connP2[A1] = self._connP2.get(A1, set()).union({A3,A4})
                self._connP2[A2] = self._connP2.get(A2, set()).union({A4,A5})
                self._connP2[A3] = [A0,A1]
                self._connP2[A4] = [A1,A2]
                self._connP2[A5] = [A0,A2]

        def vertex_to_vP1(self,v):
            if self._connP1 is None: self._build()
            return self._connP1.get(v, [])
        
        def vertex_to_vP2(self,v):
            if self._connP2 is None: self._build()
            return self._connP2.get(v,[])
        
        def middle_of_edge(self,a,b):
            if self._middle is None: self._build()
            return self._middle.get(utils.keyify(a,b),None)
        

###### ORDER 3 ######

def evaluate_P3(A,B,C,AAB, ABB, BBC, BCC, CCA, CAA, ABC, u,v):
    l1,l2,l3 = 1-u-v, u, v
    res  = l1*l1*l1*A + l2*l2*l2*B + l3*l3*l3*C
    res += 3*l1*l1*l2*AAB + 3*l2*l2*l3*BBC + 3*l3*l1*l1*CAA
    res += 3*l1*l2*l2*ABB + 3*l3*l3*l2*BCC + 3*l3*l3*l1*CCA
    res += 6*l1*l2*l3*ABC
    return res

class P3Triangulation(_PnTriangulation):

    def __init__(self, nodes: np.ndarray, faces: np.ndarray):
        """
        v
        ^
        |
        2
        | \ 
        7   6
        |     \ 
        8   9   5
        |         \ 
        0---3---4---1 -> u


        Args:
            nodes (np.ndarray): array of size V*3 storing nodes' coordinates.
            faces (np.ndarray): array of size F*10 storing face indices of the mesh. Faces are indexes using gmsh convention (see diagram above)
        """
        super().__init__(nodes, faces, 3)

    
    @classmethod
    def from_P1_mesh(cls, mesh: SurfaceMesh, curve: bool = True):
        """Generates an order 3 mesh from a simple (first order) triangular mesh.

        Args:
            mesh (SurfaceMesh): input triangular mesh (order 1)
            curve (bool, optional): if True, will apply algorithm from [1] to smooth the surface. Otherwise, will simply keep faces flat. Defaults to True.

        References:
            [1] Curved PN triangles, Vlachos et al. (2001)

        Returns:
            P3Triangulation: An order 3 Bezier triangulation
        """
                
        V = [p for p in mesh.vertices]
        F = []
        sep1 = len(mesh.vertices)
        inds = dict()
        for ie,(A,B) in enumerate(mesh.edges):
            pA, pB = mesh.vertices[A], mesh.vertices[B]
            pAAB = (2*pA+pB)/3
            pABB = (pA+2*pB)/3
            inds[(A,B)] = sep1+2*ie
            inds[(B,A)] = sep1+2*ie+1
            V += [pAAB,pABB]
        sep2 = len(V)
        for iF,(A,B,C) in enumerate(mesh.faces):
            pA, pB, pC = mesh.vertices[A], mesh.vertices[B], mesh.vertices[C]
            F.append((A,B,C, 
                inds[(A,B)], inds[(B,A)], 
                inds[(B,C)], inds[(C,B)], 
                inds[(C,A)], inds[(A,C)], 
            sep2+iF))
            V.append((pA+pB+pC)/3)

        V,F = np.array(V), np.array(F)
        if not curve: return cls(V,F)
        normals = vertex_normals(mesh, persistent=False)
        visited = np.zeros(V.shape[0], dtype=bool)
        for iA,iB,iC,iAAB,iABB, iBBC, iBCC, iCCA, iCAA, iABC in F:
            pA,pB,pC = V[iA], V[iB], V[iC]
            nA,nB,nC = normals[iA], normals[iB], normals[iC]
            if not visited[iAAB] or not visited[iABB]:
                visited[iAAB] = True
                visited[iABB] = True
                V[iAAB] = (2*pA + pB - np.dot(pB-pA, nA)*nA)/3
                V[iABB] = (2*pB + pA - np.dot(pA-pB, nB)*nB)/3
            if not visited[iBBC] or not visited[iBCC]:
                visited[iBBC] = True
                visited[iBCC] = True
                V[iBBC] = (2*pB + pC - np.dot(pC-pB, nB)*nB)/3
                V[iBCC] = (2*pC + pB - np.dot(pB-pC, nC)*nC)/3
            if not visited[iCCA] or not visited[iCAA]:
                visited[iCCA] = True
                visited[iCAA] = True
                V[iCCA] = (2*pC + pA - np.dot(pA-pC, nC)*nC)/3
                V[iCAA] = (2*pA + pC - np.dot(pC-pA, nA)*nA)/3
            e = (V[iAAB] + V[iABB] + V[iBBC] + V[iBCC] + V[iCCA] + V[iCAA])/6
            b = (pA+pB+pC)/3
            V[iABC] = e + (e-b)/2 
        return cls(V,F)

    def rasterize(self, res: int = 10, only_curves: bool = False):
        """
        Args:
            V (np.ndarray): Array of vertex positions (size Nx3)
            F (np.ndarray): Array of face indices (size Mx10)
            res (int, optional): Resolution of each P3 element. Defaults to 10.
            only_curves (bool, optional): if set to True, will only rasterize edges as Bezier curve and not draw the surface. Defaults to False.
            
        Returns:
            SurfaceMesh | Polyline: a mesh approximating the P3 mesh
        """
        if only_curves:
            return rasterize_pn_as_polylines(self.nodes, self.faces, [(0,3,4,1), (1,5,6,2), (2,7,8,0)], res)
        else:
            return self._rasterize_as_surface(res)

    def _rasterize_as_surface(self, res: int = 10):
        out = RawMeshData()

        def ij_to_p(i,j):
            return i*res - i*(i-1)//2 + j

        for triP3 in self.faces:
            pA,pB,pC,pAAB,pABB, pBBC, pBCC, pCCA,pCAA,pABC = (self.nodes[_x] for _x in triP3)
            paramU = np.linspace(0,1,res)
            paramV = np.linspace(0,1,res)
            f_ind = len(out.vertices)

            # evaluate vertices
            for i in range(res):
                for j in range(res-i):
                    out.vertices.append(evaluate_P3(pA,pB,pC,pAAB,pABB, pBBC, pBCC, pCCA, pCAA,pABC, paramU[i], paramV[j]))
                    
            # add faces
            for i in range(res-1):
                for j in range(res-1-i):
                    out.faces.append([f_ind+x for x in (ij_to_p(i,j), ij_to_p(i+1,j), ij_to_p(i,j+1))])
                    if j<res-2-i:
                        out.faces.append([f_ind+x for x in (ij_to_p(i+1,j), ij_to_p(i+1,j+1), ij_to_p(i,j+1))])
        return SurfaceMesh(out)