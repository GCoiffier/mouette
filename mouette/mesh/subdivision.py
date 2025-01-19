from .datatypes import *
from .mesh_data import RawMeshData
from .mesh import _instanciate_raw_mesh_data
from ..geometry import Vec
from ..utils import keyify, Logger

### Polyline Subdivision ###

@allowed_mesh_types(PolyLine)
def split_edge(polyline : PolyLine, edge_ind: int) -> PolyLine:
    """Split an edge of a polyline in half by adding a new vertex

    Note:
        Polylines do not need a specific subdivision class to be used in a `with` block since their connectivity is simpler. 
        Connectivity is cleared after processing.

    Args:
        mesh (PolyLine): the input polyline
        edge_ind (int): index of the edge to split

    Returns:
        PolyLine: the processed Polyline
    """
    A,B = polyline.edges[edge_ind]
    C = len(polyline.vertices)
    pC = (polyline.vertices[A]+polyline.vertices[B])/2
    polyline.vertices.append(pC)
    polyline.edges[edge_ind] += keyify(A,C)
    polyline.edges.append(keyify(B,C))
    polyline.connectivity.clear()
    return polyline


### Surface Subdivision ###

class SurfaceSubdivision(Logger):

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, mesh : SurfaceMesh, verbose:bool = False):
        super().__init__("SurfaceSubdivision", verbose)
        self.mesh = mesh

    def __enter__(self):
        self.mesh = RawMeshData(self.mesh)
        self.mesh.face_corners.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mesh.prepare()
        self.mesh = _instanciate_raw_mesh_data(self.mesh, 2)

    @allowed_mesh_types(SurfaceMesh)
    def triangulate_face(self, face_id: int) :
        """Triangulates the face "face_id"

        Parameters:
            face_id (int): the face to triangulate
        """
        F = self.mesh.faces[face_id]
        if len(F)<4: return # nothing to do
        elif len(F)==4:
            A,B,C,D = self.mesh.faces[face_id]
            self.mesh.faces[face_id] = [A,B,D]
            self.mesh.faces.append([B,C,D])
        else:
            self.split_face_as_fan(face_id)

    def split_face_as_fan(self,  face_id: int) :
        """
        Adds a vertex at the barycenter of face 'face_id' and create a fan of triangles that replaces the face

        Parameters:
            face_id (int): the face to split
        """
        
        f = self.mesh.faces[face_id]
        pV = sum([Vec(self.mesh.vertices[a]) for a in f ])/len(f) # barycenter
        iV = len(self.mesh.vertices)
        self.mesh.vertices.append(pV)
        self.mesh.faces[face_id] = [f[0], f[1], iV]
        nf=len(f)
        for k in range(1, nf):
            self.mesh.faces.append([f[k], f[(k+1)%nf], iV])
        for v in f:
            self.mesh.edges.append(keyify(v,iV))

    def triangulate(self):
        """Triangulates all faces of a mesh.
            Calls triangulate_face on every faces.
        """       
        for f in self.mesh.id_faces:
            if len(self.mesh.faces[f])!= 3 :
                self.triangulate_face(f)

    def loop_subdivision(self, n: int = 1):
        """Subdivides triangles of a mesh in 4 triangles by splitting along middle of edges.
            If the mesh is not triangulated, will triangulate the mesh first.

        References:
            [https://en.wikipedia.org/wiki/Loop_subdivision_surface](https://en.wikipedia.org/wiki/Loop_subdivision_surface)
        
        Parameters:
            n (int, optional): Number of times the subdivision is applied. Defaults to 1.
        """
        self.triangulate()

        for _ in range(n):
            newMeshData = RawMeshData()
            newMeshData.vertices += self.mesh.vertices
            # cut every edge in half
            half = dict()
            for (A,B) in self.mesh.edges:
                C = len(newMeshData.vertices)
                pC = (self.mesh.vertices[A] + self.mesh.vertices[B])/2
                newMeshData.vertices.append(pC)
                half[keyify(A,B)]=C
            new_edges = set()
            for f in self.mesh.id_faces:
                A,B,C = self.mesh.faces[f]
                mAB = half[keyify(A,B)]
                mBC = half[keyify(B,C)]
                mCA = half[keyify(C,A)]
                for new_tri in [
                    (mAB,mBC,mCA),
                    (A,mAB,mCA),
                    (B,mBC,mAB),
                    (C,mCA,mBC)
                ]:
                    newMeshData.faces.append(new_tri)
                for new_edge in [
                    (A,mAB),(mAB,B),(B,mBC),(mBC,C),(C,mCA),(mCA,A), (mAB,mBC),(mBC,mCA),(mCA,mAB)
                ]:
                    new_edges.add(keyify(new_edge))
            
            newMeshData.edges += list(new_edges)
            self.mesh = newMeshData

    @allowed_mesh_types(SurfaceMesh)
    def subdivide_triangles_6(self, repeat: int = 1) -> SurfaceMesh:
        """Subdivides triangles of a mesh in 6 triangles by adding a point at the barycenter and three middles of edges.
            If the mesh is not triangulated, will triangulate the mesh first.
            
            Same operation as subdividing into 3 quads and then splitting the quads along the corner-barycenter diagonal.

        Parameters:
            repeat (int, optional): number of successive subdivisions. Eventual first triangulation does not count. Defaults to 1.
        """
        for _ in range(repeat):
            self.subdivide_triangles_3quads()
            self.triangulate()

    @allowed_mesh_types(SurfaceMesh)
    def subdivide_triangles_3quads(self) -> SurfaceMesh:
        """Subdivides triangles of a mesh in 3 quads by adding a point at the barycenter and three middles of edges.
            If the mesh is not triangulated, will triangulate the mesh first.
        """
        self.triangulate()
        newMeshData = RawMeshData()
        newMeshData.vertices += self.mesh.vertices
        # cut every edge in half
        half = dict()
        for e in self.mesh.id_edges:
            A,B = self.mesh.edges[e]
            C = len(newMeshData.vertices)
            pC = (self.mesh.vertices[A] + self.mesh.vertices[B])/2
            newMeshData.vertices.append(pC)
            half[keyify(A,B)]=C

        bary = dict()
        for iF,F in enumerate(self.mesh.faces):
            pS = sum([self.mesh.vertices[u] for u in F])/3
            bary[iF] = len(newMeshData.vertices)
            newMeshData.vertices.append(pS)

        for f in self.mesh.id_faces:
            A,B,C = self.mesh.faces[f]
            mAB = half[keyify(A,B)]
            mBC = half[keyify(B,C)]
            mCA = half[keyify(C,A)]
            S = bary[f]
            
            for new_face in [
                [A, mAB, S, mCA],
                [B, mBC, S, mAB],
                [C, mCA, S, mBC],
            ]:
                newMeshData.faces.append(new_face)
        self.mesh = newMeshData

@allowed_mesh_types(SurfaceMesh)
def split_double_boundary_edges_triangles(mesh : SurfaceMesh) -> SurfaceMesh:
    """
    A triangle with double edge on the boundary can occur on the border in a case like :

    ```
        /\\ 
    ___/__\\___
    ```

    This function detects every occurrences of such a configuration and split the problematic
    triangle in three by adding a new vertex in the middle. Uses the SurfaceSubdivision class.

    Parameters:
        mesh (Mesh): the mesh (modified in place)

    Raises:
        Exception: Isolated vertex
        Raised when a vertex of the mesh has degree < 2, which mean that the mesh is not manifold

    Returns:
        SurfaceMesh: the modified input mesh
    """
        
    deg = [0]*len(mesh.vertices)
    for (a,b) in mesh.edges:
        deg[a] += 1
        deg[b] += 1

    pb_faces = []
    for i,f in enumerate(mesh.faces):
        for v in f:
            if deg[v]<2: 
                raise Exception("Isolated vertex")
            if deg[v]==2:
                pb_faces.append(i)
                break
    if pb_faces:
        with SurfaceSubdivision(mesh) as subdv:
            for f in pb_faces: # Triangulate face with a vertex in the middle
                subdv.split_face_as_fan(f)
    return mesh

### Volume Subdivision ###

class VolumeSubdivision(Logger):

    @allowed_mesh_types(VolumeMesh)
    def __init__(self, mesh : VolumeMesh, verbose:bool=False):
        super().__init__("VolumeSubdivision", verbose=verbose)
        self.mesh = mesh
        self.conn = None # connectivity

    def __enter__(self):
        self.conn = self.mesh.connectivity
        self.conn._compute_cell_adj()
        self.mesh = RawMeshData(self.mesh)
        self.mesh.face_corners.clear()
        self.mesh.cell_corners.clear()
        self.mesh.cell_faces.clear()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.mesh.prepare()
        self.mesh = _instanciate_raw_mesh_data(self.mesh, 3)

    def split_cell_as_fan(self, cell_id:int):
        """
        Adds a vertex at the barycenter of cell 'cell_id' and create a fan of tetrahedra that replaces the cell.
        If the cell 'cell_id' is not a tetrahedron, does nothing.

        Parameters:
            mesh (VolumeMesh): the input mesh
            cell_id (int): the (tetrahedral) cell to split

        Returns:
            (VolumeMesh): the modified input mesh
        """
        if len(self.mesh.cells[cell_id]) != 4 : return # cell is not a tet
        
        A,B,C,D = self.mesh.cells[cell_id] 
        pA,pB,pC,pD = (self.mesh.vertices[_v] for _v in (A,B,C,D))
        bary = 0.25*(pA+pB+pC+pD)
        ibary = len(self.mesh.vertices)
        self.mesh.vertices.append(bary)
        self.mesh.cells[cell_id] = (ibary,B,C,D)
        self.mesh.cells += [
            (A,ibary,C,D),
            (A,B,ibary,D),
            (A,B,C,ibary)
        ]

    def split_tet_from_face_center(self, face_id : int):
        """Split the triangle `face_id` into three triangles by adding a point at its barycenter,
        and split adjacent tetrahedra accordingly.

        Args:
            face_id (int): index of the face to split
        """
        f = self.mesh.faces[face_id]
        if len(f) != 3 : return
        A,B,C = f
        icenter = len(self.mesh.vertices)
        pcenter = sum([Vec(self.mesh.vertices[a]) for a in f ])/3 # barycenter
        self.mesh.vertices.append(pcenter)
        
        for c in self.conn.face_to_cells(face_id):
            iF = self.conn.in_cell_face_index(c,face_id)
            new_cells = []
            for i in range(4):
                if i==iF : continue # opposite point in tet from face
                cell = [_x for _x in self.mesh.cells[c]]
                cell[i] = icenter
                new_cells.append(cell)
            self.mesh.cells[c] = new_cells[0]
            self.mesh.cells.append(new_cells[1])
            self.mesh.cells.append(new_cells[2])
        
        self.mesh.faces[face_id] = [A,B,icenter]
        self.mesh.faces.append([icenter, B, C])
        self.mesh.faces.append([A, icenter, C])