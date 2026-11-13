import numpy as np
from .datatypes import *
from .mesh import from_arrays
from ..utils import keyify
from typing import Iterable

### Polyline Subdivision ###

@allowed_mesh_types(PolyLine)
def split_edges(polyline : PolyLine, edge_inds: Iterable = None) -> PolyLine:
    """Computes a new mesh where edge indices given in edge_inds have been split in half at their middle points.

    Note:
        This function erases any attributes on edges and vertices, as well as the computed connectivity.

    Args:
        mesh (PolyLine): the input polyline
        edge_ind (Iterable, optional): indices of the edges to split. If not provided, will split all the edges. Defaults to None.

    Returns:
        PolyLine: the processed Polyline
    """
    raw = RawMeshData()
    raw.vertices = [_v for _v in polyline.vertices]
    edge_inds = set(edge_inds) if edge_inds is not None else None
    for e,(A,B) in enumerate(polyline.edges):
        if edge_inds is None or e in edge_inds:
            C = (polyline.vertices[A] + polyline.vertices[B])/2
            raw.vertices.append(C)
            raw.edges += [ (A, len(polyline.vertices)), (len(polyline.vertices), B)] # keep global orientation
        else:
            raw.edges.append([A,B])
    return raw.instantiate()


### Surface Subdivision ###

@allowed_mesh_types(SurfaceMesh)
def loop_subdivision(mesh : SurfaceMesh):
    """Subdivides triangles of a mesh in 4 triangles by splitting along middle of edges. Fails if the mesh is not triangular.

    References:
        [https://en.wikipedia.org/wiki/Loop_subdivision_surface](https://en.wikipedia.org/wiki/Loop_subdivision_surface)
    """
    if not mesh.is_triangular():
        raise Exception("Loop subdivision only applies to triangular meshes. Apply triangulation first")

    ## Build all vertices at edge midpoints
    raw = RawMeshData()
    raw.vertices = [_v for _v in mesh.vertices]
    half = dict()
    for (A,B) in mesh.edges:
        C = len(raw.vertices)
        pC = (mesh.vertices[A] + mesh.vertices[B])/2
        raw.vertices.append(pC)
        half[keyify(A,B)]=C

    for (A,B,C) in mesh.faces:
        mAB = half[keyify(A,B)]
        mBC = half[keyify(B,C)]
        mCA = half[keyify(C,A)]
        for new_tri in [
            (mAB,mBC,mCA),
            (A,mAB,mCA),
            (B,mBC,mAB),
            (C,mCA,mBC)
        ]:
            raw.faces.append(new_tri)
    return raw.instantiate()



@allowed_mesh_types(SurfaceMesh)
def triangulate(mesh : SurfaceMesh) -> SurfaceMesh:      
    """Triangulates all faces of a mesh.

    Args:
        mesh (SurfaceMesh): input surface mesh

    Returns:
        SurfaceMesh: a mesh with the same vertices but where all faces are triangles
    """
    if mesh.is_triangular(): return mesh
    if mesh.is_quad():
        F = mesh.faces._data
        tri_faces = np.concatenate((F[:,[0,1,3]], F[:,[1,2,3]]))
        return from_arrays(mesh.vertices._data, F=tri_faces)
    else:
        raise Exception("Should not happen.")


@allowed_mesh_types(SurfaceMesh)
def split_faces_as_fan(mesh : SurfaceMesh, face_inds: Iterable = None) -> SurfaceMesh:
    """
    Adds a vertex at the barycenter of face 'face_id' and create a fan of triangles that replaces the face

    Parameters:
        face_id (Iterable, optional): The indices of faces to split. If not provided, will split all the faces. Defaults to None.
    """
    new_vertices = [_v for _v in mesh.vertices]
    new_faces = []
    face_inds = set(face_inds) if face_inds is not None else None
    if face_inds is not None and not mesh.is_triangular():
        raise Exception("Cannot split only a subset of faces on a non-triangular mesh. This will result in an irregular mesh which is not supported.")
    for iF,F in enumerate(mesh.faces):
        if face_inds is not None and iF not in face_inds:
            new_faces.append(F) 
            continue
        nF = len(F)
        bary =  np.mean(mesh.vertices[F],axis=0)
        P = len(new_vertices)
        new_faces += [(F[i],F[(i+1)%nF],P) for i in range(nF)]
        new_vertices.append(bary)
    return from_arrays(new_vertices, F=new_faces)


@allowed_mesh_types(SurfaceMesh)
def subdivide_triangles_3quads(mesh : SurfaceMesh) -> SurfaceMesh:
    """Subdivides triangles of a mesh in 3 quads by adding a point at the barycenter and three middles of edges.
    """
    if not mesh.is_triangular():
        raise Exception("Cannot apply this function onto meshes that are not triangular")
    
    new_vertices = [_v for _v in mesh.vertices]

    # create edge middle and face barycenters
    half = dict()
    for e,(A,B) in enumerate(mesh.edges):
        pmid = (mesh.vertices[A] + mesh.vertices[B])/2
        half[keyify(A,B)]=len(new_vertices)
        new_vertices.append(pmid)

    bary = dict()
    for iF,(A,B,C) in enumerate(mesh.faces):
        bary[iF] = len(new_vertices)
        new_vertices.append((mesh.vertices[A] + mesh.vertices[B] + mesh.vertices[C])/3)

    new_faces = []
    for f,(A,B,C) in enumerate(mesh.faces):
        mAB = half[keyify(A,B)]
        mBC = half[keyify(B,C)]
        mCA = half[keyify(C,A)]
        S = bary[f]
        new_faces += [
            [A, mAB, S, mCA],
            [B, mBC, S, mAB],
            [C, mCA, S, mBC],
        ]
    return from_arrays(new_vertices, F=new_faces)

@allowed_mesh_types(SurfaceMesh)
def subdivide_triangles_6(mesh : SurfaceMesh) -> SurfaceMesh:
    """Subdivides triangles of a mesh in 6 triangles by adding a point at the barycenter and three middles of edges.
        If the mesh is not triangulated, will triangulate the mesh first.
        
        Same operation as subdividing into 3 quads and then splitting the quads along the corner-barycenter diagonal.
    """
    return triangulate(subdivide_triangles_3quads(mesh))

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
        return split_faces_as_fan(mesh, pb_faces)
    return mesh